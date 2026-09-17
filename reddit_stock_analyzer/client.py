"""Async Reddit listing client.

Two backends, chosen automatically:

* ``asyncpraw`` when Reddit API credentials are configured and the package is
  installed. Higher rate limits, authenticated.
* The public ``https://www.reddit.com/r/<sub>/<sort>.json`` endpoints via
  ``httpx`` otherwise, so the library is usable with no credentials at all.

Both produce identical :class:`~reddit_stock_analyzer.models.RedditPost`
objects, so nothing downstream needs to know which one ran.
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from contextlib import suppress
from typing import Any, Iterable, Sequence

import httpx

from .config import (
    MAX_LIMIT,
    USER_AGENT,
    VALID_SORTS,
    VALID_TIME_FILTERS,
    is_valid_subreddit_name,
    reddit_credentials_from_env,
)
from .models import RedditPost

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp")

#: Reddit answers 429 under load; back off rather than hammering it.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    """Collect direct image URLs from a submission payload."""
    if data.get("is_self"):
        return []

    urls: list[str] = []
    url = html.unescape(str(data.get("url") or ""))

    if url.lower().endswith(_IMAGE_SUFFIXES):
        urls.append(url)
    elif data.get("is_gallery"):
        for item in (data.get("media_metadata") or {}).values():
            source = (item or {}).get("s") or {}
            if source.get("u"):
                urls.append(html.unescape(str(source["u"])))
    else:
        previews = (data.get("preview") or {}).get("images") or []
        for preview in previews:
            source = (preview or {}).get("source") or {}
            if source.get("url"):
                urls.append(html.unescape(str(source["url"])))
    return urls


def normalize_submission(data: dict[str, Any], subreddit: str = "") -> RedditPost:
    """Build a :class:`RedditPost` from a raw submission mapping.

    Accepts both the JSON API shape and the ``vars()``-style mapping produced
    from an ``asyncpraw`` submission.
    """
    permalink = str(data.get("permalink") or "")
    url = str(data.get("url") or "")
    post_url = (
        f"https://www.reddit.com{permalink}" if permalink.startswith("/") else url
    )

    return RedditPost(
        id=str(data.get("id") or ""),
        subreddit=str(data.get("subreddit") or subreddit),
        title=html.unescape(str(data.get("title") or "")),
        selftext=html.unescape(str(data.get("selftext") or "")),
        author=str(data.get("author") or ""),
        score=_coerce_int(data.get("score")),
        num_comments=_coerce_int(data.get("num_comments")),
        upvote_ratio=_coerce_float(data.get("upvote_ratio")),
        created_utc=_coerce_float(data.get("created_utc")),
        url=post_url,
        permalink=permalink,
        flair=str(data.get("link_flair_text") or ""),
        is_self=bool(data.get("is_self", True)),
        over_18=bool(data.get("over_18", False)),
        stickied=bool(data.get("stickied", False)),
        image_urls=_extract_image_urls(data),
    )


def _submission_to_mapping(submission: Any) -> dict[str, Any]:
    """Read the fields we need off an ``asyncpraw`` submission object."""
    return {
        "id": getattr(submission, "id", ""),
        "subreddit": str(getattr(submission, "subreddit", "")),
        "title": getattr(submission, "title", ""),
        "selftext": getattr(submission, "selftext", ""),
        "author": str(getattr(submission, "author", "") or ""),
        "score": getattr(submission, "score", 0),
        "num_comments": getattr(submission, "num_comments", 0),
        "upvote_ratio": getattr(submission, "upvote_ratio", 0.0),
        "created_utc": getattr(submission, "created_utc", 0.0),
        "permalink": getattr(submission, "permalink", ""),
        "url": getattr(submission, "url", ""),
        "link_flair_text": getattr(submission, "link_flair_text", ""),
        "is_self": getattr(submission, "is_self", True),
        "over_18": getattr(submission, "over_18", False),
        "stickied": getattr(submission, "stickied", False),
        "is_gallery": getattr(submission, "is_gallery", False),
        "media_metadata": getattr(submission, "media_metadata", None),
        "preview": getattr(submission, "preview", None),
    }


class RedditClient:
    """Fetch submissions from finance subreddits.

    Parameters
    ----------
    client : httpx.AsyncClient, optional
        Reuse an existing client (``fintwit-web`` passes its shared one). When
        omitted one is created lazily and closed by :meth:`close`.
    use_praw : bool, default True
        Allow the authenticated ``asyncpraw`` backend. Set ``False`` to force
        the anonymous JSON backend.
    max_retries : int, default 2
        Extra attempts for a rate-limited or 5xx JSON response.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        use_praw: bool = True,
        max_retries: int = 2,
        timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._use_praw = use_praw
        self._max_retries = max(0, int(max_retries))
        self._timeout = timeout
        self._reddit: Any = None
        self._praw_unavailable = False

    async def __aenter__(self) -> "RedditClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- backends ---------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

    async def _praw(self) -> Any:
        """Return an ``asyncpraw.Reddit`` instance, or ``None`` if unusable."""
        if not self._use_praw or self._praw_unavailable:
            return None
        if self._reddit is not None:
            return self._reddit

        creds = reddit_credentials_from_env()
        if creds is None:
            logger.debug("[reddit] no credentials configured, using anonymous JSON API")
            self._praw_unavailable = True
            return None
        try:
            import asyncpraw  # noqa: PLC0415 - optional dependency
        except ImportError:
            logger.debug("[reddit] asyncpraw not installed, using anonymous JSON API")
            self._praw_unavailable = True
            return None

        self._reddit = asyncpraw.Reddit(**creds)
        return self._reddit

    async def _fetch_praw(
        self, subreddit: str, sort: str, limit: int, time_filter: str
    ) -> list[RedditPost] | None:
        reddit = await self._praw()
        if reddit is None:
            return None
        try:
            sub = await reddit.subreddit(subreddit)
            listing = getattr(sub, sort)
            kwargs = (
                {"limit": limit, "time_filter": time_filter}
                if sort in {"top", "controversial"}
                else {"limit": limit}
            )
            return [
                normalize_submission(_submission_to_mapping(s), subreddit)
                async for s in listing(**kwargs)
            ]
        except Exception as exc:  # asyncpraw raises a wide family of errors
            logger.warning(
                "[reddit] asyncpraw fetch failed for r/%s (%r); "
                "falling back to the JSON API",
                subreddit,
                exc,
            )
            return None

    async def _fetch_json(
        self, subreddit: str, sort: str, limit: int, time_filter: str
    ) -> list[RedditPost]:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params: dict[str, Any] = {"limit": limit, "raw_json": 1}
        if sort in {"top", "controversial"}:
            params["t"] = time_filter

        client = self._http()
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                )
            except httpx.RequestError as exc:
                logger.warning("[reddit] request error for r/%s: %r", subreddit, exc)
                if attempt >= self._max_retries:
                    return []
            else:
                if response.status_code == 200:
                    return self._parse_listing(response, subreddit)
                if (
                    response.status_code not in _RETRY_STATUS
                    or attempt >= self._max_retries
                ):
                    logger.warning(
                        "[reddit] r/%s/%s returned HTTP %s",
                        subreddit,
                        sort,
                        response.status_code,
                    )
                    return []
            await asyncio.sleep(delay)
            delay *= 2
        return []

    @staticmethod
    def _parse_listing(response: httpx.Response, subreddit: str) -> list[RedditPost]:
        try:
            payload = response.json()
        except ValueError:
            logger.warning("[reddit] r/%s returned a non-JSON body", subreddit)
            return []
        children = ((payload or {}).get("data") or {}).get("children") or []
        return [
            normalize_submission((child or {}).get("data") or {}, subreddit)
            for child in children
        ]

    # -- public API -------------------------------------------------------

    async def fetch_posts(
        self,
        subreddit: str,
        *,
        sort: str = "hot",
        limit: int = 50,
        time_filter: str = "day",
        max_age_hours: float | None = None,
        skip_stickied: bool = True,
        now: float | None = None,
    ) -> list[RedditPost]:
        """Fetch one listing page from a single subreddit.

        Parameters
        ----------
        subreddit : str
            Subreddit name without the ``r/`` prefix.
        sort : {'hot', 'new', 'top', 'rising', 'controversial'}
            Listing to read.
        limit : int
            Posts to request, clamped to 1..100 (Reddit's own ceiling).
        time_filter : {'hour', 'day', 'week', 'month', 'year', 'all'}
            Only meaningful for ``top`` and ``controversial``.
        max_age_hours : float, optional
            Drop posts older than this. This is the "recent posts" knob;
            listings such as ``hot`` happily return week-old submissions.
        skip_stickied : bool
            Drop pinned posts (daily threads, rules) which skew mention counts.
        now : float, optional
            Reference timestamp for ``max_age_hours``. Defaults to wall clock;
            tests pass a fixed value.

        Returns
        -------
        list of RedditPost
            Empty when the subreddit name is invalid or Reddit is unreachable.
        """
        if not is_valid_subreddit_name(subreddit):
            logger.warning("[reddit] invalid subreddit name: %r", subreddit)
            return []
        if sort not in VALID_SORTS:
            raise ValueError(f"sort must be one of {sorted(VALID_SORTS)}, got {sort!r}")
        if time_filter not in VALID_TIME_FILTERS:
            raise ValueError(
                f"time_filter must be one of {sorted(VALID_TIME_FILTERS)}, "
                f"got {time_filter!r}"
            )

        bounded = max(1, min(int(limit), MAX_LIMIT))
        posts = await self._fetch_praw(subreddit, sort, bounded, time_filter)
        if posts is None:
            posts = await self._fetch_json(subreddit, sort, bounded, time_filter)

        cutoff = None
        if max_age_hours is not None:
            reference = time.time() if now is None else now
            cutoff = reference - max_age_hours * 3600

        return [
            post
            for post in posts
            if post.id
            and not (skip_stickied and post.stickied)
            and (cutoff is None or post.created_utc >= cutoff)
        ]

    async def fetch_many(
        self,
        subreddits: Iterable[str],
        *,
        sorts: Sequence[str] = ("hot",),
        limit: int = 50,
        time_filter: str = "day",
        max_age_hours: float | None = None,
        skip_stickied: bool = True,
        now: float | None = None,
        concurrency: int = 4,
    ) -> list[RedditPost]:
        """Fetch several subreddits (and sorts) concurrently, de-duplicated.

        Combining sorts is what makes trend baselines work: ``new`` supplies
        the freshest posts, ``hot`` supplies the ones that actually got
        traction, and the union is de-duplicated by submission id.

        Parameters
        ----------
        concurrency : int
            Maximum simultaneous listing requests. Reddit throttles bursts, so
            this stays low by default.

        Returns
        -------
        list of RedditPost
            Sorted newest first.
        """
        targets = [(sub, sort) for sub in subreddits for sort in sorts]
        if not targets:
            return []

        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def one(sub: str, sort: str) -> list[RedditPost]:
            async with semaphore:
                return await self.fetch_posts(
                    sub,
                    sort=sort,
                    limit=limit,
                    time_filter=time_filter,
                    max_age_hours=max_age_hours,
                    skip_stickied=skip_stickied,
                    now=now,
                )

        results = await asyncio.gather(
            *(one(sub, sort) for sub, sort in targets), return_exceptions=True
        )

        merged: dict[str, RedditPost] = {}
        for (sub, sort), result in zip(targets, results):
            if isinstance(result, BaseException):
                logger.warning("[reddit] r/%s/%s failed: %r", sub, sort, result)
                continue
            for post in result:
                # Keep the copy with the higher score: `new` snapshots a post
                # before votes land, `hot` sees it later.
                existing = merged.get(post.id)
                if existing is None or post.score > existing.score:
                    merged[post.id] = post

        return sorted(merged.values(), key=lambda p: p.created_utc, reverse=True)

    async def close(self) -> None:
        """Release the asyncpraw session and any client we created ourselves."""
        if self._reddit is not None:
            with suppress(Exception):
                await self._reddit.close()
            self._reddit = None
        if self._client is not None and self._owns_client:
            with suppress(Exception):
                await self._client.aclose()
            self._client = None
