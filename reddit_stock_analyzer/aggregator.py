"""High-level entry points: scrape, enrich, and rank in one call.

:class:`RedditTrendService` is the object a long-running app (such as
``fintwit-web``) should build once at startup and reuse — it owns the HTTP
session and the two loaded models. The module-level coroutines wrap it for
scripts and one-off calls.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Sequence

import httpx

from .analyzer import PostAnalyzer
from .client import RedditClient
from .config import DEFAULT_SUBREDDITS
from .models import AnalyzedPost, TrendReport
from .trends import compute_trends

logger = logging.getLogger(__name__)

#: Both a hot and a new pass: `hot` finds what got traction, `new` catches the
#: last few hours before the votes arrive. De-duplicated by submission id.
DEFAULT_SORTS: tuple[str, ...] = ("hot", "new")


class RedditTrendService:
    """Scrape finance subreddits and turn them into trend reports.

    Parameters
    ----------
    client : RedditClient, optional
        Defaults to a new client. Pass one built around an existing
        ``httpx.AsyncClient`` to share connection pooling.
    analyzer : PostAnalyzer, optional
        Defaults to a new analyzer over the shared ticker extractor.
    use_ai : bool, default True
        Only used when *analyzer* is omitted; see
        :class:`~reddit_stock_analyzer.recognizer.TickerExtractor`.

    Examples
    --------
    >>> service = RedditTrendService()                      # doctest: +SKIP
    >>> report = await service.trend_report(window_hours=12)  # doctest: +SKIP
    >>> [t.symbol for t in report.tickers[:3]]                # doctest: +SKIP
    ['NVDA', 'TSLA', 'SPY']
    """

    def __init__(
        self,
        client: RedditClient | None = None,
        analyzer: PostAnalyzer | None = None,
        *,
        use_ai: bool = True,
    ) -> None:
        self.client = client or RedditClient()
        self.analyzer = analyzer or PostAnalyzer(use_ai=use_ai)

    async def __aenter__(self) -> "RedditTrendService":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def scrape(
        self,
        subreddits: Sequence[str] | None = None,
        *,
        sorts: Sequence[str] = DEFAULT_SORTS,
        limit: int = 50,
        max_age_hours: float | None = 48.0,
        time_filter: str = "day",
        now: float | None = None,
        concurrency: int = 4,
    ) -> list[AnalyzedPost]:
        """Fetch recent posts and enrich them with tickers and sentiment.

        Returns
        -------
        list of AnalyzedPost
            Newest first. Empty when every subreddit request failed.
        """
        targets = list(subreddits) if subreddits else list(DEFAULT_SUBREDDITS)
        posts = await self.client.fetch_many(
            targets,
            sorts=sorts,
            limit=limit,
            time_filter=time_filter,
            max_age_hours=max_age_hours,
            now=now,
            concurrency=concurrency,
        )
        logger.info(
            "[reddit] fetched %s posts from %s subreddit(s)", len(posts), len(targets)
        )
        return await self.analyzer.analyze_async(posts)

    async def trend_report(
        self,
        subreddits: Sequence[str] | None = None,
        *,
        window_hours: float = 24.0,
        baseline_hours: float | None = None,
        sorts: Sequence[str] = DEFAULT_SORTS,
        limit: int = 50,
        time_filter: str = "day",
        now: float | None = None,
        concurrency: int = 4,
        **trend_kwargs: Any,
    ) -> TrendReport:
        """Scrape, enrich and rank in one call.

        The scrape window is automatically widened to
        ``window_hours + baseline_hours`` so the comparison window has posts
        in it — without that every ticker would look newly emerging.

        Other keyword arguments are forwarded to
        :func:`~reddit_stock_analyzer.trends.compute_trends`.
        """
        resolved_baseline = window_hours if baseline_hours is None else baseline_hours
        targets = list(subreddits) if subreddits else list(DEFAULT_SUBREDDITS)

        posts = await self.scrape(
            targets,
            sorts=sorts,
            limit=limit,
            max_age_hours=window_hours + resolved_baseline,
            time_filter=time_filter,
            now=now,
            concurrency=concurrency,
        )
        return compute_trends(
            posts,
            now=now,
            window_hours=window_hours,
            baseline_hours=resolved_baseline,
            subreddits=targets,
            **trend_kwargs,
        )

    async def subreddit_overview(
        self,
        subreddit: str,
        *,
        limit: int = 50,
        sort: str = "hot",
        max_age_hours: float | None = None,
        top_tickers: int = 10,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Sentiment and ticker snapshot for a single subreddit.

        Cheaper than :meth:`trend_report`: one listing, no baseline window, no
        momentum. Use it for a "what is r/stocks talking about right now" panel.
        """
        posts = await self.client.fetch_posts(
            subreddit,
            sort=sort,
            limit=limit,
            max_age_hours=max_age_hours,
            now=now,
        )
        analyzed = await self.analyzer.analyze_async(posts)

        ticker_counts: Counter[str] = Counter()
        for post in analyzed:
            ticker_counts.update(post.tickers)
        sentiment_counts = Counter(post.sentiment for post in analyzed)
        sentiment_score = (
            round(sum(p.sentiment_score for p in analyzed) / len(analyzed), 4)
            if analyzed
            else 0.0
        )

        return {
            "subreddit": subreddit,
            "sample_size": len(analyzed),
            "overall_mood": (
                sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"
            ),
            "sentiment_score": sentiment_score,
            "sentiment_breakdown": dict(sentiment_counts),
            "top_tickers": dict(ticker_counts.most_common(top_tickers)),
            "posts": [post.to_dict() for post in analyzed],
        }

    async def close(self) -> None:
        """Close the underlying Reddit client."""
        await self.client.close()


async def get_trend_report(
    subreddits: Sequence[str] | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    use_ai: bool = True,
    **kwargs: Any,
) -> TrendReport:
    """One-shot :meth:`RedditTrendService.trend_report`.

    Builds and tears down a service per call, so prefer the class in a
    long-running process — this reloads nothing only because the ticker
    extractor is process-wide, but the HTTP session is rebuilt each time.
    """
    service = RedditTrendService(client=RedditClient(http_client), use_ai=use_ai)
    try:
        return await service.trend_report(subreddits, **kwargs)
    finally:
        await service.close()


async def get_trending_tickers(
    subreddits: Sequence[str] | None = None,
    *,
    top_n: int = 10,
    http_client: httpx.AsyncClient | None = None,
    use_ai: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Return the ranked tickers of a trend report as JSON-ready dicts."""
    report = await get_trend_report(
        subreddits, top_n=top_n, http_client=http_client, use_ai=use_ai, **kwargs
    )
    return [ticker.to_dict() for ticker in report.tickers]


async def get_subreddit_overview(
    subreddit: str,
    limit: int = 50,
    *,
    http_client: httpx.AsyncClient | None = None,
    use_ai: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """One-shot :meth:`RedditTrendService.subreddit_overview`."""
    service = RedditTrendService(client=RedditClient(http_client), use_ai=use_ai)
    try:
        return await service.subreddit_overview(subreddit, limit=limit, **kwargs)
    finally:
        await service.close()
