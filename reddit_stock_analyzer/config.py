"""Subreddit catalogue and runtime configuration.

The catalogue is grouped by category so a caller can ask for "everything
retail traders shout about" without hard-coding subreddit names, while still
being free to pass an explicit list.
"""

from __future__ import annotations

import os
import re

SUBREDDIT_NAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")

#: Finance subreddits grouped by the kind of discussion they carry. The
#: grouping matters for interpretation: a ticker trending in ``retail`` is a
#: hype signal, the same ticker trending in ``research`` is a thesis signal.
SUBREDDIT_CATEGORIES: dict[str, tuple[str, ...]] = {
    # High-volume, meme-heavy retail flow. Noisy but fastest moving.
    "retail": (
        "wallstreetbets",
        "smallstreetbets",
        "WallStreetbetsELITE",
        "Shortsqueeze",
        "pennystocks",
    ),
    # General equity discussion, slower and more moderated.
    "stocks": (
        "stocks",
        "StockMarket",
        "investing",
        "dividends",
    ),
    # Longer-form fundamental analysis.
    "research": (
        "SecurityAnalysis",
        "ValueInvesting",
        "stocks",
    ),
    # Active traders: shorter horizons, more ticker mentions per post.
    "trading": (
        "Daytrading",
        "swingtrading",
        "options",
        "thetagang",
        "algotrading",
    ),
    # Rates, macro and general finance. Few tickers, useful for mood.
    "macro": (
        "economics",
        "finance",
        "economy",
    ),
    "crypto": (
        "CryptoCurrency",
        "CryptoMarkets",
        "satoshistreetbets",
        "ethtrader",
        "Bitcoin",
    ),
}

#: A balanced default: enough retail volume to see hype forming, enough
#: general discussion to tell hype from consensus, without 20 API round trips.
DEFAULT_SUBREDDITS: tuple[str, ...] = (
    "wallstreetbets",
    "stocks",
    "StockMarket",
    "investing",
    "options",
    "pennystocks",
)

#: Sort orders the Reddit listing endpoints accept.
VALID_SORTS: frozenset[str] = frozenset(
    {"hot", "new", "top", "rising", "controversial"}
)

#: ``top``/``controversial`` accept an extra time filter.
VALID_TIME_FILTERS: frozenset[str] = frozenset(
    {"hour", "day", "week", "month", "year", "all"}
)

#: Reddit rejects listing requests above 100 items per call.
MAX_LIMIT = 100

USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT_OVERRIDE",
    "reddit-stock-analyzer/0.1 (+https://github.com/StephanAkkerman/reddit-stock-analyzer)",
)


def is_valid_subreddit_name(name: str) -> bool:
    """Return ``True`` when *name* is a syntactically valid subreddit name."""
    return bool(SUBREDDIT_NAME_RE.match(name or ""))


def subreddits_for(*categories: str) -> tuple[str, ...]:
    """Return the de-duplicated subreddits of one or more categories.

    Parameters
    ----------
    *categories : str
        Keys of :data:`SUBREDDIT_CATEGORIES`. Passing none returns
        :data:`DEFAULT_SUBREDDITS`.

    Returns
    -------
    tuple of str
        Subreddit names in declaration order, without duplicates.

    Raises
    ------
    KeyError
        If a category is unknown.
    """
    if not categories:
        return DEFAULT_SUBREDDITS

    seen: dict[str, None] = {}
    for category in categories:
        if category not in SUBREDDIT_CATEGORIES:
            raise KeyError(
                f"Unknown subreddit category {category!r}. "
                f"Known categories: {sorted(SUBREDDIT_CATEGORIES)}"
            )
        for name in SUBREDDIT_CATEGORIES[category]:
            seen.setdefault(name, None)
    return tuple(seen)


def all_subreddits() -> tuple[str, ...]:
    """Return every subreddit in the catalogue, de-duplicated."""
    return subreddits_for(*SUBREDDIT_CATEGORIES)


def reddit_credentials_from_env() -> dict[str, str] | None:
    """Collect Reddit API credentials from the environment.

    Both the names used by this project's ``.env.example`` and the ones
    ``fintwit-web`` already sets are accepted, so a single ``.env`` works for
    both.

    Returns
    -------
    dict or None
        Keyword arguments for ``asyncpraw.Reddit``, or ``None`` when no
        client id/secret pair is configured.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID") or os.getenv("REDDIT_PERSONAL_USE")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET") or os.getenv("REDDIT_SECRET")
    if not client_id or not client_secret:
        return None

    creds = {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": (
            os.getenv("REDDIT_USER_AGENT") or os.getenv("REDDIT_APP_NAME") or USER_AGENT
        ),
    }

    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    if username and password:
        creds["username"] = username
        creds["password"] = password
    return creds
