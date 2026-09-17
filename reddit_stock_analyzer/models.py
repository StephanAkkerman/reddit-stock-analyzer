"""Plain dataclasses shared across the package.

Deliberately dependency-free: consumers such as ``fintwit-web`` map these onto
their own Pydantic schemas, so pulling Pydantic in here would only add a
version constraint. Every model exposes :meth:`to_dict` returning JSON-safe
primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Sentiment = Literal["bullish", "bearish", "neutral"]


@dataclass(slots=True)
class RedditPost:
    """A single submission, normalised across the asyncpraw and JSON backends."""

    id: str
    subreddit: str
    title: str
    selftext: str = ""
    author: str = ""
    score: int = 0
    num_comments: int = 0
    upvote_ratio: float = 0.0
    created_utc: float = 0.0
    url: str = ""
    permalink: str = ""
    flair: str = ""
    is_self: bool = True
    over_18: bool = False
    stickied: bool = False
    image_urls: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Title and body joined, for ticker extraction and sentiment."""
        body = self.selftext.strip()
        return f"{self.title}\n\n{body}".strip() if body else self.title

    @property
    def engagement(self) -> int:
        """Upvotes plus comments, weighted towards discussion.

        Comments count double: a post with 40 comments generated far more
        conversation about its tickers than one with 40 silent upvotes.
        """
        return int(self.score) + 2 * int(self.num_comments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "title": self.title,
            "selftext": self.selftext,
            "author": self.author,
            "score": self.score,
            "num_comments": self.num_comments,
            "upvote_ratio": self.upvote_ratio,
            "created_utc": self.created_utc,
            "url": self.url,
            "permalink": self.permalink,
            "flair": self.flair,
            "is_self": self.is_self,
            "over_18": self.over_18,
            "stickied": self.stickied,
            "image_urls": list(self.image_urls),
        }


@dataclass(slots=True)
class AnalyzedPost:
    """A post enriched with recognised tickers and a sentiment label."""

    post: RedditPost
    tickers: list[str] = field(default_factory=list)
    sentiment: Sentiment = "neutral"
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    #: Signed sentiment per ticker, for posts that discuss several. A post
    #: reading "long NVDA, short INTC" is bullish and bearish at once, and one
    #: post-level label would attribute the wrong half to each. Only populated
    #: where the two differ; use :meth:`sentiment_for` rather than reading it.
    ticker_sentiment: dict[str, float] = field(default_factory=dict)

    def sentiment_for(self, symbol: str) -> float:
        """Signed sentiment towards *symbol*, falling back to the post's own."""
        return self.ticker_sentiment.get(symbol, self.sentiment_score)

    @property
    def id(self) -> str:
        return self.post.id

    @property
    def created_utc(self) -> float:
        return self.post.created_utc

    @property
    def subreddit(self) -> str:
        return self.post.subreddit

    def to_dict(self) -> dict[str, Any]:
        """Flatten post fields and analysis into one JSON-ready mapping."""
        return {
            **self.post.to_dict(),
            "tickers": list(self.tickers),
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "ticker_sentiment": dict(self.ticker_sentiment),
        }


@dataclass(slots=True)
class TickerTrend:
    """Aggregated statistics for one ticker over the trend window."""

    symbol: str
    mentions: int = 0
    previous_mentions: int = 0
    unique_authors: int = 0
    subreddits: dict[str, int] = field(default_factory=dict)
    score_sum: int = 0
    comment_sum: int = 0
    engagement: int = 0
    mentions_per_hour: float = 0.0
    change_ratio: float | None = None
    momentum: float = 0.0
    spike_score: float = 0.0
    heat_score: float = 0.0
    sentiment: Sentiment = "neutral"
    sentiment_score: float = 0.0
    sentiment_breakdown: dict[str, int] = field(default_factory=dict)
    is_emerging: bool = False
    first_seen_utc: float = 0.0
    last_seen_utc: float = 0.0
    sample_posts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mentions": self.mentions,
            "previous_mentions": self.previous_mentions,
            "unique_authors": self.unique_authors,
            "subreddits": dict(self.subreddits),
            "score_sum": self.score_sum,
            "comment_sum": self.comment_sum,
            "engagement": self.engagement,
            "mentions_per_hour": self.mentions_per_hour,
            "change_ratio": self.change_ratio,
            "momentum": self.momentum,
            "spike_score": self.spike_score,
            "heat_score": self.heat_score,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "sentiment_breakdown": dict(self.sentiment_breakdown),
            "is_emerging": self.is_emerging,
            "first_seen_utc": self.first_seen_utc,
            "last_seen_utc": self.last_seen_utc,
            "sample_posts": list(self.sample_posts),
        }


@dataclass(slots=True)
class SubredditSummary:
    """Per-subreddit slice of a :class:`TrendReport`."""

    subreddit: str
    posts: int = 0
    posts_with_tickers: int = 0
    mood: Sentiment = "neutral"
    sentiment_score: float = 0.0
    sentiment_breakdown: dict[str, int] = field(default_factory=dict)
    top_tickers: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subreddit": self.subreddit,
            "posts": self.posts,
            "posts_with_tickers": self.posts_with_tickers,
            "mood": self.mood,
            "sentiment_score": self.sentiment_score,
            "sentiment_breakdown": dict(self.sentiment_breakdown),
            "top_tickers": dict(self.top_tickers),
        }


@dataclass(slots=True)
class TrendTimeline:
    """Hourly mention counts per ticker, oldest bucket first."""

    bucket_seconds: int = 3600
    bucket_starts: list[float] = field(default_factory=list)
    series: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_seconds": self.bucket_seconds,
            "bucket_starts": list(self.bucket_starts),
            "series": {k: list(v) for k, v in self.series.items()},
        }


@dataclass(slots=True)
class TrendReport:
    """What the subreddits were talking about over the window, and how that moved."""

    generated_at: float
    window_hours: float
    baseline_hours: float
    window_start: float
    baseline_start: float
    subreddits: list[str] = field(default_factory=list)
    posts_analyzed: int = 0
    posts_in_window: int = 0
    mood: Sentiment = "neutral"
    sentiment_score: float = 0.0
    sentiment_breakdown: dict[str, int] = field(default_factory=dict)
    tickers: list[TickerTrend] = field(default_factory=list)
    rising: list[str] = field(default_factory=list)
    fading: list[str] = field(default_factory=list)
    emerging: list[str] = field(default_factory=list)
    by_subreddit: list[SubredditSummary] = field(default_factory=list)
    timeline: TrendTimeline = field(default_factory=TrendTimeline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "window_hours": self.window_hours,
            "baseline_hours": self.baseline_hours,
            "window_start": self.window_start,
            "baseline_start": self.baseline_start,
            "subreddits": list(self.subreddits),
            "posts_analyzed": self.posts_analyzed,
            "posts_in_window": self.posts_in_window,
            "mood": self.mood,
            "sentiment_score": self.sentiment_score,
            "sentiment_breakdown": dict(self.sentiment_breakdown),
            "tickers": [t.to_dict() for t in self.tickers],
            "rising": list(self.rising),
            "fading": list(self.fading),
            "emerging": list(self.emerging),
            "by_subreddit": [s.to_dict() for s in self.by_subreddit],
            "timeline": self.timeline.to_dict(),
        }
