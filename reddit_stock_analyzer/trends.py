"""Trend analytics over analyzed Reddit posts.

Everything here is pure: given a list of
:class:`~reddit_stock_analyzer.models.AnalyzedPost` and a reference timestamp
it returns a :class:`~reddit_stock_analyzer.models.TrendReport`. No network,
no models, no clock unless you omit ``now`` — which makes the maths easy to
test and cheap to re-run over a cached scrape.

The central idea is a **two-window comparison**. Posts are split into the
*window* (the recent period you care about, default 24h) and an equally long
*baseline* immediately before it. A ticker's raw mention count says how loud
it is; comparing the two windows says whether it is getting louder, which is
the part that actually precedes moves.

Metric glossary
---------------
mentions
    Number of posts in the window mentioning the ticker. Counted once per
    post: a post screaming ``GME`` twenty times counts once, so a single
    ranting post cannot manufacture a trend.
previous_mentions
    The same count over the baseline window.
momentum
    ``(mentions - previous_mentions) / (previous_mentions + 1)``. The ``+1``
    keeps a ticker that went 0 → 5 finite and comparable (4.0) instead of
    infinite, while still ranking it above one that went 50 → 60 (0.2).
change_ratio
    ``mentions / previous_mentions``, or ``None`` when the baseline is zero.
    The unsmoothed version, for display.
spike_score
    Standard deviations above the mean mention count of this scrape. Answers
    "is this unusual *for this cohort*", which is what separates a genuine
    spike from a subreddit where everything is busy today.
heat_score
    Weighted 0..1 blend of mentions, engagement, unique authors and momentum
    (see :data:`DEFAULT_HEAT_WEIGHTS`). The default ranking: loud, upvoted,
    discussed by many *different* people, and accelerating.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AnalyzedPost,
    Sentiment,
    SubredditSummary,
    TickerTrend,
    TrendReport,
    TrendTimeline,
)

#: Contribution of each normalised component to ``heat_score``.
DEFAULT_HEAT_WEIGHTS: dict[str, float] = {
    "mentions": 0.40,
    "engagement": 0.25,
    "authors": 0.15,
    "momentum": 0.20,
}

#: Above/below this mean signed score a cohort is called bullish/bearish.
SENTIMENT_THRESHOLD = 0.15

#: Momentum is clamped before normalisation: a 0 → 40 ticker would otherwise
#: flatten every other momentum score to nearly zero.
_MOMENTUM_FLOOR = -1.0
_MOMENTUM_CEILING = 3.0

_ANONYMOUS_AUTHORS = frozenset({"", "none", "[deleted]", "automoderator"})


def _clean_author(author: str) -> str | None:
    name = (author or "").strip()
    return None if name.lower() in _ANONYMOUS_AUTHORS else name


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Scale values onto 0..1 by their maximum (all-zero maps to all-zero)."""
    if not values:
        return {}
    peak = max(values.values())
    if peak <= 0:
        return {key: 0.0 for key in values}
    return {key: value / peak for key, value in values.items()}


def _mood_from_score(score: float) -> Sentiment:
    if score > SENTIMENT_THRESHOLD:
        return "bullish"
    if score < -SENTIMENT_THRESHOLD:
        return "bearish"
    return "neutral"


def _sample_post(post: AnalyzedPost) -> dict[str, Any]:
    return {
        "id": post.post.id,
        "title": post.post.title,
        "url": post.post.url,
        "subreddit": post.post.subreddit,
        "score": post.post.score,
        "num_comments": post.post.num_comments,
        "created_utc": post.post.created_utc,
        "sentiment": post.sentiment,
    }


class _Accumulator:
    """Mutable per-ticker tallies, collapsed into a TickerTrend at the end."""

    __slots__ = (
        "mentions",
        "authors",
        "subreddits",
        "score_sum",
        "comment_sum",
        "engagement",
        "sentiment_scores",
        "sentiment_counts",
        "first_seen",
        "last_seen",
        "posts",
    )

    def __init__(self) -> None:
        self.mentions = 0
        self.authors: set[str] = set()
        self.subreddits: Counter[str] = Counter()
        self.score_sum = 0
        self.comment_sum = 0
        self.engagement = 0
        self.sentiment_scores: list[float] = []
        self.sentiment_counts: Counter[str] = Counter()
        self.first_seen = math.inf
        self.last_seen = 0.0
        self.posts: list[AnalyzedPost] = []

    def add(self, post: AnalyzedPost, symbol: str) -> None:
        self.mentions += 1
        author = _clean_author(post.post.author)
        if author:
            self.authors.add(author)
        self.subreddits[post.post.subreddit] += 1
        self.score_sum += int(post.post.score)
        self.comment_sum += int(post.post.num_comments)
        self.engagement += post.post.engagement
        # Sentiment towards *this* ticker where the analyzer could attribute
        # it (a post going long one name and short another), the post's own
        # label otherwise.
        score = post.sentiment_for(symbol)
        attributed = symbol in post.ticker_sentiment
        self.sentiment_scores.append(score)
        self.sentiment_counts[
            _mood_from_score(score) if attributed else post.sentiment
        ] += 1
        self.first_seen = min(self.first_seen, post.created_utc)
        self.last_seen = max(self.last_seen, post.created_utc)
        self.posts.append(post)


def _build_timeline(
    posts: Sequence[AnalyzedPost],
    symbols: Iterable[str],
    window_start: float,
    now: float,
    bucket_seconds: int,
) -> TrendTimeline:
    """Bucket mentions of *symbols* into fixed-width bins over the window."""
    tracked = list(symbols)
    span = max(now - window_start, 0.0)
    bucket_count = max(1, math.ceil(span / bucket_seconds)) if tracked else 0
    if not bucket_count:
        return TrendTimeline(bucket_seconds=bucket_seconds)

    starts = [window_start + i * bucket_seconds for i in range(bucket_count)]
    series = {symbol: [0] * bucket_count for symbol in tracked}
    tracked_set = set(tracked)

    for post in posts:
        index = int((post.created_utc - window_start) // bucket_seconds)
        index = min(max(index, 0), bucket_count - 1)
        for symbol in post.tickers:
            if symbol in tracked_set:
                series[symbol][index] += 1

    return TrendTimeline(
        bucket_seconds=bucket_seconds, bucket_starts=starts, series=series
    )


def _summarize_subreddits(
    posts: Sequence[AnalyzedPost], top_tickers_per_sub: int
) -> list[SubredditSummary]:
    grouped: dict[str, list[AnalyzedPost]] = defaultdict(list)
    for post in posts:
        grouped[post.post.subreddit].append(post)

    summaries: list[SubredditSummary] = []
    for subreddit, group in grouped.items():
        counts: Counter[str] = Counter()
        for post in group:
            counts.update(post.tickers)
        score = round(fmean([p.sentiment_score for p in group]), 4) if group else 0.0
        summaries.append(
            SubredditSummary(
                subreddit=subreddit,
                posts=len(group),
                posts_with_tickers=sum(1 for p in group if p.tickers),
                mood=_mood_from_score(score),
                sentiment_score=score,
                sentiment_breakdown=dict(Counter(p.sentiment for p in group)),
                top_tickers=dict(counts.most_common(top_tickers_per_sub)),
            )
        )
    summaries.sort(key=lambda s: s.posts, reverse=True)
    return summaries


def compute_trends(
    posts: Sequence[AnalyzedPost],
    *,
    now: float | None = None,
    window_hours: float = 24.0,
    baseline_hours: float | None = None,
    subreddits: Sequence[str] | None = None,
    min_mentions: int = 1,
    top_n: int = 25,
    max_sample_posts: int = 3,
    emerging_min_mentions: int = 3,
    rising_min_momentum: float = 0.5,
    fading_max_momentum: float = -0.5,
    timeline_symbols: int = 10,
    timeline_bucket_seconds: int = 3600,
    top_tickers_per_subreddit: int = 5,
    weights: Mapping[str, float] | None = None,
) -> TrendReport:
    """Rank what the given posts were talking about, and how that is moving.

    Parameters
    ----------
    posts : sequence of AnalyzedPost
        Posts from one or more subreddits. Should span both the window and the
        baseline — pass roughly ``2 * window_hours`` worth of posts, otherwise
        every ticker looks like it is emerging.
    now : float, optional
        Reference epoch seconds. Defaults to the wall clock.
    window_hours : float, default 24
        Length of the recent window under analysis.
    baseline_hours : float, optional
        Length of the comparison window immediately before it. Defaults to
        ``window_hours`` (a like-for-like comparison).
    subreddits : sequence of str, optional
        Recorded on the report for provenance; does not filter.
    min_mentions : int, default 1
        Drop tickers mentioned fewer times than this in the window.
    top_n : int, default 25
        Maximum tickers returned, ranked by ``heat_score``.
    max_sample_posts : int, default 3
        Highest-engagement example posts attached to each ticker.
    emerging_min_mentions : int, default 3
        Mentions needed before a ticker absent from the baseline counts as
        emerging — below this it is one person posting twice.
    rising_min_momentum, fading_max_momentum : float
        Momentum cut-offs for the ``rising`` and ``fading`` lists.
    timeline_symbols : int, default 10
        How many top tickers get a per-bucket time series.
    timeline_bucket_seconds : int, default 3600
        Timeline bucket width.
    top_tickers_per_subreddit : int, default 5
        Tickers kept in each per-subreddit summary.
    weights : mapping, optional
        Override :data:`DEFAULT_HEAT_WEIGHTS`. Missing keys keep their default.

    Returns
    -------
    TrendReport

    Raises
    ------
    ValueError
        If ``window_hours`` or ``baseline_hours`` is not positive.
    """
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    baseline_hours = window_hours if baseline_hours is None else baseline_hours
    if baseline_hours <= 0:
        raise ValueError("baseline_hours must be positive")

    reference = time.time() if now is None else now
    window_start = reference - window_hours * 3600
    baseline_start = window_start - baseline_hours * 3600

    current = [p for p in posts if p.created_utc >= window_start]
    baseline = [p for p in posts if baseline_start <= p.created_utc < window_start]

    baseline_counts: Counter[str] = Counter()
    for post in baseline:
        baseline_counts.update(post.tickers)

    accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    for post in current:
        for symbol in post.tickers:
            accumulators[symbol].add(post, symbol)

    eligible = {
        symbol: acc
        for symbol, acc in accumulators.items()
        if acc.mentions >= min_mentions
    }

    mention_counts = [acc.mentions for acc in eligible.values()]
    mean_mentions = fmean(mention_counts) if mention_counts else 0.0
    stdev_mentions = pstdev(mention_counts) if len(mention_counts) > 1 else 0.0

    momentum_by_symbol: dict[str, float] = {}
    for symbol, acc in eligible.items():
        previous = baseline_counts.get(symbol, 0)
        momentum_by_symbol[symbol] = (acc.mentions - previous) / (previous + 1)

    resolved_weights = {**DEFAULT_HEAT_WEIGHTS, **dict(weights or {})}
    norm_mentions = _normalize({s: float(a.mentions) for s, a in eligible.items()})
    # log1p keeps one 40k-upvote post from pinning every other ticker to ~0.
    norm_engagement = _normalize(
        {s: math.log1p(max(a.engagement, 0)) for s, a in eligible.items()}
    )
    norm_authors = _normalize({s: float(len(a.authors)) for s, a in eligible.items()})
    momentum_span = _MOMENTUM_CEILING - _MOMENTUM_FLOOR
    norm_momentum = {
        symbol: (min(max(value, _MOMENTUM_FLOOR), _MOMENTUM_CEILING) - _MOMENTUM_FLOOR)
        / momentum_span
        for symbol, value in momentum_by_symbol.items()
    }

    trends: list[TickerTrend] = []
    for symbol, acc in eligible.items():
        previous = baseline_counts.get(symbol, 0)
        momentum = momentum_by_symbol[symbol]
        sentiment_score = round(fmean(acc.sentiment_scores), 4)
        spike = (
            (acc.mentions - mean_mentions) / stdev_mentions
            if stdev_mentions > 0
            else 0.0
        )
        heat = (
            resolved_weights["mentions"] * norm_mentions.get(symbol, 0.0)
            + resolved_weights["engagement"] * norm_engagement.get(symbol, 0.0)
            + resolved_weights["authors"] * norm_authors.get(symbol, 0.0)
            + resolved_weights["momentum"] * norm_momentum.get(symbol, 0.0)
        )
        samples = sorted(acc.posts, key=lambda p: p.post.engagement, reverse=True)

        trends.append(
            TickerTrend(
                symbol=symbol,
                mentions=acc.mentions,
                previous_mentions=previous,
                unique_authors=len(acc.authors),
                subreddits=dict(acc.subreddits.most_common()),
                score_sum=acc.score_sum,
                comment_sum=acc.comment_sum,
                engagement=acc.engagement,
                mentions_per_hour=round(acc.mentions / window_hours, 4),
                change_ratio=(round(acc.mentions / previous, 4) if previous else None),
                momentum=round(momentum, 4),
                spike_score=round(spike, 4),
                heat_score=round(heat, 4),
                sentiment=_mood_from_score(sentiment_score),
                sentiment_score=sentiment_score,
                sentiment_breakdown=dict(acc.sentiment_counts),
                is_emerging=previous == 0 and acc.mentions >= emerging_min_mentions,
                first_seen_utc=0.0 if acc.first_seen is math.inf else acc.first_seen,
                last_seen_utc=acc.last_seen,
                sample_posts=[_sample_post(p) for p in samples[:max_sample_posts]],
            )
        )

    # Ties on heat_score fall back to mentions then symbol, so the ordering is
    # stable across runs over the same data.
    trends.sort(key=lambda t: (-t.heat_score, -t.mentions, t.symbol))
    ranked = trends[: max(0, top_n)]

    rising = [
        t.symbol
        for t in ranked
        if t.previous_mentions > 0 and t.momentum >= rising_min_momentum
    ]
    emerging = [t.symbol for t in ranked if t.is_emerging]

    # Fading is computed over the baseline, not the window: a ticker that went
    # quiet entirely has no window entry to rank, and that is the strongest
    # fade there is.
    fading: list[tuple[str, float]] = []
    for symbol, previous in baseline_counts.items():
        if previous < emerging_min_mentions:
            continue
        mentions = accumulators[symbol].mentions if symbol in accumulators else 0
        momentum = (mentions - previous) / (previous + 1)
        if momentum <= fading_max_momentum:
            fading.append((symbol, momentum))
    fading.sort(key=lambda item: item[1])

    overall_score = (
        round(fmean([p.sentiment_score for p in current]), 4) if current else 0.0
    )
    covered = (
        list(subreddits)
        if subreddits is not None
        else sorted({p.post.subreddit for p in posts})
    )

    return TrendReport(
        generated_at=reference,
        window_hours=window_hours,
        baseline_hours=baseline_hours,
        window_start=window_start,
        baseline_start=baseline_start,
        subreddits=covered,
        posts_analyzed=len(posts),
        posts_in_window=len(current),
        mood=_mood_from_score(overall_score),
        sentiment_score=overall_score,
        sentiment_breakdown=dict(Counter(p.sentiment for p in current)),
        tickers=ranked,
        rising=rising,
        fading=[symbol for symbol, _ in fading[: max(0, top_n)]],
        emerging=emerging,
        by_subreddit=_summarize_subreddits(current, top_tickers_per_subreddit),
        timeline=_build_timeline(
            current,
            [t.symbol for t in ranked[: max(0, timeline_symbols)]],
            window_start,
            reference,
            timeline_bucket_seconds,
        ),
    )
