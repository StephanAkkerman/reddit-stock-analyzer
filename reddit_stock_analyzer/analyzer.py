"""Enrich scraped posts with tickers and sentiment."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from .models import AnalyzedPost, RedditPost, Sentiment
from .recognizer import TickerExtractor, get_default_extractor
from .sentiment import SentimentAnalyzer, signed_score

logger = logging.getLogger(__name__)

#: Sentence and paragraph boundaries. Reddit prose is not clean enough for a
#: real sentence tokeniser to be worth the dependency, and segments only need
#: to be small enough that one of them rarely holds two opposing calls.
_SEGMENT_RE = re.compile(r"(?<=[.!?])\s+|[\n;]+")

#: Segments shorter than this carry no usable signal ("Edit:", "TL;DR").
_MIN_SEGMENT_CHARS = 12


def split_segments(text: str) -> list[str]:
    """Split post text into sentence-ish segments for sentiment attribution."""
    return [
        segment.strip()
        for segment in _SEGMENT_RE.split(text or "")
        if len(segment.strip()) >= _MIN_SEGMENT_CHARS
    ]


def _symbol_pattern(symbol: str) -> re.Pattern[str]:
    """Match a ticker's literal surface form, with or without a cashtag.

    Case-sensitive on purpose: lowercase ``now`` is prose, uppercase ``NOW``
    is ServiceNow.
    """
    return re.compile(rf"\$?\b{re.escape(symbol)}\b")


class PostAnalyzer:
    """Run ticker recognition and sentiment over a batch of posts.

    Parameters
    ----------
    extractor : TickerExtractor, optional
        Defaults to the shared process-wide extractor.
    sentiment : SentimentAnalyzer, optional
        Defaults to a lazily loaded FinTwitBERT-wsb analyzer. Pass
        ``SentimentAnalyzer(pipeline=...)`` to share a loaded model.
    use_ai : bool, default True
        Only used when *extractor* is omitted; see :class:`TickerExtractor`.
    tickers_only : bool, default False
        Drop posts in which no ticker was recognised.
    sentiment_on_title_only : bool, default False
        Classify just the title. Cheaper, and often more accurate on long DD
        posts where the body wanders into unrelated territory.
    per_ticker_sentiment : bool, default True
        Attribute sentiment per ticker in posts that mention several. "Long
        NVDA, short INTC" is bullish and bearish at once, and a single
        post-level label gives one of them the wrong sign. Costs nothing for
        single-ticker posts, which are the majority — see :meth:`analyze`.
    """

    def __init__(
        self,
        extractor: TickerExtractor | None = None,
        sentiment: SentimentAnalyzer | None = None,
        *,
        use_ai: bool = True,
        tickers_only: bool = False,
        sentiment_on_title_only: bool = False,
        per_ticker_sentiment: bool = True,
    ) -> None:
        self.extractor = extractor or get_default_extractor(use_ai=use_ai)
        self.sentiment = sentiment if sentiment is not None else SentimentAnalyzer()
        self.tickers_only = tickers_only
        self.sentiment_on_title_only = sentiment_on_title_only
        self.per_ticker_sentiment = per_ticker_sentiment

    def _sentiment_text(self, post: RedditPost) -> str:
        return post.title if self.sentiment_on_title_only else post.text

    # -- per-ticker attribution -------------------------------------------

    def _attribution_plan(
        self, posts: Sequence[RedditPost], ticker_lists: Sequence[list[str]]
    ) -> tuple[list[str], list[dict[str, list[int]]]]:
        """Work out which text segments need classifying, and for whom.

        A segment naming only one ticker is unambiguous evidence about it; one
        naming several ("NVDA vs INTC: the only pair trade that matters") says
        nothing about either in particular. So a ticker is scored from its
        exclusive segments where it has any, and only falls back to shared
        ones when it has none.

        Returns
        -------
        segments : list of str
            Every segment to classify, flattened across posts, ready for one
            batched forward pass.
        plans : list of dict
            Per post, a mapping of ticker to indices into *segments*. Empty
            for posts that need no attribution.
        """
        segments: list[str] = []
        plans: list[dict[str, list[int]]] = []

        for post, tickers in zip(posts, ticker_lists):
            # One ticker means the post-level label is already about it, and
            # zero means there is nothing to attribute. Either way: no extra
            # inference, which is the common case.
            if len(tickers) < 2:
                plans.append({})
                continue

            post_segments = split_segments(post.text)
            # Which tickers each segment names.
            named: dict[int, set[str]] = {}
            for symbol in tickers:
                pattern = _symbol_pattern(symbol)
                for index, segment in enumerate(post_segments):
                    if pattern.search(segment):
                        named.setdefault(index, set()).add(symbol)

            plan: dict[str, list[int]] = {}
            # Allocate a global slot per segment on first use, so a segment
            # claimed by two tickers is still classified once.
            offsets: dict[int, int] = {}
            for symbol in tickers:
                owned = sorted(i for i, names in named.items() if symbol in names)
                exclusive = [i for i in owned if len(named[i]) == 1]
                chosen = exclusive or owned
                if not chosen:
                    continue
                for index in chosen:
                    if index not in offsets:
                        offsets[index] = len(segments)
                        segments.append(post_segments[index])
                plan[symbol] = [offsets[i] for i in chosen]
            plans.append(plan)

        return segments, plans

    @staticmethod
    def _attribute(
        plan: dict[str, list[int]],
        segment_sentiments: Sequence[tuple[Sentiment, float]],
        post_score: float,
    ) -> dict[str, float]:
        """Average each ticker's segment scores, keeping only real differences."""
        attributed: dict[str, float] = {}
        for symbol, indices in plan.items():
            scores = [
                signed_score(*segment_sentiments[i])
                for i in indices
                if i < len(segment_sentiments)
            ]
            if not scores:
                continue
            mean = round(sum(scores) / len(scores), 4)
            # Storing a value identical to the post's own would only make
            # `sentiment_for` slower and the payload bigger.
            if mean != post_score:
                attributed[symbol] = mean
        return attributed

    def _combine(
        self,
        posts: Sequence[RedditPost],
        ticker_lists: Sequence[list[str]],
        sentiments: Sequence[tuple[Sentiment, float]],
        plans: Sequence[dict[str, list[int]]] | None = None,
        segment_sentiments: Sequence[tuple[Sentiment, float]] = (),
    ) -> list[AnalyzedPost]:
        analyzed: list[AnalyzedPost] = []
        for index, (post, tickers, (label, confidence)) in enumerate(
            zip(posts, ticker_lists, sentiments)
        ):
            if self.tickers_only and not tickers:
                continue
            post_score = signed_score(label, confidence)
            plan = plans[index] if plans else {}
            analyzed.append(
                AnalyzedPost(
                    post=post,
                    tickers=list(tickers),
                    sentiment=label,
                    sentiment_score=post_score,
                    sentiment_confidence=confidence,
                    ticker_sentiment=(
                        self._attribute(plan, segment_sentiments, post_score)
                        if plan
                        else {}
                    ),
                )
            )
        return analyzed

    # -- public API -------------------------------------------------------

    def analyze(self, posts: Sequence[RedditPost]) -> list[AnalyzedPost]:
        """Enrich *posts* (blocking).

        Runs at most two batched forward passes: one over the posts, and —
        when ``per_ticker_sentiment`` is on and some post mentions more than
        one ticker — one over just the segments of those posts that name a
        ticker.
        """
        if not posts:
            return []
        ticker_lists = self.extractor.extract_many([p.text for p in posts])
        sentiments = self.sentiment.analyze_batch(
            [self._sentiment_text(p) for p in posts]
        )

        if not self.per_ticker_sentiment:
            return self._combine(posts, ticker_lists, sentiments)

        segments, plans = self._attribution_plan(posts, ticker_lists)
        segment_sentiments = self.sentiment.analyze_batch(segments)
        return self._combine(posts, ticker_lists, sentiments, plans, segment_sentiments)

    async def analyze_async(self, posts: Sequence[RedditPost]) -> list[AnalyzedPost]:
        """Enrich *posts* without blocking the event loop.

        Both models are CPU-bound and synchronous, so each runs in a worker
        thread. They run sequentially rather than concurrently: two model
        forward passes at once contend for the same cores and the GIL is
        released either way.
        """
        if not posts:
            return []
        ticker_lists = await self.extractor.extract_many_async([p.text for p in posts])
        sentiments = await self.sentiment.analyze_batch_async(
            [self._sentiment_text(p) for p in posts]
        )

        if not self.per_ticker_sentiment:
            return self._combine(posts, ticker_lists, sentiments)

        segments, plans = self._attribution_plan(posts, ticker_lists)
        segment_sentiments = await self.sentiment.analyze_batch_async(segments)
        return self._combine(posts, ticker_lists, sentiments, plans, segment_sentiments)

    def warm_up(self) -> None:
        """Load both models eagerly, e.g. in a FastAPI lifespan handler."""
        self.extractor.warm_up()
        self.sentiment.warm_up()
