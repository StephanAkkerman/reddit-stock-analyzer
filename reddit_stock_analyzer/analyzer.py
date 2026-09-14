"""Enrich scraped posts with tickers and sentiment."""

from __future__ import annotations

import logging
from typing import Sequence

from .models import AnalyzedPost, RedditPost
from .recognizer import TickerExtractor, get_default_extractor
from .sentiment import SentimentAnalyzer, signed_score

logger = logging.getLogger(__name__)


class PostAnalyzer:
    """Run ticker recognition and sentiment over a batch of posts.

    Parameters
    ----------
    extractor : TickerExtractor, optional
        Defaults to the shared process-wide extractor.
    sentiment : SentimentAnalyzer, optional
        Defaults to a lazily loaded FinTwitBERT analyzer. Pass
        ``SentimentAnalyzer(pipeline=...)`` to share a loaded model.
    use_ai : bool, default True
        Only used when *extractor* is omitted; see :class:`TickerExtractor`.
    tickers_only : bool, default False
        Drop posts in which no ticker was recognised.
    sentiment_on_title_only : bool, default False
        Classify just the title. Cheaper, and often more accurate on long DD
        posts where the body wanders into unrelated territory.
    """

    def __init__(
        self,
        extractor: TickerExtractor | None = None,
        sentiment: SentimentAnalyzer | None = None,
        *,
        use_ai: bool = True,
        tickers_only: bool = False,
        sentiment_on_title_only: bool = False,
    ) -> None:
        self.extractor = extractor or get_default_extractor(use_ai=use_ai)
        self.sentiment = sentiment if sentiment is not None else SentimentAnalyzer()
        self.tickers_only = tickers_only
        self.sentiment_on_title_only = sentiment_on_title_only

    def _sentiment_text(self, post: RedditPost) -> str:
        return post.title if self.sentiment_on_title_only else post.text

    def _combine(
        self,
        posts: Sequence[RedditPost],
        ticker_lists: Sequence[list[str]],
        sentiments: Sequence[tuple[str, float]],
    ) -> list[AnalyzedPost]:
        analyzed: list[AnalyzedPost] = []
        for post, tickers, (label, confidence) in zip(posts, ticker_lists, sentiments):
            if self.tickers_only and not tickers:
                continue
            analyzed.append(
                AnalyzedPost(
                    post=post,
                    tickers=list(tickers),
                    sentiment=label,  # type: ignore[arg-type]
                    sentiment_score=signed_score(label, confidence),  # type: ignore[arg-type]
                    sentiment_confidence=confidence,
                )
            )
        return analyzed

    def analyze(self, posts: Sequence[RedditPost]) -> list[AnalyzedPost]:
        """Enrich *posts* (blocking)."""
        if not posts:
            return []
        ticker_lists = self.extractor.extract_many([p.text for p in posts])
        sentiments = self.sentiment.analyze_batch(
            [self._sentiment_text(p) for p in posts]
        )
        return self._combine(posts, ticker_lists, sentiments)

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
        return self._combine(posts, ticker_lists, sentiments)

    def warm_up(self) -> None:
        """Load both models eagerly, e.g. in a FastAPI lifespan handler."""
        self.extractor.warm_up()
        self.sentiment.warm_up()
