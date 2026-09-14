"""Tests for post enrichment."""

from __future__ import annotations

from conftest import FakePipeline, FakeRecognizer, make_post

from reddit_stock_analyzer.analyzer import PostAnalyzer
from reddit_stock_analyzer.recognizer import TickerExtractor
from reddit_stock_analyzer.sentiment import SentimentAnalyzer


def _analyzer(**kwargs) -> PostAnalyzer:
    return PostAnalyzer(
        extractor=TickerExtractor(recognizer=FakeRecognizer(), use_ai=False),
        sentiment=SentimentAnalyzer(
            pipeline=FakePipeline(
                {
                    "$GME squeeze": ("BULLISH", 0.95),
                    "$GME squeeze\n\nbody": ("BEARISH", 0.6),
                }
            )
        ),
        **kwargs,
    )


class TestAnalyze:
    def test_attaches_tickers_and_sentiment(self):
        posts = [make_post("a", title="$GME squeeze")]
        analyzed = _analyzer().analyze(posts)
        assert analyzed[0].tickers == ["GME"]
        assert analyzed[0].sentiment == "bullish"
        assert analyzed[0].sentiment_score == 0.95
        assert analyzed[0].sentiment_confidence == 0.95

    def test_bearish_posts_get_a_negative_score(self):
        posts = [make_post("a", title="$GME squeeze", selftext="body")]
        assert _analyzer().analyze(posts)[0].sentiment_score == -0.6

    def test_title_only_mode_ignores_the_body(self):
        posts = [make_post("a", title="$GME squeeze", selftext="body")]
        analyzed = _analyzer(sentiment_on_title_only=True).analyze(posts)
        assert analyzed[0].sentiment == "bullish"

    def test_tickers_only_drops_unrelated_posts(self):
        posts = [make_post("a", title="$GME squeeze"), make_post("b", title="hello")]
        analyzed = _analyzer(tickers_only=True).analyze(posts)
        assert [p.id for p in analyzed] == ["a"]

    def test_empty_input(self):
        assert _analyzer().analyze([]) == []

    def test_analyzed_post_serializes_post_and_analysis_together(self):
        analyzed = _analyzer().analyze([make_post("a", title="$GME squeeze")])[0]
        payload = analyzed.to_dict()
        assert payload["id"] == "a"
        assert payload["tickers"] == ["GME"]
        assert payload["sentiment"] == "bullish"

    async def test_async_matches_sync(self):
        posts = [make_post("a", title="$GME squeeze")]
        analyzed = await _analyzer().analyze_async(posts)
        assert analyzed[0].tickers == ["GME"]
        assert await _analyzer().analyze_async([]) == []
