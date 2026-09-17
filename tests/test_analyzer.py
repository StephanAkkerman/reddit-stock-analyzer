"""Tests for post enrichment."""

from __future__ import annotations

import pytest
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


class TestPerTickerSentiment:
    """A post can be long one name and short another; one label cannot say so."""

    @staticmethod
    def _analyzer(**kwargs) -> PostAnalyzer:
        return PostAnalyzer(
            extractor=TickerExtractor(
                recognizer=FakeRecognizer(known={"NVDA", "INTC", "AMD"}), use_ai=False
            ),
            sentiment=SentimentAnalyzer(
                pipeline=FakePipeline(
                    {
                        # The whole post (title + body, as RedditPost.text
                        # joins them), then each segment of it.
                        "Long NVDA, short INTC.\n\nNVDA is printing money. "
                        "INTC is dead money here.": ("BULLISH", 0.7),
                        "Long NVDA, short INTC.": ("BULLISH", 0.6),
                        "NVDA is printing money.": ("BULLISH", 0.9),
                        "INTC is dead money here.": ("BEARISH", 0.8),
                    }
                )
            ),
            **kwargs,
        )

    def test_opposing_calls_get_opposite_signs(self):
        post = make_post(
            "a",
            title="Long NVDA, short INTC.",
            selftext="NVDA is printing money. INTC is dead money here.",
        )
        analyzed = self._analyzer().analyze([post])[0]

        assert analyzed.sentiment == "bullish"  # the post as a whole
        assert analyzed.sentiment_for("NVDA") > 0
        assert analyzed.sentiment_for("INTC") < 0

    def test_single_ticker_posts_skip_the_extra_pass(self):
        analyzer = self._analyzer()
        analyzer.analyze([make_post("a", title="NVDA is printing money.")])
        # One call for the post batch, and no segment batch: attribution is
        # free for the common case.
        assert len(analyzer.sentiment._pipe.calls) == 1

    def test_multi_ticker_posts_trigger_one_extra_batched_pass(self):
        analyzer = self._analyzer()
        analyzer.analyze(
            [
                make_post(
                    "a",
                    title="Long NVDA, short INTC.",
                    selftext="NVDA is printing money. INTC is dead money here.",
                )
            ]
        )
        assert len(analyzer.sentiment._pipe.calls) == 2

    def test_attribution_can_be_switched_off(self):
        post = make_post(
            "a",
            title="Long NVDA, short INTC.",
            selftext="NVDA is printing money. INTC is dead money here.",
        )
        analyzed = self._analyzer(per_ticker_sentiment=False).analyze([post])[0]
        assert analyzed.ticker_sentiment == {}
        # Both fall back to the post's own score.
        assert analyzed.sentiment_for("INTC") == analyzed.sentiment_score

    def test_unlocatable_ticker_falls_back_to_the_post_score(self):
        # "AMD" never appears verbatim (the recognizer mapped a company name),
        # so there is no segment to attribute and the post label stands.
        post = make_post(
            "a",
            title="Long NVDA, short INTC.",
            selftext="NVDA is printing money. INTC is dead money here.",
        )
        analyzed = self._analyzer().analyze([post])[0]
        analyzed.tickers.append("AMD")
        assert analyzed.sentiment_for("AMD") == analyzed.sentiment_score

    def test_serialized_payload_carries_the_attribution(self):
        post = make_post(
            "a",
            title="Long NVDA, short INTC.",
            selftext="NVDA is printing money. INTC is dead money here.",
        )
        payload = self._analyzer().analyze([post])[0].to_dict()
        assert payload["ticker_sentiment"]["INTC"] < 0

    def test_a_segment_naming_both_tickers_is_skipped_when_better_exists(self):
        # The title names both, so it says nothing about either in particular;
        # each ticker is scored from the sentence that names only it.
        post = make_post(
            "a",
            title="Long NVDA, short INTC.",
            selftext="NVDA is printing money. INTC is dead money here.",
        )
        analyzed = self._analyzer().analyze([post])[0]
        assert analyzed.sentiment_for("NVDA") == pytest.approx(0.9)
        assert analyzed.sentiment_for("INTC") == pytest.approx(-0.8)

    def test_shared_segments_are_used_when_nothing_else_names_the_ticker(self):
        post = make_post("a", title="Long NVDA, short INTC.")
        analyzed = self._analyzer().analyze([post])[0]
        # Only the shared title exists, so both fall back to it and agree.
        assert analyzed.sentiment_for("NVDA") == analyzed.sentiment_for("INTC")

    def test_a_shared_segment_is_classified_once(self):
        analyzer = self._analyzer()
        analyzer.analyze([make_post("a", title="Long NVDA, short INTC.")])
        segment_batch = analyzer.sentiment._pipe.calls[1]
        assert segment_batch == ["Long NVDA, short INTC."]
