"""Tests for the sentiment wrapper."""

from __future__ import annotations

import pytest
from conftest import FakePipeline

from reddit_stock_analyzer.sentiment import (
    SentimentAnalyzer,
    normalize_label,
    signed_score,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BULLISH", "bullish"),
        ("bullish", "bullish"),
        ("POSITIVE", "bullish"),
        ("LABEL_2", "bullish"),
        ("BEARISH", "bearish"),
        ("LABEL_0", "bearish"),
        ("NEUTRAL", "neutral"),
        ("something else", "neutral"),
        ("", "neutral"),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [("bullish", 0.9, 0.9), ("bearish", 0.9, -0.9), ("neutral", 0.9, 0.0)],
)
def test_signed_score(label, confidence, expected):
    assert signed_score(label, confidence) == pytest.approx(expected)


class TestAnalyzeBatch:
    def test_classifies_each_text(self):
        pipe = FakePipeline({"moon": ("BULLISH", 0.95), "crash": ("BEARISH", 0.8)})
        analyzer = SentimentAnalyzer(pipeline=pipe)
        assert analyzer.analyze_batch(["moon", "crash"]) == [
            ("bullish", 0.95),
            ("bearish", 0.8),
        ]

    def test_blank_texts_never_reach_the_model(self):
        pipe = FakePipeline({"moon": ("BULLISH", 0.95)})
        analyzer = SentimentAnalyzer(pipeline=pipe)
        results = analyzer.analyze_batch(["", "moon", "   "])
        assert results == [("neutral", 0.0), ("bullish", 0.95), ("neutral", 0.0)]
        assert pipe.calls == [["moon"]]

    def test_long_text_is_trimmed_before_tokenisation(self):
        pipe = FakePipeline()
        SentimentAnalyzer(pipeline=pipe).analyze_batch(["x" * 10_000])
        assert len(pipe.calls[0][0]) == 2000

    def test_top_k_style_results_are_unwrapped(self):
        class TopKPipeline:
            def __call__(self, texts, **kwargs):
                return [[{"label": "BULLISH", "score": 0.7}] for _ in texts]

        analyzer = SentimentAnalyzer(pipeline=TopKPipeline())
        assert analyzer.analyze_batch(["moon"]) == [("bullish", 0.7)]

    def test_empty_input(self):
        assert SentimentAnalyzer(pipeline=FakePipeline()).analyze_batch([]) == []

    def test_inference_failure_degrades_to_neutral(self):
        class Exploding:
            def __call__(self, texts, **kwargs):
                raise RuntimeError("CUDA out of memory")

        analyzer = SentimentAnalyzer(pipeline=Exploding())
        assert analyzer.analyze_batch(["moon", "crash"]) == [
            ("neutral", 0.0),
            ("neutral", 0.0),
        ]

    def test_missing_transformers_degrades_to_neutral(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        analyzer = SentimentAnalyzer()
        assert analyzer.analyze_batch(["moon"]) == [("neutral", 0.0)]
        assert analyzer.available is False

    async def test_async_wrapper_matches_sync(self):
        analyzer = SentimentAnalyzer(pipeline=FakePipeline({"moon": ("BULLISH", 0.9)}))
        assert await analyzer.analyze_batch_async(["moon"]) == [("bullish", 0.9)]
        assert await analyzer.analyze_batch_async([]) == []

    def test_available_is_true_with_an_injected_pipeline(self):
        assert SentimentAnalyzer(pipeline=FakePipeline()).available is True
