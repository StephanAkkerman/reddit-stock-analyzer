"""Tests for the stock-recognizer adapter."""

from __future__ import annotations

import pytest
from conftest import FakeRecognizer

from reddit_stock_analyzer.recognizer import (
    TickerExtractor,
    get_default_extractor,
    reset_default_extractor,
)


@pytest.fixture(autouse=True)
def _clean_default():
    reset_default_extractor()
    yield
    reset_default_extractor()


class TestExtract:
    def test_uses_the_ai_path_when_enabled(self, fake_recognizer: FakeRecognizer):
        extractor = TickerExtractor(recognizer=fake_recognizer, use_ai=True)
        assert extractor.extract("$PLAB is a TSMC supplier") == ["PLAB", "TSM"]
        assert fake_recognizer.recognize_ai_calls

    def test_uses_the_regex_path_when_disabled(self, fake_recognizer: FakeRecognizer):
        extractor = TickerExtractor(recognizer=fake_recognizer, use_ai=False)
        assert extractor.extract("$PLAB is a TSMC supplier") == ["PLAB"]
        assert not fake_recognizer.recognize_ai_calls
        assert fake_recognizer.recognize_calls

    def test_falls_back_when_the_recognizer_has_no_ai_method(self):
        class RegexOnly:
            def recognize(self, text: str) -> list[str]:
                return ["GME"]

        extractor = TickerExtractor(recognizer=RegexOnly(), use_ai=True)
        assert extractor.extract("GME") == ["GME"]

    def test_results_are_deduplicated_uppercased_and_sorted(self):
        class Noisy:
            def recognize(self, text: str) -> list[str]:
                return ["nvda", "GME", "NVDA", ""]

        extractor = TickerExtractor(recognizer=Noisy(), use_ai=False)
        assert extractor.extract("anything") == ["GME", "NVDA"]

    def test_blocklist_drops_symbols(self, fake_recognizer: FakeRecognizer):
        extractor = TickerExtractor(
            recognizer=fake_recognizer, use_ai=False, blocklist=["dd"]
        )
        assert extractor.extract("$DD and $GME") == ["GME"]

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_short_circuits(self, text):
        class Exploding:
            def recognize(self, text: str) -> list[str]:  # pragma: no cover
                raise AssertionError("should not be called")

        assert TickerExtractor(recognizer=Exploding()).extract(text) == []

    def test_a_failing_recognizer_does_not_sink_the_scrape(self):
        class Exploding:
            def recognize(self, text: str) -> list[str]:
                raise RuntimeError("model exploded")

        extractor = TickerExtractor(recognizer=Exploding(), use_ai=False)
        assert extractor.extract("$GME") == []

    def test_extract_many_preserves_order(self, fake_recognizer: FakeRecognizer):
        extractor = TickerExtractor(recognizer=fake_recognizer, use_ai=False)
        assert extractor.extract_many(["$GME", "", "$AMC"]) == [["GME"], [], ["AMC"]]

    async def test_extract_many_async_matches_sync(self, fake_recognizer):
        extractor = TickerExtractor(recognizer=fake_recognizer, use_ai=False)
        assert await extractor.extract_many_async(["$GME"]) == [["GME"]]
        assert await extractor.extract_many_async([]) == []


class TestLazyConstruction:
    def test_recognizer_is_built_lazily_and_once(self, monkeypatch):
        built = {"n": 0}
        extractor = TickerExtractor(use_ai=False)

        def build():
            built["n"] += 1
            return FakeRecognizer()

        monkeypatch.setattr(extractor, "_build", build)
        assert built["n"] == 0, "constructing must not load the model"

        assert extractor.extract("$GME") == ["GME"]
        assert extractor.extract("$AMC") == ["AMC"]
        assert built["n"] == 1

    def test_missing_package_raises_a_helpful_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "stock_recognizer":
                raise ImportError("no module named stock_recognizer")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match="pip install stock-recognizer"):
            TickerExtractor(use_ai=False).extract("$GME")

    def test_ai_load_failure_degrades_to_the_regex_path(self, monkeypatch):
        import sys
        import types

        attempts: list[bool] = []

        class FlakyRecognizer:
            def __init__(self, use_ai=True, **kwargs):
                attempts.append(use_ai)
                if use_ai:
                    raise RuntimeError("no torch here")

            def recognize(self, text: str) -> list[str]:
                return ["GME"]

        module = types.ModuleType("stock_recognizer")
        module.StockRecognizer = FlakyRecognizer
        monkeypatch.setitem(sys.modules, "stock_recognizer", module)

        extractor = TickerExtractor(use_ai=True)
        assert extractor.extract("GME") == ["GME"]
        assert attempts == [True, False]
        assert extractor.use_ai is False


class TestDefaultExtractor:
    def test_default_extractor_is_cached(self):
        assert get_default_extractor() is get_default_extractor()

    def test_reset_clears_the_cache(self):
        first = get_default_extractor()
        reset_default_extractor()
        assert get_default_extractor() is not first
