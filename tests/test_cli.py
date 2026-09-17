"""Tests for the command line entry point."""

from __future__ import annotations

import pytest
from conftest import HOUR, NOW, make_analyzed

from reddit_stock_analyzer.__main__ import (
    _build_parser,
    _format_report,
    _resolve_subreddits,
)
from reddit_stock_analyzer.config import DEFAULT_SUBREDDITS, subreddits_for
from reddit_stock_analyzer.trends import compute_trends


def _parse(*argv: str):
    return _build_parser().parse_args(list(argv))


class TestArguments:
    def test_no_targets_falls_back_to_the_defaults(self):
        assert _resolve_subreddits(_parse()) == list(DEFAULT_SUBREDDITS)

    def test_repeated_subreddit_flags_accumulate(self):
        args = _parse("-s", "wallstreetbets", "-s", "options")
        assert _resolve_subreddits(args) == ["wallstreetbets", "options"]

    def test_categories_expand_and_merge_without_duplicates(self):
        args = _parse("-c", "retail", "-s", "wallstreetbets")
        resolved = _resolve_subreddits(args)
        assert resolved.count("wallstreetbets") == 1
        assert set(subreddits_for("retail")) <= set(resolved)

    def test_unknown_category_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            _parse("-c", "nonsense")


class TestFormatting:
    @staticmethod
    def _report():
        posts = [
            *[
                make_analyzed(
                    f"n{i}",
                    tickers=["NVDA"],
                    author=f"u{i}",
                    sentiment="bullish",
                    sentiment_score=0.8,
                    created_utc=NOW - HOUR,
                )
                for i in range(5)
            ],
            *[
                make_analyzed(f"a{i}", tickers=["AMC"], created_utc=NOW - 30 * HOUR)
                for i in range(6)
            ],
        ]
        return compute_trends(
            posts, now=NOW, subreddits=["wallstreetbets"], emerging_min_mentions=3
        )

    def test_renders_a_row_per_ticker(self):
        rendered = _format_report(self._report(), top=10)
        assert "r/wallstreetbets" in rendered
        assert "NVDA" in rendered
        assert "bullish" in rendered
        assert "emerging: NVDA" in rendered
        assert "fading: AMC" in rendered

    def test_top_limits_the_rows(self):
        posts = [
            make_analyzed(f"p{i}", tickers=[f"AA{chr(65 + i)}"], created_utc=NOW)
            for i in range(4)
        ]
        rendered = _format_report(compute_trends(posts, now=NOW), top=2)
        assert rendered.count("AA") == 2

    def test_empty_report_says_so_instead_of_printing_a_bare_header(self):
        rendered = _format_report(compute_trends([], now=NOW), top=10)
        assert "no tickers recognised" in rendered

    def test_no_blank_line_between_mover_lists(self):
        rendered = _format_report(self._report(), top=10)
        assert "emerging: NVDA\nfading: AMC" in rendered
