"""Tests for the trend maths."""

from __future__ import annotations

import pytest
from conftest import HOUR, NOW, make_analyzed

from reddit_stock_analyzer.trends import compute_trends


def _report(posts, **kwargs):
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("window_hours", 24.0)
    return compute_trends(posts, **kwargs)


def _by_symbol(report):
    return {t.symbol: t for t in report.tickers}


class TestWindowSplit:
    def test_counts_only_posts_inside_the_window(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW - HOUR),
            make_analyzed("b", tickers=["GME"], created_utc=NOW - 30 * HOUR),
        ]
        report = _report(posts)
        assert _by_symbol(report)["GME"].mentions == 1
        assert _by_symbol(report)["GME"].previous_mentions == 1
        assert report.posts_in_window == 1
        assert report.posts_analyzed == 2

    def test_posts_older_than_the_baseline_are_ignored(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW - HOUR),
            make_analyzed("ancient", tickers=["GME"], created_utc=NOW - 200 * HOUR),
        ]
        assert _by_symbol(_report(posts))["GME"].previous_mentions == 0

    def test_baseline_hours_can_differ_from_the_window(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW - HOUR),
            make_analyzed("b", tickers=["GME"], created_utc=NOW - 20 * HOUR),
            make_analyzed("c", tickers=["GME"], created_utc=NOW - 60 * HOUR),
        ]
        report = _report(posts, window_hours=12.0, baseline_hours=72.0)
        trend = _by_symbol(report)["GME"]
        assert trend.mentions == 1
        assert trend.previous_mentions == 2

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_windows_are_rejected(self, bad):
        with pytest.raises(ValueError):
            compute_trends([], window_hours=bad)
        with pytest.raises(ValueError):
            compute_trends([], window_hours=24, baseline_hours=bad)


class TestMentionCounting:
    def test_a_ticker_counts_once_per_post(self):
        # The recognizer returns a set per post, so twenty shouts are one
        # mention. This is what stops a single ranting post making a trend.
        posts = [make_analyzed("a", tickers=["GME"], created_utc=NOW)]
        assert _by_symbol(_report(posts))["GME"].mentions == 1

    def test_unique_authors_excludes_deleted_and_bots(self):
        posts = [
            make_analyzed("a", tickers=["GME"], author="ape", created_utc=NOW),
            make_analyzed("b", tickers=["GME"], author="ape", created_utc=NOW),
            make_analyzed("c", tickers=["GME"], author="[deleted]", created_utc=NOW),
            make_analyzed(
                "d", tickers=["GME"], author="AutoModerator", created_utc=NOW
            ),
        ]
        trend = _by_symbol(_report(posts))["GME"]
        assert trend.mentions == 4
        assert trend.unique_authors == 1

    def test_min_mentions_filters_noise(self):
        posts = [
            make_analyzed("a", tickers=["GME", "AMC"], created_utc=NOW),
            make_analyzed("b", tickers=["GME"], created_utc=NOW),
        ]
        report = _report(posts, min_mentions=2)
        assert set(_by_symbol(report)) == {"GME"}

    def test_subreddit_and_engagement_tallies(self):
        posts = [
            make_analyzed(
                "a",
                tickers=["GME"],
                subreddit="wallstreetbets",
                score=10,
                num_comments=5,
                created_utc=NOW,
            ),
            make_analyzed(
                "b",
                tickers=["GME"],
                subreddit="stocks",
                score=20,
                num_comments=1,
                created_utc=NOW,
            ),
        ]
        trend = _by_symbol(_report(posts))["GME"]
        assert trend.subreddits == {"wallstreetbets": 1, "stocks": 1}
        assert trend.score_sum == 30
        assert trend.comment_sum == 6
        assert trend.engagement == 42  # 30 upvotes + 2x6 comments


class TestMomentum:
    def test_momentum_is_smoothed_by_one(self):
        posts = [
            *[
                make_analyzed(f"cur{i}", tickers=["GME"], created_utc=NOW - HOUR)
                for i in range(5)
            ],
            make_analyzed("old", tickers=["GME"], created_utc=NOW - 30 * HOUR),
        ]
        trend = _by_symbol(_report(posts))["GME"]
        assert trend.momentum == pytest.approx((5 - 1) / (1 + 1))
        assert trend.change_ratio == pytest.approx(5.0)

    def test_change_ratio_is_none_without_a_baseline(self):
        posts = [make_analyzed("a", tickers=["GME"], created_utc=NOW)]
        trend = _by_symbol(_report(posts))["GME"]
        assert trend.change_ratio is None
        assert trend.momentum == pytest.approx(1.0)

    def test_a_new_ticker_outranks_a_grinding_one(self):
        posts = [
            *[
                make_analyzed(f"new{i}", tickers=["NVDA"], created_utc=NOW - HOUR)
                for i in range(6)
            ],
            *[
                make_analyzed(f"cur{i}", tickers=["SPY"], created_utc=NOW - HOUR)
                for i in range(6)
            ],
            *[
                make_analyzed(f"old{i}", tickers=["SPY"], created_utc=NOW - 30 * HOUR)
                for i in range(20)
            ],
        ]
        trends = _by_symbol(_report(posts))
        assert trends["NVDA"].momentum > trends["SPY"].momentum
        assert trends["SPY"].momentum < 0


class TestClassification:
    def test_emerging_needs_a_minimum_volume(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW),
            make_analyzed("b", tickers=["GME"], created_utc=NOW),
            *[
                make_analyzed(f"n{i}", tickers=["AMC"], created_utc=NOW)
                for i in range(4)
            ],
        ]
        report = _report(posts, emerging_min_mentions=3)
        assert report.emerging == ["AMC"]
        assert _by_symbol(report)["GME"].is_emerging is False

    def test_rising_requires_an_existing_baseline(self):
        posts = [
            *[
                make_analyzed(f"c{i}", tickers=["GME"], created_utc=NOW - HOUR)
                for i in range(6)
            ],
            make_analyzed("o1", tickers=["GME"], created_utc=NOW - 30 * HOUR),
            *[
                make_analyzed(f"n{i}", tickers=["AMC"], created_utc=NOW)
                for i in range(6)
            ],
        ]
        report = _report(posts)
        assert report.rising == ["GME"]  # AMC has no baseline, so it is emerging
        assert report.emerging == ["AMC"]

    def test_fading_includes_tickers_that_went_silent(self):
        posts = [
            *[
                make_analyzed(f"o{i}", tickers=["AMC"], created_utc=NOW - 30 * HOUR)
                for i in range(10)
            ],
            make_analyzed("c", tickers=["GME"], created_utc=NOW),
        ]
        report = _report(posts)
        assert report.fading == ["AMC"]
        assert "AMC" not in _by_symbol(report)

    def test_spike_score_is_zero_for_a_flat_cohort(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW),
            make_analyzed("b", tickers=["AMC"], created_utc=NOW),
        ]
        report = _report(posts)
        assert all(t.spike_score == 0.0 for t in report.tickers)

    def test_spike_score_flags_the_outlier(self):
        posts = [
            *[
                make_analyzed(f"g{i}", tickers=["GME"], created_utc=NOW)
                for i in range(20)
            ],
            make_analyzed("a", tickers=["AMC"], created_utc=NOW),
            make_analyzed("n", tickers=["NVDA"], created_utc=NOW),
        ]
        trends = _by_symbol(_report(posts))
        assert trends["GME"].spike_score > 1.0
        assert trends["AMC"].spike_score < 0


class TestRanking:
    def test_heat_score_ranks_the_loudest_first(self):
        posts = [
            *[
                make_analyzed(
                    f"g{i}",
                    tickers=["GME"],
                    author=f"ape{i}",
                    score=100,
                    created_utc=NOW,
                )
                for i in range(10)
            ],
            make_analyzed("a", tickers=["AMC"], author="solo", created_utc=NOW),
        ]
        report = _report(posts)
        assert [t.symbol for t in report.tickers] == ["GME", "AMC"]
        assert report.tickers[0].heat_score > report.tickers[1].heat_score

    def test_weights_are_overridable(self):
        posts = [
            *[
                make_analyzed(f"g{i}", tickers=["GME"], created_utc=NOW - HOUR)
                for i in range(4)
            ],
            *[
                make_analyzed(f"go{i}", tickers=["GME"], created_utc=NOW - 30 * HOUR)
                for i in range(10)
            ],
            make_analyzed("n1", tickers=["NVDA"], created_utc=NOW),
            make_analyzed("n2", tickers=["NVDA"], created_utc=NOW),
            make_analyzed("n3", tickers=["NVDA"], created_utc=NOW),
        ]
        loudest = _report(posts, weights={"mentions": 1.0, "momentum": 0.0})
        fastest = _report(posts, weights={"mentions": 0.0, "momentum": 1.0})
        assert loudest.tickers[0].symbol == "GME"
        assert fastest.tickers[0].symbol == "NVDA"

    def test_top_n_truncates(self):
        posts = [
            make_analyzed(f"p{i}", tickers=[f"AA{chr(65 + i)}"], created_utc=NOW)
            for i in range(5)
        ]
        assert len(_report(posts, top_n=2).tickers) == 2

    def test_ordering_is_stable_for_identical_tickers(self):
        posts = [
            make_analyzed("a", tickers=["AMC"], created_utc=NOW),
            make_analyzed("b", tickers=["GME"], created_utc=NOW),
        ]
        assert [t.symbol for t in _report(posts).tickers] == ["AMC", "GME"]

    def test_sample_posts_are_the_most_engaged(self):
        posts = [
            make_analyzed("quiet", tickers=["GME"], score=1, created_utc=NOW),
            make_analyzed("loud", tickers=["GME"], score=9000, created_utc=NOW),
        ]
        trend = _by_symbol(_report(posts, max_sample_posts=1))["GME"]
        assert [p["id"] for p in trend.sample_posts] == ["loud"]


class TestSentiment:
    def test_ticker_sentiment_averages_its_posts(self):
        posts = [
            make_analyzed(
                "a",
                tickers=["GME"],
                sentiment="bullish",
                sentiment_score=0.9,
                created_utc=NOW,
            ),
            make_analyzed(
                "b",
                tickers=["GME"],
                sentiment="bearish",
                sentiment_score=-0.3,
                created_utc=NOW,
            ),
        ]
        trend = _by_symbol(_report(posts))["GME"]
        assert trend.sentiment_score == pytest.approx(0.3)
        assert trend.sentiment == "bullish"
        assert trend.sentiment_breakdown == {"bullish": 1, "bearish": 1}

    def test_mixed_sentiment_reads_neutral(self):
        posts = [
            make_analyzed(
                "a",
                tickers=["GME"],
                sentiment="bullish",
                sentiment_score=0.5,
                created_utc=NOW,
            ),
            make_analyzed(
                "b",
                tickers=["GME"],
                sentiment="bearish",
                sentiment_score=-0.5,
                created_utc=NOW,
            ),
        ]
        assert _by_symbol(_report(posts))["GME"].sentiment == "neutral"

    def test_report_mood_reflects_the_window_only(self):
        posts = [
            make_analyzed(
                "old",
                sentiment="bearish",
                sentiment_score=-0.9,
                created_utc=NOW - 30 * HOUR,
            ),
            make_analyzed(
                "new", sentiment="bullish", sentiment_score=0.9, created_utc=NOW
            ),
        ]
        report = _report(posts)
        assert report.mood == "bullish"
        assert report.sentiment_breakdown == {"bullish": 1}


class TestBreakdowns:
    def test_per_subreddit_summary(self):
        posts = [
            make_analyzed(
                "a",
                tickers=["GME"],
                subreddit="wallstreetbets",
                sentiment="bullish",
                sentiment_score=0.8,
                created_utc=NOW,
            ),
            make_analyzed(
                "b",
                subreddit="wallstreetbets",
                sentiment="bullish",
                sentiment_score=0.6,
                created_utc=NOW,
            ),
            make_analyzed("c", tickers=["SPY"], subreddit="stocks", created_utc=NOW),
        ]
        summaries = {s.subreddit: s for s in _report(posts).by_subreddit}
        wsb = summaries["wallstreetbets"]
        assert wsb.posts == 2
        assert wsb.posts_with_tickers == 1
        assert wsb.mood == "bullish"
        assert wsb.top_tickers == {"GME": 1}
        assert summaries["stocks"].top_tickers == {"SPY": 1}

    def test_timeline_buckets_mentions_per_hour(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW - 30 * 60),
            make_analyzed("b", tickers=["GME"], created_utc=NOW - 90 * 60),
        ]
        report = _report(posts, window_hours=3.0, timeline_bucket_seconds=3600)
        series = report.timeline.series["GME"]
        assert len(series) == 3
        assert sum(series) == 2
        assert series[-1] == 1  # the most recent bucket holds the newest post

    def test_timeline_only_covers_ranked_tickers(self):
        posts = [
            make_analyzed("a", tickers=["GME"], created_utc=NOW),
            make_analyzed("b", tickers=["AMC"], created_utc=NOW),
        ]
        report = _report(posts, timeline_symbols=1)
        assert len(report.timeline.series) == 1


class TestEdges:
    def test_empty_input_yields_an_empty_report(self):
        report = _report([])
        assert report.tickers == []
        assert report.mood == "neutral"
        assert report.posts_analyzed == 0
        assert report.timeline.series == {}

    def test_posts_without_tickers_still_count_towards_mood(self):
        posts = [
            make_analyzed(
                "a", sentiment="bullish", sentiment_score=0.9, created_utc=NOW
            )
        ]
        report = _report(posts)
        assert report.tickers == []
        assert report.mood == "bullish"

    def test_subreddits_default_to_those_seen_in_the_posts(self):
        posts = [
            make_analyzed("a", subreddit="stocks", created_utc=NOW),
            make_analyzed("b", subreddit="options", created_utc=NOW),
        ]
        assert _report(posts).subreddits == ["options", "stocks"]

    def test_report_is_json_serializable(self):
        import json

        posts = [make_analyzed("a", tickers=["GME"], created_utc=NOW)]
        payload = json.dumps(_report(posts).to_dict())
        assert '"GME"' in payload


class TestPerTickerSentiment:
    def test_attributed_sentiment_beats_the_post_label(self):
        # One post, long NVDA and short INTC. Without attribution both would
        # inherit the post's bullish label and INTC would read bullish.
        posts = [
            make_analyzed(
                "a",
                tickers=["NVDA", "INTC"],
                sentiment="bullish",
                sentiment_score=0.7,
                ticker_sentiment={"NVDA": 0.9, "INTC": -0.8},
                created_utc=NOW,
            )
        ]
        trends = _by_symbol(_report(posts))
        assert trends["NVDA"].sentiment == "bullish"
        assert trends["INTC"].sentiment == "bearish"
        assert trends["INTC"].sentiment_score == pytest.approx(-0.8)
        assert trends["INTC"].sentiment_breakdown == {"bearish": 1}

    def test_unattributed_tickers_still_use_the_post_label(self):
        posts = [
            make_analyzed(
                "a",
                tickers=["NVDA", "INTC"],
                sentiment="bullish",
                sentiment_score=0.7,
                ticker_sentiment={"INTC": -0.8},
                created_utc=NOW,
            )
        ]
        trends = _by_symbol(_report(posts))
        assert trends["NVDA"].sentiment_score == pytest.approx(0.7)
        assert trends["NVDA"].sentiment_breakdown == {"bullish": 1}

    def test_subreddit_mood_stays_post_level(self):
        # A subreddit's mood is about its posts, not about any one ticker.
        posts = [
            make_analyzed(
                "a",
                tickers=["NVDA", "INTC"],
                sentiment="bullish",
                sentiment_score=0.7,
                ticker_sentiment={"INTC": -0.8},
                created_utc=NOW,
            )
        ]
        assert _report(posts).by_subreddit[0].mood == "bullish"
