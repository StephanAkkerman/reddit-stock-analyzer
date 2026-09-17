"""Tests for the high-level service, with the network faked out."""

from __future__ import annotations

from typing import Sequence

from conftest import HOUR, NOW, FakePipeline, FakeRecognizer, make_post

from reddit_stock_analyzer.aggregator import RedditTrendService
from reddit_stock_analyzer.analyzer import PostAnalyzer
from reddit_stock_analyzer.models import RedditPost
from reddit_stock_analyzer.recognizer import TickerExtractor
from reddit_stock_analyzer.sentiment import SentimentAnalyzer


class FakeClient:
    """Records what was asked for and replays canned posts."""

    def __init__(self, posts: Sequence[RedditPost] = ()) -> None:
        self.posts = list(posts)
        self.fetch_many_calls: list[dict] = []
        self.fetch_posts_calls: list[dict] = []
        self.closed = False

    async def fetch_many(self, subreddits, **kwargs) -> list[RedditPost]:
        self.fetch_many_calls.append({"subreddits": list(subreddits), **kwargs})
        return self.posts

    async def fetch_posts(self, subreddit, **kwargs) -> list[RedditPost]:
        self.fetch_posts_calls.append({"subreddit": subreddit, **kwargs})
        return [p for p in self.posts if p.subreddit == subreddit]

    async def close(self) -> None:
        self.closed = True


def _service(posts: Sequence[RedditPost] = ()) -> tuple[RedditTrendService, FakeClient]:
    client = FakeClient(posts)
    analyzer = PostAnalyzer(
        extractor=TickerExtractor(recognizer=FakeRecognizer(), use_ai=False),
        sentiment=SentimentAnalyzer(
            pipeline=FakePipeline({"$GME to the moon": ("BULLISH", 0.9)})
        ),
    )
    return RedditTrendService(client=client, analyzer=analyzer), client  # type: ignore[arg-type]


class TestScrape:
    async def test_enriches_fetched_posts(self):
        service, _ = _service([make_post("a", title="$GME to the moon")])
        analyzed = await service.scrape(["wallstreetbets"])
        assert analyzed[0].tickers == ["GME"]
        assert analyzed[0].sentiment == "bullish"

    async def test_defaults_to_the_catalogue(self):
        from reddit_stock_analyzer.config import DEFAULT_SUBREDDITS

        service, client = _service()
        await service.scrape()
        assert client.fetch_many_calls[0]["subreddits"] == list(DEFAULT_SUBREDDITS)

    async def test_passes_the_recency_window_through(self):
        service, client = _service()
        await service.scrape(["stocks"], max_age_hours=6.0, now=NOW)
        assert client.fetch_many_calls[0]["max_age_hours"] == 6.0
        assert client.fetch_many_calls[0]["now"] == NOW


class TestTrendReport:
    async def test_widens_the_scrape_to_cover_the_baseline(self):
        service, client = _service()
        await service.trend_report(["stocks"], window_hours=12.0, now=NOW)
        # 12h window + 12h baseline: scraping only 12h would make every ticker
        # look like it just appeared.
        assert client.fetch_many_calls[0]["max_age_hours"] == 24.0

    async def test_explicit_baseline_widens_further(self):
        service, client = _service()
        await service.trend_report(
            ["stocks"], window_hours=6.0, baseline_hours=48.0, now=NOW
        )
        assert client.fetch_many_calls[0]["max_age_hours"] == 54.0

    async def test_ranks_the_scraped_posts(self):
        posts = [
            make_post(f"g{i}", title="$GME to the moon", created_utc=NOW - HOUR)
            for i in range(3)
        ] + [make_post("s", title="$SPY chop", created_utc=NOW - HOUR)]
        service, _ = _service(posts)
        report = await service.trend_report(["wallstreetbets"], now=NOW)
        assert [t.symbol for t in report.tickers] == ["GME", "SPY"]
        assert report.subreddits == ["wallstreetbets"]
        assert report.mood == "bullish"

    async def test_forwards_trend_options(self):
        posts = [
            make_post("a", title="$GME to the moon", created_utc=NOW),
            make_post("b", title="$SPY chop", created_utc=NOW),
        ]
        service, _ = _service(posts)
        report = await service.trend_report(["wallstreetbets"], now=NOW, top_n=1)
        assert len(report.tickers) == 1


class TestSubredditOverview:
    async def test_summarizes_one_subreddit(self):
        posts = [
            make_post("a", title="$GME to the moon"),
            make_post("b", title="$GME again"),
            make_post("c", title="no tickers here"),
        ]
        service, _ = _service(posts)
        overview = await service.subreddit_overview("wallstreetbets", limit=10)
        assert overview["subreddit"] == "wallstreetbets"
        assert overview["sample_size"] == 3
        assert overview["top_tickers"] == {"GME": 2}
        assert len(overview["posts"]) == 3
        assert overview["posts"][0]["tickers"] == ["GME"]

    async def test_empty_subreddit_is_neutral(self):
        service, _ = _service()
        overview = await service.subreddit_overview("stocks")
        assert overview["sample_size"] == 0
        assert overview["overall_mood"] == "neutral"
        assert overview["top_tickers"] == {}


async def test_context_manager_closes_the_client():
    service, client = _service()
    async with service:
        pass
    assert client.closed is True
