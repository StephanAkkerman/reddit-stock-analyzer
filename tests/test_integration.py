"""End-to-end: real client + real analyzer + real trend maths, faked network.

Every other module is tested in isolation; this one checks that they actually
fit together — a listing response in, a ranked trend report out.
"""

from __future__ import annotations

import httpx
from conftest import HOUR, NOW, FakePipeline, FakeRecognizer

from reddit_stock_analyzer.aggregator import RedditTrendService
from reddit_stock_analyzer.analyzer import PostAnalyzer
from reddit_stock_analyzer.client import RedditClient
from reddit_stock_analyzer.recognizer import TickerExtractor
from reddit_stock_analyzer.sentiment import SentimentAnalyzer

_POSTS = {
    "wallstreetbets": [
        ("w1", "$NVDA earnings run", NOW - 1 * HOUR, 500, 120, "ape"),
        ("w2", "NVDA is the whole market now", NOW - 2 * HOUR, 200, 40, "bull"),
        ("w3", "$NVDA calls printing", NOW - 3 * HOUR, 50, 10, "degen"),
        ("w4", "$GME again", NOW - 4 * HOUR, 10, 2, "ape"),
        ("w5", "$GME baseline chatter", NOW - 30 * HOUR, 10, 2, "ape"),
        ("w6", "$GME more baseline", NOW - 31 * HOUR, 10, 2, "bull"),
        ("w7", "$GME even more baseline", NOW - 32 * HOUR, 10, 2, "degen"),
    ],
    "stocks": [
        ("s1", "Is TSMC still cheap?", NOW - 5 * HOUR, 80, 30, "valueguy"),
        ("s2", "daily discussion", NOW - 1 * HOUR, 5, 900, "AutoModerator"),
    ],
}


def _handler(request: httpx.Request) -> httpx.Response:
    subreddit = request.url.path.split("/")[2]
    children = [
        {
            "kind": "t3",
            "data": {
                "id": post_id,
                "subreddit": subreddit,
                "title": title,
                "selftext": "",
                "author": author,
                "score": score,
                "num_comments": comments,
                "created_utc": created,
                "permalink": f"/r/{subreddit}/comments/{post_id}/",
                "is_self": True,
                "stickied": post_id == "s2",
            },
        }
        for post_id, title, created, score, comments, author in _POSTS.get(
            subreddit, []
        )
    ]
    return httpx.Response(200, json={"data": {"children": children}})


def _service(http: httpx.AsyncClient) -> RedditTrendService:
    return RedditTrendService(
        client=RedditClient(http, use_praw=False, max_retries=0),
        analyzer=PostAnalyzer(
            extractor=TickerExtractor(
                recognizer=FakeRecognizer(known={"NVDA", "GME", "TSM"}), use_ai=True
            ),
            sentiment=SentimentAnalyzer(
                pipeline=FakePipeline(
                    {
                        "$NVDA earnings run": ("BULLISH", 0.95),
                        "NVDA is the whole market now": ("BULLISH", 0.8),
                        "$NVDA calls printing": ("BULLISH", 0.9),
                        "Is TSMC still cheap?": ("BEARISH", 0.55),
                    }
                )
            ),
        ),
    )


async def test_scrape_to_ranked_report():
    http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    try:
        async with _service(http) as service:
            report = await service.trend_report(
                ["wallstreetbets", "stocks"],
                window_hours=24.0,
                now=NOW,
                emerging_min_mentions=3,
            )
    finally:
        await http.aclose()

    trends = {t.symbol: t for t in report.tickers}

    # Ranked by heat: NVDA is loudest, most engaged and newest.
    assert report.tickers[0].symbol == "NVDA"
    assert trends["NVDA"].mentions == 3
    assert trends["NVDA"].unique_authors == 3
    assert trends["NVDA"].previous_mentions == 0
    assert trends["NVDA"].is_emerging is True
    assert trends["NVDA"].sentiment == "bullish"

    # GME was busier yesterday than today: present, but fading.
    assert trends["GME"].mentions == 1
    assert trends["GME"].previous_mentions == 3
    assert trends["GME"].momentum < 0
    assert "GME" in report.fading

    # The AI path mapped "TSMC" onto TSM, from a subreddit that never wrote
    # the ticker itself.
    assert trends["TSM"].subreddits == {"stocks": 1}

    # The stickied daily discussion never entered the corpus despite its 900
    # comments, so it cannot dominate engagement.
    assert all("s2" != p["id"] for t in report.tickers for p in t.sample_posts)
    assert report.posts_in_window == 5

    # Provenance and shape.
    assert report.subreddits == ["wallstreetbets", "stocks"]
    assert {s.subreddit for s in report.by_subreddit} == {"wallstreetbets", "stocks"}
    assert len(report.timeline.series["NVDA"]) == 24
    assert sum(report.timeline.series["NVDA"]) == 3


async def test_unreachable_reddit_yields_an_empty_report():
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network is down")

    http = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    try:
        async with _service(http) as service:
            report = await service.trend_report(["wallstreetbets"], now=NOW)
    finally:
        await http.aclose()

    assert report.tickers == []
    assert report.posts_analyzed == 0
    assert report.mood == "neutral"
