"""Tests for the Reddit listing client (anonymous JSON backend)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from conftest import HOUR, NOW

from reddit_stock_analyzer.client import RedditClient, normalize_submission


def _child(**data: object) -> dict:
    base = {
        "id": "p1",
        "subreddit": "wallstreetbets",
        "title": "GME to the moon",
        "selftext": "body",
        "author": "ape",
        "score": 100,
        "num_comments": 20,
        "upvote_ratio": 0.9,
        "created_utc": NOW,
        "permalink": "/r/wallstreetbets/comments/p1/",
        "is_self": True,
    }
    base.update(data)
    return {"kind": "t3", "data": base}


def _listing(*children: dict) -> dict:
    return {"data": {"children": list(children)}}


def _client_for(handler) -> tuple[RedditClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return RedditClient(http, use_praw=False, max_retries=0), http


class TestNormalizeSubmission:
    def test_builds_absolute_url_from_permalink(self):
        post = normalize_submission(_child()["data"])
        assert post.url == "https://www.reddit.com/r/wallstreetbets/comments/p1/"
        assert post.text == "GME to the moon\n\nbody"

    def test_unescapes_html_entities(self):
        post = normalize_submission(_child(title="Puts &amp; calls")["data"])
        assert post.title == "Puts & calls"

    def test_engagement_weights_comments_double(self):
        post = normalize_submission(_child(score=10, num_comments=5)["data"])
        assert post.engagement == 20

    def test_collects_gallery_images(self):
        post = normalize_submission(
            _child(
                is_self=False,
                is_gallery=True,
                url="https://reddit.com/gallery/p1",
                media_metadata={
                    "a": {"s": {"u": "https://i.redd.it/a.jpg?x=1&amp;y=2"}}
                },
            )["data"]
        )
        assert post.image_urls == ["https://i.redd.it/a.jpg?x=1&y=2"]

    def test_coerces_malformed_numbers(self):
        post = normalize_submission(
            _child(score="not-a-number", created_utc=None)["data"]
        )
        assert post.score == 0
        assert post.created_utc == 0.0


class TestFetchPosts:
    async def test_parses_listing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/r/wallstreetbets/hot.json"
            assert request.url.params["limit"] == "5"
            return httpx.Response(200, json=_listing(_child()))

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_posts("wallstreetbets", limit=5)
        finally:
            await http.aclose()

        assert [p.id for p in posts] == ["p1"]

    async def test_skips_stickied_posts(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=_listing(_child(id="pinned", stickied=True), _child())
            )

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_posts("wallstreetbets")
        finally:
            await http.aclose()

        assert [p.id for p in posts] == ["p1"]

    async def test_max_age_hours_drops_old_posts(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_listing(
                    _child(id="fresh", created_utc=NOW - HOUR),
                    _child(id="stale", created_utc=NOW - 30 * HOUR),
                ),
            )

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_posts(
                "wallstreetbets", max_age_hours=24, now=NOW
            )
        finally:
            await http.aclose()

        assert [p.id for p in posts] == ["fresh"]

    async def test_time_filter_is_sent_for_top(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.url.params)
            return httpx.Response(200, json=_listing())

        client, http = _client_for(handler)
        try:
            await client.fetch_posts("stocks", sort="top", time_filter="week")
        finally:
            await http.aclose()

        assert seen["t"] == "week"

    async def test_invalid_subreddit_name_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not have made a request")

        client, http = _client_for(handler)
        try:
            assert await client.fetch_posts("no") == []
        finally:
            await http.aclose()

    async def test_rejects_unknown_sort(self):
        client, http = _client_for(lambda request: httpx.Response(200, json=_listing()))
        try:
            with pytest.raises(ValueError, match="sort must be one of"):
                await client.fetch_posts("stocks", sort="spicy")
        finally:
            await http.aclose()

    async def test_http_error_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        client, http = _client_for(handler)
        try:
            assert await client.fetch_posts("stocks") == []
        finally:
            await http.aclose()

    async def test_retries_rate_limited_response(self, monkeypatch: pytest.MonkeyPatch):
        real_sleep = asyncio.sleep

        async def instant_sleep(_delay: float) -> None:
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429)
            return httpx.Response(200, json=_listing())

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = RedditClient(http, use_praw=False, max_retries=1)
        try:
            await client.fetch_posts("stocks")
        finally:
            await http.aclose()

        assert attempts["n"] == 2

    async def test_non_json_body_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>blocked</html>")

        client, http = _client_for(handler)
        try:
            assert await client.fetch_posts("stocks") == []
        finally:
            await http.aclose()


class TestFetchMany:
    async def test_dedupes_across_sorts_keeping_higher_score(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/new.json"):
                return httpx.Response(200, json=_listing(_child(id="p1", score=1)))
            return httpx.Response(200, json=_listing(_child(id="p1", score=500)))

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_many(["wallstreetbets"], sorts=("hot", "new"))
        finally:
            await http.aclose()

        assert len(posts) == 1
        assert posts[0].score == 500

    async def test_merges_subreddits_newest_first(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "stocks" in request.url.path:
                return httpx.Response(
                    200,
                    json=_listing(
                        _child(id="old", subreddit="stocks", created_utc=NOW - 5 * HOUR)
                    ),
                )
            return httpx.Response(200, json=_listing(_child(id="new", created_utc=NOW)))

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_many(["wallstreetbets", "stocks"])
        finally:
            await http.aclose()

        assert [p.id for p in posts] == ["new", "old"]

    async def test_one_failing_subreddit_does_not_sink_the_batch(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "stocks" in request.url.path:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json=_listing(_child()))

        client, http = _client_for(handler)
        try:
            posts = await client.fetch_many(["wallstreetbets", "stocks"])
        finally:
            await http.aclose()

        assert [p.id for p in posts] == ["p1"]

    async def test_no_targets_returns_empty(self):
        client, http = _client_for(lambda request: httpx.Response(200, json=_listing()))
        try:
            assert await client.fetch_many([]) == []
        finally:
            await http.aclose()


async def test_close_is_safe_without_a_session():
    client = RedditClient(use_praw=False)
    await client.close()
