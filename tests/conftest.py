"""Shared fixtures and fakes.

The suite never touches the network and never loads a model: the Reddit client
runs against ``httpx.MockTransport`` and both ML components are injected as
lightweight fakes.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pytest

from reddit_stock_analyzer.models import AnalyzedPost, RedditPost

#: 2024-01-15 12:00:00 UTC — a fixed "now" so window maths is deterministic.
NOW = 1705320000.0
HOUR = 3600.0


class FakeRecognizer:
    """Stand-in for ``stock_recognizer.StockRecognizer``.

    Recognises ``$CASHTAGS`` plus any symbol in *known*, which is enough to
    exercise every code path without loading market data or a 400 MB model.
    """

    def __init__(self, known: Iterable[str] = ()) -> None:
        self.known = {s.upper() for s in known}
        self.recognize_calls: list[str] = []
        self.recognize_ai_calls: list[str] = []

    def _find(self, text: str) -> list[str]:
        found = {m.upper() for m in re.findall(r"\$([A-Za-z]{1,6})\b", text)}
        found |= {w for w in re.findall(r"\b[A-Z]{1,6}\b", text) if w in self.known}
        return sorted(found)

    def recognize(self, text: str) -> list[str]:
        self.recognize_calls.append(text)
        return self._find(text)

    def recognize_ai(self, text: str) -> list[str]:
        self.recognize_ai_calls.append(text)
        # The "AI" contribution: map a company name onto its ticker.
        found = set(self._find(text))
        if "tsmc" in text.lower():
            found.add("TSM")
        return sorted(found)


class FakePipeline:
    """Stand-in for a ``transformers`` text-classification pipeline."""

    def __init__(self, labels: dict[str, tuple[str, float]] | None = None) -> None:
        self.labels = labels or {}
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str], **kwargs: object) -> list[dict]:
        self.calls.append(list(texts))
        results = []
        for text in texts:
            label, score = self.labels.get(text, ("NEUTRAL", 0.5))
            results.append({"label": label, "score": score})
        return results


def make_post(
    post_id: str = "abc",
    *,
    subreddit: str = "wallstreetbets",
    title: str = "A post",
    selftext: str = "",
    author: str = "someone",
    score: int = 10,
    num_comments: int = 5,
    created_utc: float = NOW,
    stickied: bool = False,
) -> RedditPost:
    """Build a :class:`RedditPost` with sensible defaults."""
    return RedditPost(
        id=post_id,
        subreddit=subreddit,
        title=title,
        selftext=selftext,
        author=author,
        score=score,
        num_comments=num_comments,
        created_utc=created_utc,
        url=f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/",
        permalink=f"/r/{subreddit}/comments/{post_id}/",
        stickied=stickied,
    )


def make_analyzed(
    post_id: str = "abc",
    *,
    tickers: Sequence[str] = (),
    sentiment: str = "neutral",
    sentiment_score: float = 0.0,
    **post_kwargs: object,
) -> AnalyzedPost:
    """Build an :class:`AnalyzedPost` with sensible defaults."""
    return AnalyzedPost(
        post=make_post(post_id, **post_kwargs),  # type: ignore[arg-type]
        tickers=list(tickers),
        sentiment=sentiment,  # type: ignore[arg-type]
        sentiment_score=sentiment_score,
        sentiment_confidence=abs(sentiment_score),
    )


@pytest.fixture(autouse=True)
def _no_reddit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real ``.env`` out of the tests."""
    for name in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_PERSONAL_USE",
        "REDDIT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    return FakeRecognizer(known={"GME", "AMC", "NVDA", "SPY", "TSM"})
