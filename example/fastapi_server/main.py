"""FastAPI wiring example — the shape `fintwit-web` should use.

The important part is the lifespan handler: :class:`RedditTrendService` owns an
HTTP session and two loaded models, so it is built once at startup and reused,
never per request.

Run it with::

    pip install "reddit-stock-analyzer[praw]" fastapi uvicorn
    uvicorn example.fastapi_server.main:app --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from reddit_stock_analyzer import (
    DEFAULT_SUBREDDITS,
    RedditTrendService,
    SUBREDDIT_CATEGORIES,
    subreddits_for,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the service once, and warm the models before serving traffic."""
    service = RedditTrendService()
    # Optional: pay the model load cost at startup instead of on the first
    # request. Skip it if you would rather start fast.
    service.analyzer.warm_up()
    app.state.reddit = service
    try:
        yield
    finally:
        await service.close()


app = FastAPI(title="reddit-stock-analyzer example", lifespan=lifespan)


@app.get("/api/reddit/categories")
async def list_categories() -> dict:
    """The subreddit catalogue, for populating a filter in the UI."""
    return {
        "default": list(DEFAULT_SUBREDDITS),
        "categories": {k: list(v) for k, v in SUBREDDIT_CATEGORIES.items()},
    }


@app.get("/api/reddit/trends")
async def read_trends(
    subreddit: Annotated[list[str] | None, Query()] = None,
    category: str | None = None,
    window_hours: Annotated[float, Query(gt=0, le=168)] = 24.0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    top_n: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict:
    """Ranked tickers across one or more subreddits."""
    if category is not None:
        try:
            targets = list(subreddits_for(category))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        targets = subreddit or list(DEFAULT_SUBREDDITS)

    report = await app.state.reddit.trend_report(
        targets, window_hours=window_hours, limit=limit, top_n=top_n
    )
    return report.to_dict()


@app.get("/api/reddit/{subreddit}")
async def read_subreddit(
    subreddit: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict:
    """Sentiment and ticker snapshot for a single subreddit."""
    overview = await app.state.reddit.subreddit_overview(subreddit, limit=limit)
    if not overview["posts"]:
        raise HTTPException(status_code=404, detail=f"No posts for r/{subreddit}")
    return overview
