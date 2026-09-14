# Using this package in `fintwit-web`

`fintwit-web` currently scrapes Reddit itself in `app/services/reddit_service.py`
(hot posts, normalised to dicts) and counts symbols in
`app/services/mention_aggregator.py`. This package is the extracted, tested
version of that pipeline plus the trend layer, so the app can delete its own
copy and gain momentum/spike analytics for free.

It follows `fintwit-web`'s architecture rules: no Discord, no DataFrames as a
transport, pure dataclasses with `to_dict()` that map cleanly onto Pydantic
schemas, and async throughout with the CPU-bound model work pushed off the
event loop.

## Install

```toml
# pyproject.toml
dependencies = [
    "reddit-stock-analyzer @ git+https://github.com/StephanAkkerman/reddit-stock-analyzer.git",
]
```

Note that the core dependency `stock-recognizer` pulls in `torch` and
`transformers`. That matches where the existing ML stack already lives
(`app/ml/`), but it means **this package does not belong in `fintwit-web`'s
`[test]` extra** — that extra deliberately excludes torch. Import it lazily
inside functions, exactly as `app/ml/` already does, so the test suite never
touches it.

## The 7-step migration order

### 1. Schema (`app/schemas/`)

`TickerTrend.to_dict()` and `TrendReport.to_dict()` are the wire format. A
Pydantic mirror:

```python
class RedditTickerTrend(BaseModel):
    symbol: str
    mentions: int
    previous_mentions: int
    unique_authors: int
    subreddits: dict[str, int]
    engagement: int
    momentum: float
    change_ratio: float | None
    spike_score: float
    heat_score: float
    sentiment: Literal["bullish", "bearish", "neutral"]
    sentiment_score: float
    is_emerging: bool
    sample_posts: list[dict]
```

### 2. Service (`app/services/reddit_service.py`)

Delegate instead of re-implementing. The service keeps one
`RedditTrendService` for the lifetime of the app — it owns the HTTP session
and both loaded models, so building one per request would reload the
recognizer's market data every time.

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _service():
    from reddit_stock_analyzer import RedditClient, RedditTrendService

    return RedditTrendService(client=RedditClient())


async def get_reddit_trends(window_hours: float = 24.0) -> dict:
    report = await _service().trend_report(window_hours=window_hours)
    return report.to_dict()
```

The existing `get_reddit_hot_posts()` signature maps onto
`RedditClient.fetch_posts(subreddit, sort="hot", limit=...)`, which does the
same html-unescaping, gallery/preview image extraction and stickied filtering.

### 3. Repository (`app/infra/repos.py`)

Persist one row per `(symbol, snapshot_at)` so trends survive restarts and can
be charted over weeks rather than the 48h a live scrape can see. `compute_trends`
only ever looks at the posts handed to it, so history is the app's job, not the
library's.

### 4. Runtime worker (`app/runtime/`)

```python
async def run_reddit_trends(interval_seconds: int = 900) -> None:
    while True:
        try:
            report = await get_reddit_trends()
            await repo.save_snapshot(report)
            await broadcast.publish("reddit.trends", report)
        except Exception:
            logger.exception("[reddit] trend worker iteration failed")
        await asyncio.sleep(interval_seconds)
```

15 minutes is a reasonable floor: it is well inside Reddit's rate limits for
six subreddits × two listings, and mention counts do not move meaningfully
faster than that.

Warm the models during lifespan startup (`service.analyzer.warm_up()`) so the
first request does not pay for the load.

### 5. API endpoint (`app/api/`)

```
GET /api/reddit/trends?window_hours=24&category=retail
GET /api/reddit/{subreddit}
```

See [`example/fastapi_server/main.py`](../example/fastapi_server/main.py) for a
complete, runnable version including the lifespan handler.

### 6–7. Frontend

`heat_score` drives ordering, `momentum` and `spike_score` drive the
badge/arrow, `timeline.series[symbol]` is a ready-made sparkline, and
`subreddits` per ticker shows whether hype has spread past r/wallstreetbets.
`sample_posts` gives each row a drill-down without a second request.

## Mapping the existing code

| `fintwit-web` today | Replacement |
| --- | --- |
| `reddit_service.get_reddit_hot_posts` | `RedditClient.fetch_posts` |
| `reddit_service._normalize_post_payload` | `client.normalize_submission` |
| `reddit_service._reddit_credentials_from_env` | `config.reddit_credentials_from_env` (accepts the same legacy names) |
| ticker counting in `mention_aggregator` | `compute_trends(...)` |
| — | `rising` / `emerging` / `fading`, momentum, spike z-scores, timeline |

`mention_aggregator.py` also aggregates Twitter mentions, so it should keep its
own cross-source rollup — only its Reddit leg is replaced.
