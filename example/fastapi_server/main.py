from core.aggregator import get_subreddit_overview
from fastapi import FastAPI

app = FastAPI()


@app.get("/api/reddit/{subreddit}")
async def read_subreddit(subreddit: str, limit: int = 50):
    # The API doesn't care HOW the data is fetched or analyzed,
    # it just calls the core function.
    data = await get_subreddit_overview(subreddit, limit)
    return data
