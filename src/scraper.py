import os

import asyncpraw

from .utils import extract_tickers


class RedditScraper:
    def __init__(self):
        self.reddit = asyncpraw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="reddit-sentiment-core-v1",
        )

    async def fetch_posts(self, subreddit_name: str, limit: int = 50):
        subreddit = await self.reddit.subreddit(subreddit_name)
        posts = []
        async for submission in subreddit.hot(limit=limit):
            if submission.stickied:
                continue

            combined_text = f"{submission.title} {submission.selftext}"
            tickers = extract_tickers(combined_text)

            posts.append(
                {
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "url": f"https://reddit.com{submission.permalink}",
                    "tickers": tickers,
                    "created_utc": submission.created_utc,
                }
            )
        return posts
