import os

import asyncpraw
from dotenv import load_dotenv

from .utils import extract_tickers_gliner


class RedditScraper:
    def __init__(self):
        # Load the .env file with Reddit credentials
        load_dotenv()
        self.reddit = None

    async def _init_reddit(self):
        if self.reddit is None:
            self.reddit = asyncpraw.Reddit(
                client_id=os.getenv("REDDIT_PERSONAL_USE"),
                client_secret=os.getenv("REDDIT_SECRET"),
                user_agent=os.getenv("REDDIT_APP_NAME"),
                username=os.getenv("REDDIT_USERNAME"),
                password=os.getenv("REDDIT_PASSWORD"),
            )

    async def fetch_posts(
        self, subreddit_name: str, limit: int = 50, tickers_only: bool = False
    ) -> list[dict]:
        # Ensure the client is initialized before fetching
        await self._init_reddit()

        subreddit = await self.reddit.subreddit(subreddit_name)
        posts = []

        async for submission in subreddit.hot(limit=limit):
            if submission.stickied:
                continue

            combined_text = f"{submission.title} {submission.selftext}"
            tickers = extract_tickers_gliner(combined_text)

            if tickers_only and not tickers:
                continue

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

    async def close(self):
        """Clean up the aiohttp session to prevent 'Unclosed client session' warnings."""
        if self.reddit is not None:
            await self.reddit.close()


if __name__ == "__main__":
    import asyncio

    scraper = RedditScraper()
    posts = asyncio.run(scraper.fetch_posts("wallstreetbets", limit=50))
    for post in posts:
        print(f"{post['title']} - Tickers: {post['tickers']}")

    # Close the scraper to clean up resources
    asyncio.run(scraper.close())
