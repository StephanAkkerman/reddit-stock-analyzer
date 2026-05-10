from collections import Counter

from .analyzer import SentimentAnalyzer
from .scraper import RedditScraper


async def get_subreddit_overview(subreddit: str, limit: int = 50):
    scraper = RedditScraper()
    analyzer = SentimentAnalyzer()

    posts = await scraper.fetch_posts(subreddit, limit=limit)
    if not posts:
        return {"error": "No posts found"}

    # Run sentiment on all titles (faster than full text for an overview)
    titles = [p["title"] for p in posts]
    sentiments = analyzer.analyze_batch(titles)

    for i, post in enumerate(posts):
        post["sentiment"] = sentiments[i]["label"]
        post["confidence"] = sentiments[i]["score"]

    # Aggregate Ticker Mentions
    all_tickers = [t for p in posts for t in p["tickers"]]
    ticker_counts = Counter(all_tickers)

    # Calculate Subreddit Mood
    sentiment_counts = Counter([p["sentiment"] for p in posts])

    return {
        "subreddit": subreddit,
        "sample_size": len(posts),
        "overall_mood": sentiment_counts.most_common(1)[0][0],
        "sentiment_breakdown": dict(sentiment_counts),
        "top_tickers": dict(ticker_counts.most_common(10)),
        "data": posts,
    }
