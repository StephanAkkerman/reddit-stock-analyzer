"""Scrape finance subreddits, recognise the stocks discussed, and rank trends."""

from .aggregator import (
    DEFAULT_SORTS,
    RedditTrendService,
    get_subreddit_overview,
    get_trend_report,
    get_trending_tickers,
)
from .analyzer import PostAnalyzer
from .client import RedditClient
from .config import (
    DEFAULT_SUBREDDITS,
    SUBREDDIT_CATEGORIES,
    all_subreddits,
    subreddits_for,
)
from .models import (
    AnalyzedPost,
    RedditPost,
    SubredditSummary,
    TickerTrend,
    TrendReport,
    TrendTimeline,
)
from .recognizer import TickerExtractor, get_default_extractor
from .sentiment import SentimentAnalyzer
from .trends import DEFAULT_HEAT_WEIGHTS, compute_trends

__version__ = "0.1.0"

__all__ = [
    "AnalyzedPost",
    "DEFAULT_HEAT_WEIGHTS",
    "DEFAULT_SORTS",
    "DEFAULT_SUBREDDITS",
    "PostAnalyzer",
    "RedditClient",
    "RedditPost",
    "RedditTrendService",
    "SUBREDDIT_CATEGORIES",
    "SentimentAnalyzer",
    "SubredditSummary",
    "TickerExtractor",
    "TickerTrend",
    "TrendReport",
    "TrendTimeline",
    "__version__",
    "all_subreddits",
    "compute_trends",
    "get_default_extractor",
    "get_subreddit_overview",
    "get_trend_report",
    "get_trending_tickers",
    "subreddits_for",
]
