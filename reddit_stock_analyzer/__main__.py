"""Command line entry point: ``python -m reddit_stock_analyzer``.

A smoke test and a genuinely useful terminal view::

    python -m reddit_stock_analyzer --window 12 --category retail
    python -m reddit_stock_analyzer -s wallstreetbets -s options --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone

from .aggregator import DEFAULT_SORTS, RedditTrendService
from .client import RedditClient
from .config import DEFAULT_SUBREDDITS, SUBREDDIT_CATEGORIES, subreddits_for
from .models import TrendReport


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reddit-stock-analyzer",
        description="Scrape finance subreddits and rank the tickers being discussed.",
    )
    parser.add_argument(
        "-s",
        "--subreddit",
        action="append",
        dest="subreddits",
        help="Subreddit to scrape; repeat for several. Defaults to %s."
        % ", ".join(DEFAULT_SUBREDDITS),
    )
    parser.add_argument(
        "-c",
        "--category",
        action="append",
        dest="categories",
        choices=sorted(SUBREDDIT_CATEGORIES),
        help="Add every subreddit of a catalogue category; repeat for several.",
    )
    parser.add_argument(
        "--window", type=float, default=24.0, help="Trend window in hours (default 24)."
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Comparison window in hours (default: same as --window).",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Posts per subreddit per sort."
    )
    parser.add_argument(
        "--sort",
        action="append",
        dest="sorts",
        choices=["hot", "new", "top", "rising", "controversial"],
        help="Listing to read; repeat for several. Defaults to %s."
        % ", ".join(DEFAULT_SORTS),
    )
    parser.add_argument("--top", type=int, default=15, help="Tickers to display.")
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip the GLiNER2 ticker model (regex + market data only).",
    )
    parser.add_argument(
        "--no-praw",
        action="store_true",
        help="Force the anonymous JSON API even when credentials are set.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full report as JSON."
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging.")
    return parser


def _resolve_subreddits(args: argparse.Namespace) -> list[str]:
    names: dict[str, None] = {}
    for name in args.subreddits or ():
        names.setdefault(name, None)
    for category in args.categories or ():
        for name in subreddits_for(category):
            names.setdefault(name, None)
    return list(names) or list(DEFAULT_SUBREDDITS)


def _format_report(report: TrendReport, top: int) -> str:
    when = datetime.fromtimestamp(report.generated_at, tz=timezone.utc)
    lines = [
        f"r/{' + r/'.join(report.subreddits)}",
        f"{report.posts_in_window} posts in the last {report.window_hours:g}h "
        f"(of {report.posts_analyzed} scraped) — mood: {report.mood} "
        f"({report.sentiment_score:+.2f}) — {when:%Y-%m-%d %H:%M UTC}",
        "",
        f"{'TICKER':<8}{'MENTIONS':>9}{'PREV':>6}{'MOM':>8}{'SPIKE':>7}"
        f"{'HEAT':>7}  SENTIMENT",
    ]
    for trend in report.tickers[:top]:
        lines.append(
            f"{trend.symbol:<8}{trend.mentions:>9}{trend.previous_mentions:>6}"
            f"{trend.momentum:>+8.2f}{trend.spike_score:>+7.2f}{trend.heat_score:>7.3f}"
            f"  {trend.sentiment} ({trend.sentiment_score:+.2f})"
        )
    if not report.tickers:
        lines.append("(no tickers recognised in the window)")

    movers = [
        f"{label}: {', '.join(symbols)}"
        for label, symbols in (
            ("rising", report.rising),
            ("emerging", report.emerging),
            ("fading", report.fading),
        )
        if symbols
    ]
    if movers:
        lines.extend(["", *movers])
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    subreddits = _resolve_subreddits(args)
    service = RedditTrendService(
        client=RedditClient(use_praw=not args.no_praw),
        use_ai=not args.no_ai,
    )
    try:
        report = await service.trend_report(
            subreddits,
            window_hours=args.window,
            baseline_hours=args.baseline,
            limit=args.limit,
            sorts=tuple(args.sorts) if args.sorts else DEFAULT_SORTS,
            top_n=max(args.top, 25),
        )
    finally:
        await service.close()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_report(report, args.top))
    return 0 if report.posts_analyzed else 1


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    if not args.debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - optional dependency
    except ImportError:
        pass
    else:
        load_dotenv()

    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
