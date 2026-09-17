# Reddit Stock Analyzer 📊

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---
<p align="center">
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/StephanAkkerman/reddit-stock-analyzer/pyversions.yml?label=python%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13&logo=python&style=flat-square">
  <img src="https://img.shields.io/github/license/StephanAkkerman/reddit-stock-analyzer.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

## Introduction

Scrapes recent posts from finance subreddits, works out **which stocks each
post is about** using
[`stock-recognizer`](https://github.com/StephanAkkerman/stock-recognizer),
scores their sentiment with
[FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment),
and ranks what is **trending** — not just what is loud, but what is getting
louder.

It is built as a library first: every entry point returns plain dataclasses
with `to_dict()`, so [`fintwit-web`](https://github.com/StephanAkkerman/fintwit-web)
can serve it straight out of a FastAPI route. See
[`docs/fintwit-web-integration.md`](docs/fintwit-web-integration.md).

## Table of Contents 🗂

- [Key Features](#key-features-)
- [Installation](#installation-)
- [Usage](#usage-)
- [Sentiment](#sentiment-)
- [Trend Metrics](#trend-metrics-)
- [Configuration](#configuration-)
- [Citation](#citation-)
- [Contributing](#contributing-)
- [License](#license-)

## Key Features 🔑

- **Curated subreddit catalogue** — `retail`, `stocks`, `research`, `trading`,
  `macro` and `crypto` groups, so you can ask for "everything retail traders
  shout about" instead of hard-coding names.
- **Credentials optional** — uses authenticated `asyncpraw` when Reddit API
  keys are configured, and the public JSON endpoints when they are not.
- **Real ticker recognition** — `stock-recognizer` handles cashtags, market-data
  validation, "DD" the company vs. "DD" the due diligence, and company-name
  mapping (`TSMC` → `TSM`).
- **WSB-native sentiment** — [FinTwitBERT-wsb](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment)
  reads "puts printing" the way a trader does, per post *and per ticker*.
- **Trend analytics** — two-window momentum, spike z-scores, emerging/fading
  detection, per-subreddit breakdowns and an hourly mention timeline.
- **Async and shareable** — concurrent scraping, models loaded once, CPU-bound
  work pushed off the event loop.

## Installation ⚙️

```bash
pip install git+https://github.com/StephanAkkerman/reddit-stock-analyzer.git
```

Optional extras:

```bash
pip install "reddit-stock-analyzer[praw]"       # authenticated scraping
pip install "reddit-stock-analyzer[sentiment]"  # FinTwitBERT post sentiment
```

`[sentiment]` is usually redundant: `transformers` and `torch` already arrive
with `stock-recognizer`, so sentiment works out of the box. If they are
missing, every post is labelled `neutral` and everything else still works.
Credentials are read from the environment or a `.env` file — see
[`.env.example`](.env.example).

## Usage ⌨️

### Library

```python
import asyncio
from reddit_stock_analyzer import RedditTrendService, subreddits_for

async def main():
    async with RedditTrendService() as service:
        report = await service.trend_report(
            subreddits_for("retail", "trading"),
            window_hours=24,   # the period under analysis
            baseline_hours=24, # the period it is compared against
        )

    for trend in report.tickers[:5]:
        print(
            f"{trend.symbol:<6} {trend.mentions:>3} mentions "
            f"(was {trend.previous_mentions}) "
            f"momentum {trend.momentum:+.2f} "
            f"{trend.sentiment} ({trend.sentiment_score:+.2f})"
        )

    print("emerging:", report.emerging)
    print("fading:", report.fading)

asyncio.run(main())
```

A single subreddit snapshot, without the trend maths:

```python
overview = await service.subreddit_overview("wallstreetbets", limit=50)
# {'subreddit': ..., 'overall_mood': 'bullish', 'top_tickers': {'NVDA': 7, ...},
#  'sentiment_breakdown': {...}, 'posts': [...]}
```

Scraping and analysis are separable — cache a scrape and re-rank it with
different parameters for free:

```python
from reddit_stock_analyzer import compute_trends

posts = await service.scrape(["wallstreetbets"], max_age_hours=48)
day = compute_trends(posts, window_hours=24)
morning = compute_trends(posts, window_hours=4, baseline_hours=44)
```

### Command line

```bash
python -m reddit_stock_analyzer --category retail --window 12 --top 10
python -m reddit_stock_analyzer -s wallstreetbets -s options --json
python -m reddit_stock_analyzer --no-ai        # skip the GLiNER2 model
```

```text
r/wallstreetbets + r/options
412 posts in the last 12h (of 780 scraped) — mood: bullish (+0.21)

TICKER   MENTIONS  PREV     MOM  SPIKE   HEAT  SENTIMENT
NVDA           41    18   +1.21  +2.84  0.912  bullish (+0.44)
SPY            33    35   -0.06  +2.01  0.604  neutral (+0.03)
...

rising: NVDA, SMCI
emerging: RKLB
fading: AMC
```

## Sentiment 🐂🐻

Posts are classified by
[`StephanAkkerman/FinTwitBERT-wsb-sentiment`](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment)
— BERT pre-trained on financial tweets and fine-tuned on WallStreetBets-style
text, so it reads "puts printing" and "she's gonna rip" as a trader would.
Labels are normalised to `bullish` / `bearish` / `neutral`, and every score is
**signed by direction**: `+0.9` is confidently bullish, `-0.9` confidently
bearish, `0.0` neutral. That makes them averageable, which is what every
aggregate in a report is built on.

Sentiment surfaces at four levels:

| Level | Field |
| --- | --- |
| Post | `AnalyzedPost.sentiment`, `.sentiment_score`, `.sentiment_confidence` |
| Ticker within a post | `AnalyzedPost.sentiment_for("INTC")` |
| Ticker across the window | `TickerTrend.sentiment`, `.sentiment_score`, `.sentiment_breakdown` |
| Subreddit / overall | `SubredditSummary.mood`, `TrendReport.mood` |

### Per-ticker attribution

A post reading *"Long NVDA, short INTC"* is bullish and bearish at once. One
post-level label would give one of the two the wrong sign, so posts that
mention **more than one** ticker are split into segments, and each ticker is
scored from the segments that name **only it** — a sentence mentioning both
says nothing about either in particular, and is used only when a ticker has
no sentence of its own:

```python
analyzed.sentiment            # 'bullish'  — the post as a whole
analyzed.sentiment_for("NVDA")  # +0.9
analyzed.sentiment_for("INTC")  # -0.8
```

Single-ticker posts — the majority — cost nothing extra: the post label is
already about that ticker. A ticker the recognizer resolved from a company
name ("TSMC" → `TSM`) has no literal segment to match, and falls back to the
post's score. Switch the whole thing off with
`PostAnalyzer(per_ticker_sentiment=False)`.

`TickerTrend.sentiment_score` is the mean of these attributed scores, so a
ticker that everyone is shorting reads bearish even in a bullish subreddit.

### Swapping the model

```python
from reddit_stock_analyzer import PostAnalyzer, SentimentAnalyzer

analyzer = PostAnalyzer(sentiment=SentimentAnalyzer("your-org/your-model"))
```

Labels named `BULLISH`/`BEARISH`/`NEUTRAL` (or `POSITIVE`/`NEGATIVE`) are
understood as-is. A checkpoint published without an `id2label` mapping reports
`LABEL_0`, `LABEL_1`... — that order is **not** guessed, since guessing wrong
silently inverts every score. Those read neutral and log a warning until you
say which is which:

```python
SentimentAnalyzer(
    "your-org/your-model",
    label_map={"LABEL_0": "bearish", "LABEL_1": "neutral", "LABEL_2": "bullish"},
)
```

## Trend Metrics 📈

Posts are split into a **window** (the recent period, default 24h) and an
equally long **baseline** immediately before it. Raw counts say how loud a
ticker is; the comparison says whether it is getting louder.

| Metric | Meaning |
| --- | --- |
| `mentions` | Posts in the window mentioning the ticker — counted **once per post**, so one ranting post cannot manufacture a trend |
| `previous_mentions` | The same count over the baseline window |
| `momentum` | `(mentions - previous) / (previous + 1)` — smoothed so 0 → 5 is finite (4.0) and ranks above 50 → 60 (0.2) |
| `change_ratio` | `mentions / previous`, or `null` when there is no baseline |
| `spike_score` | Standard deviations above this scrape's mean mention count — "unusual *for today*" |
| `heat_score` | 0–1 blend of mentions (40%), engagement (25%), unique authors (15%) and momentum (20%); the default ranking |
| `is_emerging` | No baseline mentions at all, and at least `emerging_min_mentions` now |

Alongside the ranked tickers a report carries `rising` / `emerging` / `fading`
shortlists, a `by_subreddit` breakdown (is this hype confined to r/wallstreetbets,
or has it reached r/investing?) and an hourly `timeline` per top ticker.

Every weight and threshold is a keyword argument:

```python
report = compute_trends(
    posts,
    window_hours=6,
    min_mentions=3,                 # ignore one-off noise
    emerging_min_mentions=5,
    weights={"momentum": 0.5, "mentions": 0.2},  # rank by acceleration
)
```

## Configuration 🔧

| Subreddit category | Contents |
| --- | --- |
| `retail` | wallstreetbets, smallstreetbets, WallStreetbetsELITE, Shortsqueeze, pennystocks |
| `stocks` | stocks, StockMarket, investing, dividends |
| `research` | SecurityAnalysis, ValueInvesting, stocks |
| `trading` | Daytrading, swingtrading, options, thetagang, algotrading |
| `macro` | economics, finance, economy |
| `crypto` | CryptoCurrency, CryptoMarkets, satoshistreetbets, ethtrader, Bitcoin |

```python
from reddit_stock_analyzer import DEFAULT_SUBREDDITS, all_subreddits, subreddits_for

subreddits_for("retail", "trading")  # merged, de-duplicated, declaration order
```

## Citation ✍️

If you use this project in your research, please cite as follows:

```bibtex
@misc{reddit_stock_analyzer,
  author  = {Stephan Akkerman},
  title   = {Reddit Stock Analyzer},
  year    = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/reddit-stock-analyzer}}
}
```

## Contributing 🛠

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/reddit-stock-analyzer/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/reddit-stock-analyzer)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
