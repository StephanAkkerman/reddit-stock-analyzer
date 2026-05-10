import re

# Matches $AAPL or mentions of 2-5 uppercase letters
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")


def extract_tickers(text: str, whitelist: set = None) -> list[str]:
    """
    Extracts tickers from text. If a whitelist is provided,
    it filters out common words that look like tickers.
    """
    matches = TICKER_PATTERN.findall(text)
    # Flatten the regex groups
    extracted = [m[0] or m[1] for m in matches]

    if whitelist:
        return list(set([t for t in extracted if t in whitelist]))

    # Basic filter to avoid common false positives if no whitelist provided
    blacklist = {"FOR", "THE", "AND", "IT", "IS", "ARE", "HAS"}
    return list(set([t.upper() for t in extracted if t.upper() not in blacklist]))
