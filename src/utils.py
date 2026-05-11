import re

from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# 1. The Regex for $TICKERS (The "Golden Rule")
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")

# 2. Expanded Mapper
COMPANY_TO_TICKER = {
    "APPLE": "AAPL",
    "TSMC": "TSM",
    "TAIWAN SEMI": "TSM",
    "PLAB": "PLAB",
    "PHOTRONICS": "PLAB",
    "WENDY": "WEN",
    "SERVICE NOW": "NOW",
    "SERVICENOW": "NOW",
}


def validate_ticker(symbol: str) -> bool:
    symbol = symbol.replace("$", "").strip().upper()
    # Basic sanity checks
    if not (1 <= len(symbol) <= 5):
        return False
    if not symbol.isalpha():
        return False
    return True


def extract_tickers_gliner(text: str) -> list[str]:
    final_tickers = set()

    # --- STEP 1: REGEX (Catch $PLAB) ---
    cashtags = CASHTAG_RE.findall(text.upper())
    for tag in cashtags:
        if validate_ticker(tag):
            final_tickers.add(tag)

    # --- STEP 2: GLiNER (Catch TSMC) ---
    labels = ["company", "ticker"]
    try:
        result = extractor.extract_entities(text, labels)
    except:
        result = {}

    if isinstance(result, dict):
        entities_dict = result.get("entities", result)

        # Process AI Tickers
        for t in entities_dict.get("ticker", []):
            t_upper = t.replace("$", "").upper().strip()
            # HALLUCINATION CHECK: Is the predicted ticker actually in the text?
            if validate_ticker(t_upper) and t_upper in text.upper():
                final_tickers.add(t_upper)

        # Process AI Companies
        for company in entities_dict.get("company", []):
            clean_company = company.upper().strip()

            # Map company names to tickers
            for comp_key, tick in COMPANY_TO_TICKER.items():
                if comp_key in clean_company or clean_company in comp_key:
                    final_tickers.add(tick)
                    break

    return list(final_tickers)
