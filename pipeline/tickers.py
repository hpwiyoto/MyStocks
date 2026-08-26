import json
import os

_IDX_TICKERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idx_tickers.json")

# Original 10 blue-chip tickers Fase 0-6 were built and tested against.
# Kept as a fast, known-good subset for quick dev/debug runs.
BLUE_CHIP_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM",
    "ASII", "UNVR", "ICBP", "ANTM", "ADRO",
]


def _load_all_idx_tickers() -> list[str]:
    with open(_IDX_TICKERS_PATH) as f:
        entries = json.load(f)
    return [e["code"] for e in entries]


# Full IDX universe scraped from stockanalysis.com/list/indonesia-stock-exchange/
# (2026-08-25, 914 tickers, sorted by market cap) -- everything yfinance is
# expected to have data for. This is what ingest_price/build_features/predict
# operate on by default; swap to BLUE_CHIP_TICKERS for a fast 10-ticker subset.
ALL_IDX_TICKERS = _load_all_idx_tickers()

SEED_TICKERS = ALL_IDX_TICKERS


def to_yfinance_symbol(code: str) -> str:
    return f"{code}.JK"
