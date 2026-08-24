import time

import pandas as pd
import yfinance as yf

from pipeline.logging_config import get_logger

logger = get_logger("pipeline.yfinance")

MAX_RETRIES = 3
BACKOFF_SECONDS = 5


def _with_retry(fn, description: str):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # yfinance raises plain Exception/requests errors
            last_error = exc
            wait = BACKOFF_SECONDS * attempt
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %ds",
                description, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    logger.error("%s failed after %d attempts: %s", description, MAX_RETRIES, last_error)
    raise last_error


def fetch_history(symbol: str, start: str | None = None, period: str | None = None) -> pd.DataFrame:
    def _fetch():
        ticker = yf.Ticker(symbol)
        if start:
            df = ticker.history(start=start, auto_adjust=False)
        else:
            df = ticker.history(period=period or "max", auto_adjust=False)
        if df is None:
            raise ValueError(f"yfinance returned no data for {symbol}")
        return df

    return _with_retry(_fetch, f"fetch_history({symbol})")


def fetch_profile(symbol: str) -> dict:
    def _fetch():
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
        }

    try:
        return _with_retry(_fetch, f"fetch_profile({symbol})")
    except Exception:
        # Profile metadata is a nice-to-have; missing it should not block price ingestion.
        return {"name": None, "sector": None}
