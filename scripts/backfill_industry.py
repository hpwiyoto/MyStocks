"""One-off backfill for the new `industry` (sub-sector) column on already-
registered stocks -- pipeline.ingest_price.upsert_stock only sets it for
BRAND NEW tickers (skips existing rows entirely), so the ~908 tickers
already in `stocks` need this separate pass.

Usage:
    python -m scripts.backfill_industry
"""
import time

from sqlalchemy import select, update

from pipeline.db import get_engine, stocks
from pipeline.logging_config import get_logger
from pipeline.tickers import to_yfinance_symbol
from pipeline.yfinance_source import fetch_profile

logger = get_logger("scripts.backfill_industry")


def run():
    engine = get_engine()
    with engine.connect() as conn:
        codes = [r[0] for r in conn.execute(select(stocks.c.code)).fetchall()]

    updated = 0
    failures = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        try:
            profile = fetch_profile(to_yfinance_symbol(code))
            with engine.begin() as conn:
                conn.execute(update(stocks).where(stocks.c.code == code).values(industry=profile["industry"]))
            updated += 1
        except Exception as exc:
            logger.error("%s: failed -- %s", code, exc)
            failures.append(code)
        if i % 100 == 0 or i == len(codes):
            elapsed = time.time() - t0
            logger.info("[%d/%d] %d updated, %.0fs elapsed, ~%.0fs remaining",
                        i, len(codes), updated, elapsed, elapsed / i * (len(codes) - i))

    logger.info("Done. Updated %d/%d. Failures: %s", updated, len(codes), failures or "none")


if __name__ == "__main__":
    run()
