"""One-off backfill for feature_daily.net_foreign_flow from the RapidAPI IDX
source (see pipeline/idx_rapidapi_source.py) -- yfinance has no foreign-flow
data at all, this is the only source for it in this project.

Deliberately UPDATE-only, not upsert: a date the RapidAPI source has but this
ticker's feature_daily doesn't (e.g. outside the pipeline's own price/warmup
window) is silently skipped, never inserted as a bare partial row -- keeps
feature_daily's existing "every row has a full feature set, or a deliberate
NULL for a genuinely-missing value" shape intact rather than introducing
sparse rows that only have net_foreign_flow and nothing else.

Usage:
    python -m scripts.backfill_foreign_flow [--tickers CODE,CODE,...] [--timeframe 1y]
"""
import argparse
import time

import pandas as pd
from sqlalchemy import bindparam, update

from features.db import feature_daily
from pipeline.db import get_engine
from pipeline.idx_rapidapi_source import fetch_foreign_flow_all
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS

logger = get_logger("scripts.backfill_foreign_flow")


def _update_ticker(conn, code: str, df: pd.DataFrame) -> int:
    """UPDATE-only: sets net_foreign_flow on whichever (code, date) rows in
    `df` already exist in feature_daily. Rows in `df` with no matching
    feature_daily row are silently skipped (see module docstring)."""
    if df.empty:
        return 0
    stmt = (
        update(feature_daily)
        .where(feature_daily.c.stock_code == bindparam("code"), feature_daily.c.date == bindparam("d"))
        .values(net_foreign_flow=bindparam("val"))
    )
    rows = [{"code": code, "d": r["date"], "val": float(r["value"])} for _, r in df.iterrows()]
    result = conn.execute(stmt, rows)
    return result.rowcount if result.rowcount and result.rowcount > 0 else 0


def run(tickers=None, timeframe: str = "1y"):
    tickers = tickers or SEED_TICKERS
    engine = get_engine()

    logger.info("Fetching net foreign buy/sell for %d tickers (timeframe=%s)...", len(tickers), timeframe)
    t0 = time.time()
    flows = fetch_foreign_flow_all(tickers, timeframe=timeframe)
    logger.info("Fetched %d/%d tickers in %.0fs", len(flows), len(tickers), time.time() - t0)

    total_updated = 0
    zero_match = []
    for code, df in flows.items():
        with engine.begin() as conn:
            n = _update_ticker(conn, code, df)
        total_updated += n
        if n == 0:
            zero_match.append(code)

    if zero_match:
        logger.warning(
            "%d ticker(s) fetched but matched zero existing feature_daily rows (dates outside the pipeline's "
            "existing feature range, or build_features hasn't run for these dates yet): %s",
            len(zero_match), zero_match[:20] + (["..."] if len(zero_match) > 20 else []),
        )

    missing = sorted(set(tickers) - set(flows))
    logger.info(
        "Done. %d feature_daily rows updated across %d tickers. %d ticker(s) had no fetch result at all.",
        total_updated, len(flows), len(missing),
    )
    return {"rows_updated": total_updated, "tickers_fetched": len(flows), "tickers_missing": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=None, help="comma-separated ticker codes (default: full universe)")
    parser.add_argument("--timeframe", default="1y", help="1y/3y/5y etc. (default: 1y)")
    args = parser.parse_args()
    run(tickers=args.tickers.split(",") if args.tickers else None, timeframe=args.timeframe)
