"""One-off retroactive cleanup for phantom rows ingested on days IDX
itself was closed, ingested BEFORE pipeline/ingest_price.py's holiday
guard (load_trading_days) existed to prevent them going forward.

Root cause (confirmed empirically, 2026-09-16, not hypothetical):
yfinance sometimes returns a STALE quote for an individual ticker on a
day the exchange is actually closed -- flat open=high=low=close (copied
verbatim from the last real close) with volume=0. Confirmed directly:
ENRG's 2026-08-17 (Independence Day) row was open=high=low=close=1255,
exactly 2026-08-14's real close, volume=0 -- while IHSG (^JKSE) itself
correctly has no row at all for that date. This affects a MINORITY of
tickers per holiday (43-87 of ~900 in the 5 confirmed 2026 instances:
Labor Day, Pancasila Day, a day near Idul Adha, Independence Day, a
likely cuti-bersama day) -- most tickers already correctly return
nothing for a closed day.

Uses IHSG's own trading-day calendar as the authoritative reference
(empirically confirmed to NOT have this glitch) rather than a maintained
Indonesian public-holiday list, which would need yearly upkeep for
variable religious holidays. Any date price_history has data for that
IHSG never traded on is by definition impossible -- the exchange was
closed, so no genuine quote could exist for ANY ticker that day -- so
every row on such a date is deleted, not just ones matching the flat/
zero-volume signature (a stale phantom row doesn't have to look exactly
like that pattern to still be impossible).

Cleans BOTH price_history AND feature_daily (feature_daily's rows on
these dates were computed FROM the same bad price_history rows, so carry
the same contamination) for the exact phantom dates. Deliberately does
NOT attempt to recompute feature_daily for the (up to ~200-day, for a
SMA200 window) stretch of dates AFTER each phantom date whose rolling
indicators were fed one bad input row -- out of scope: the distortion is
a single extra data point in a multi-hundred-row rolling window, and a
full historical recompute across 5 years x ~900 tickers is a much larger
undertaking than this cleanup, better done deliberately if ever needed
rather than as a side effect here.

Usage:
    python -m scripts.cleanup_holiday_phantom_rows          # dry run, reports only
    python -m scripts.cleanup_holiday_phantom_rows --delete # actually deletes
"""
import argparse

import pandas as pd
from sqlalchemy import text

from pipeline.db import coerce_date, get_engine
from pipeline.logging_config import get_logger
from pipeline.yfinance_source import fetch_history

logger = get_logger("scripts.cleanup_holiday_phantom_rows")


def find_phantom_dates(engine) -> list:
    logger.info("Fetching IHSG (^JKSE) full history as the authoritative trading-day calendar...")
    # Observed once during testing: yfinance occasionally returns an
    # empty/malformed frame (a plain Index instead of DatetimeIndex) for
    # this exact query with no exception raised -- fetch_history's own
    # retry loop only catches actual exceptions, not "returned something,
    # just not usable", so this needs its own retry on top of that one.
    ihsg = fetch_history("^JKSE", period="max")
    if ihsg.empty or not isinstance(ihsg.index, pd.DatetimeIndex):
        logger.warning("First IHSG fetch returned an unusable frame, retrying once...")
        ihsg = fetch_history("^JKSE", period="max")
    trading_days = set(ihsg.index.date)
    logger.info("%d IDX trading days in IHSG's own calendar (%s to %s)",
                len(trading_days), min(trading_days), max(trading_days))

    distinct_dates = pd.read_sql("SELECT DISTINCT date FROM price_history", engine)
    distinct_dates["date"] = distinct_dates["date"].apply(coerce_date)
    phantom_dates = sorted(d for d in distinct_dates["date"] if d not in trading_days)
    return phantom_dates


def run(apply: bool = False):
    engine = get_engine()
    phantom_dates = find_phantom_dates(engine)
    logger.info("Found %d date(s) in price_history that IHSG never traded on: %s",
                len(phantom_dates), phantom_dates)

    if not phantom_dates:
        logger.info("Nothing to clean up.")
        return

    total_price_rows, total_feature_rows = 0, 0
    with engine.connect() as conn:
        for d in phantom_dates:
            n_price = conn.execute(text("SELECT COUNT(*) FROM price_history WHERE date = :d"), {"d": d}).scalar()
            n_feat = conn.execute(text("SELECT COUNT(*) FROM feature_daily WHERE date = :d"), {"d": d}).scalar()
            total_price_rows += n_price
            total_feature_rows += n_feat
            logger.info("  %s: %d price_history row(s), %d feature_daily row(s)", d, n_price, n_feat)

    logger.info("TOTAL: %d price_history + %d feature_daily phantom rows across %d holiday dates",
                total_price_rows, total_feature_rows, len(phantom_dates))

    if not apply:
        logger.info("Dry run only -- re-run with --delete to actually remove these rows.")
        return

    with engine.begin() as conn:
        for d in phantom_dates:
            r1 = conn.execute(text("DELETE FROM price_history WHERE date = :d"), {"d": d})
            r2 = conn.execute(text("DELETE FROM feature_daily WHERE date = :d"), {"d": d})
            logger.info("  %s: deleted %d price_history + %d feature_daily row(s)", d, r1.rowcount, r2.rowcount)
    logger.info("Done. Deleted %d price_history + %d feature_daily phantom rows total.",
                total_price_rows, total_feature_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete phantom rows (default: dry run, report only)")
    args = parser.parse_args()
    run(apply=args.delete)
