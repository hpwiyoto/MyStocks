"""One-off backfill for the v5 feature additions (obv_zscore_20,
price_vs_vwap20_pct, sector_relative_strength_20d_pct, trailing_pe,
price_to_book, market_cap_log) across EVERY existing feature_daily row.

Deliberately separate from features.build_features.run(): that function's
`last_feature_date`/`only_dates_after` incremental logic exists specifically
to avoid re-walking the slow cross-ticker pattern-similarity pass on dates
already scored -- exactly what we do NOT want to skip here, since it's the
new columns on those ALREADY-scored historical rows that need filling in.
Pattern-similarity/regime/other existing columns are never touched (this
only ever UPDATEs the 6 new columns, never re-derives or overwrites the
rest) -- so this is safe to run without re-doing the (many-hours) full
pattern-similarity backfill.

Usage:
    python -m scripts.backfill_v5_features [--tickers CODE,CODE,...]
"""
import argparse
import math
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from features.build_features import _build_sector_composites, _load_ihsg_close, _safe_float, load_price_history
from features.db import FEATURE_VERSION, feature_daily
from features.technical import MIN_ROWS_FOR_TECHNICAL_FEATURES, compute_vwap
from features.technical import compute_all as compute_technical
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS

logger = get_logger("scripts.backfill_v5_features")

NEW_COLUMNS = [
    "obv_zscore_20", "price_vs_vwap20_pct", "sector_relative_strength_20d_pct",
    "trailing_pe", "price_to_book", "market_cap_log",
]


def _update_new_columns(conn, code: str, df: pd.DataFrame) -> int:
    """UPDATE-only upsert: touches just NEW_COLUMNS on rows that already
    exist (matched by the stock_code+date+feature_version unique key) --
    every other column (technical indicators computed earlier, pattern-
    similarity, regime) is left exactly as-is."""
    if df.empty:
        return 0
    rows = []
    for date, row in df.iterrows():
        record = {"stock_code": code, "date": date, "feature_version": FEATURE_VERSION}
        for col in NEW_COLUMNS:
            record[col] = _safe_float(row[col]) if col in row.index else None
        rows.append(record)

    stmt = mysql_insert(feature_daily).values(rows)
    update_cols = {c: stmt.inserted[c] for c in NEW_COLUMNS}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    result = conn.execute(stmt)
    return result.rowcount


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine = get_engine()

    logger.info("Loading price history for %d tickers...", len(tickers))
    price_by_code = {}
    for code in tickers:
        with engine.connect() as conn:
            df = load_price_history(conn, code)
        if len(df) >= MIN_ROWS_FOR_TECHNICAL_FEATURES:
            price_by_code[code] = df
    logger.info("%d/%d tickers have enough history", len(price_by_code), len(tickers))

    with engine.connect() as conn:
        sector_by_code = dict(conn.execute(text("SELECT code, sector FROM stocks")).fetchall())
        # Reuse whatever fundamental snapshot each ticker already has (most
        # recent row) instead of re-fetching ~900 tickers from yfinance --
        # this is a static/latest-value feature anyway (see
        # build_features._merge_fundamental_features), so there's no
        # accuracy lost by not re-fetching, only ~5-10 minutes of API calls
        # saved.
        fundamental_rows = conn.execute(text("""
            SELECT f.stock_code, f.trailing_pe, f.price_to_book, f.market_cap
            FROM feature_fundamental_snapshot f
            INNER JOIN (
                SELECT stock_code, MAX(snapshot_date) AS max_d
                FROM feature_fundamental_snapshot GROUP BY stock_code
            ) latest ON f.stock_code = latest.stock_code AND f.snapshot_date = latest.max_d
        """)).mappings().fetchall()
        latest_fundamental = {r["stock_code"]: dict(r) for r in fundamental_rows}

    logger.info("Fetching IHSG full history...")
    ihsg_close = _load_ihsg_close()

    logger.info("Building sector composites from %d tickers...", len(price_by_code))
    sector_composites = _build_sector_composites(price_by_code, sector_by_code)
    logger.info("Built %d sector composites", len(sector_composites))

    total_updated = 0
    failures = []
    t_start = time.time()
    for i, (code, df) in enumerate(price_by_code.items(), 1):
        try:
            sector_close = sector_composites.get(sector_by_code.get(code))
            technical = compute_technical(df, ihsg_close, sector_close)
            vwap = compute_vwap(df)
            merged = pd.concat([technical[["obv_zscore_20", "sector_relative_strength_20d_pct"]], vwap], axis=1)

            fund = latest_fundamental.get(code, {})
            merged["trailing_pe"] = fund.get("trailing_pe")
            merged["price_to_book"] = fund.get("price_to_book")
            market_cap = fund.get("market_cap")
            merged["market_cap_log"] = math.log10(market_cap) if market_cap and market_cap > 0 else None

            with engine.begin() as conn:
                n = _update_new_columns(conn, code, merged)
            total_updated += n
            if i % 50 == 0 or i == len(price_by_code):
                elapsed = time.time() - t_start
                logger.info("[%d/%d] %s: updated %d rows (%.0fs elapsed, ~%.0fs remaining)",
                            i, len(price_by_code), code, n, elapsed,
                            elapsed / i * (len(price_by_code) - i))
        except Exception as exc:
            logger.error("%s: backfill failed — %s", code, exc)
            failures.append(code)

    logger.info("Done. Updated %d rows total. Failures: %s", total_updated, failures or "none")
    return {"rows_updated": total_updated, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=None, help="comma-separated ticker codes (default: full universe)")
    args = parser.parse_args()
    run(tickers=args.tickers.split(",") if args.tickers else None)
