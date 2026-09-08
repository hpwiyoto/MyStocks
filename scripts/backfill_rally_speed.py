"""One-time backfill for feature_daily.ret_10d_pct / ret_10d_atr_norm on
every EXISTING row -- these two columns were added (features/db.py,
features/technical.py's compute_rally_speed) after scripts/train_v5.py's
main pipeline had already been running for years, so historical rows
have them NULL until this runs once. Unlike scripts/backfill_foreign_flow.py,
this needs no external API or quota: both columns are pure derivatives of
price_history.close (already in the DB) and feature_daily.atr_pct_14
(already computed for every row), so it's a fast, purely local
computation.

Going forward, features/build_features.py's normal daily run computes
these for every new row automatically (compute_all -> compute_rally_speed)
-- this script only needs to run once, to fill in the past.

Usage:
    python -m scripts.backfill_rally_speed
"""
import pandas as pd
from sqlalchemy import text

from pipeline.db import get_engine
from pipeline.logging_config import get_logger

logger = get_logger("scripts.backfill_rally_speed")

RALLY_WINDOW = 10  # matches the Swing model's own 10-day HORIZON (scripts/train_v5.py)


def run():
    engine = get_engine()
    with engine.connect() as conn:
        codes = [r[0] for r in conn.execute(text("SELECT DISTINCT stock_code FROM feature_daily")).fetchall()]
    logger.info("Backfilling ret_10d_pct/ret_10d_atr_norm for %d tickers", len(codes))

    total_updated = 0
    for i, code in enumerate(codes, 1):
        with engine.begin() as conn:
            prices = pd.read_sql(
                text("SELECT date, close FROM price_history WHERE stock_code = :code ORDER BY date"),
                conn, params={"code": code},
            )
            feats = pd.read_sql(
                text("SELECT date, atr_pct_14 FROM feature_daily WHERE stock_code = :code ORDER BY date"),
                conn, params={"code": code},
            )
            if prices.empty or feats.empty:
                continue

            prices["date"] = pd.to_datetime(prices["date"])
            feats["date"] = pd.to_datetime(feats["date"])
            prices["ret_10d_pct"] = prices["close"].pct_change(RALLY_WINDOW) * 100
            merged = feats.merge(prices[["date", "ret_10d_pct"]], on="date", how="left")
            merged["ret_10d_atr_norm"] = merged["ret_10d_pct"] / merged["atr_pct_14"].replace(0, float("nan"))

            rows = [
                {"code": code, "d": row["date"].date(), "r1": None if pd.isna(row["ret_10d_pct"]) else float(row["ret_10d_pct"]),
                 "r2": None if pd.isna(row["ret_10d_atr_norm"]) else float(row["ret_10d_atr_norm"])}
                for _, row in merged.iterrows()
            ]
            conn.execute(
                text(
                    "UPDATE feature_daily SET ret_10d_pct = :r1, ret_10d_atr_norm = :r2 "
                    "WHERE stock_code = :code AND date = :d"
                ),
                rows,
            )
            total_updated += len(rows)

        if i % 100 == 0 or i == len(codes):
            logger.info("[%d/%d] %s: %d rows updated so far", i, len(codes), code, total_updated)

    logger.info("Done. %d feature_daily rows updated across %d tickers", total_updated, len(codes))


if __name__ == "__main__":
    run()
