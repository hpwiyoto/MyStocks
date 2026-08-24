"""Fase 2 orchestrator: reads price_history from MySQL, computes technical +
fundamental features, writes to feature_daily / feature_fundamental_snapshot.

Usage:
    python -m features.build_features
"""
import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from features.db import FEATURE_VERSION, feature_daily, feature_fundamental_snapshot, init_schema
from features.fundamental import IHSG_SYMBOL, compute_relative_strength, fetch_fundamental_snapshot
from features.pattern_similarity import compute_pattern_similarity
from features.regime import classify_regime
from features.structure import compute_structure
from features.technical import MIN_ROWS_FOR_TECHNICAL_FEATURES
from features.technical import compute_all as compute_technical
from pipeline.db import get_engine, price_history
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS, to_yfinance_symbol
from pipeline.yfinance_source import fetch_history

logger = get_logger("features.build_features")

BOOL_COLS = ("higher_high_20d", "higher_low_20d", "lower_high_20d", "lower_low_20d")
INT_COLS = ("obv", "similar_pattern_count")


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _safe_bool(value):
    if value is None or pd.isna(value):
        return None
    return bool(value)


def load_price_history(conn, code: str) -> pd.DataFrame:
    rows = conn.execute(
        select(
            price_history.c.date,
            price_history.c.open,
            price_history.c.high,
            price_history.c.low,
            price_history.c.close,
            price_history.c.volume,
        )
        .where(price_history.c.stock_code == code)
        .order_by(price_history.c.date.asc())
    ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df.set_index("date")
    df = df.set_index("date")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    technical = compute_technical(df)
    structure = compute_structure(df)
    merged = pd.concat([technical, structure], axis=1)
    merged["regime"] = classify_regime(merged, df["close"])
    pattern = compute_pattern_similarity(df)
    return pd.concat([merged, pattern], axis=1)


def upsert_feature_daily(conn, code: str, features: pd.DataFrame) -> int:
    if features.empty:
        return 0

    rows = []
    for date, row in features.iterrows():
        record = {"stock_code": code, "date": date, "feature_version": FEATURE_VERSION}
        for col in features.columns:
            value = row[col]
            if col in BOOL_COLS:
                record[col] = _safe_bool(value)
            elif col in INT_COLS:
                record[col] = _safe_int(value)
            elif col == "regime":
                record[col] = value if pd.notna(value) else None
            else:
                record[col] = _safe_float(value)
        rows.append(record)

    stmt = mysql_insert(feature_daily).values(rows)
    update_cols = {c: stmt.inserted[c] for c in features.columns}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)
    return len(rows)


def upsert_fundamental_snapshot(conn, code: str, snapshot: dict) -> None:
    row = {"stock_code": code, "snapshot_date": dt.date.today()}
    for key, value in snapshot.items():
        row[key] = _safe_int(value) if key == "market_cap" else _safe_float(value)

    stmt = mysql_insert(feature_fundamental_snapshot).values([row])
    update_cols = {c: stmt.inserted[c] for c in row if c not in ("stock_code", "snapshot_date")}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)


def _load_ihsg_close() -> pd.Series:
    try:
        ihsg_df = fetch_history(IHSG_SYMBOL, period="1y")
    except Exception as exc:
        logger.warning("Failed to fetch IHSG (%s): %s — relative_strength will be null", IHSG_SYMBOL, exc)
        return pd.Series(dtype=float)

    if ihsg_df is None or ihsg_df.empty:
        return pd.Series(dtype=float)

    close = ihsg_df["Close"]
    close.index = close.index.date
    return close


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine = get_engine()
    init_schema(engine)

    logger.info("Fetching IHSG (%s) history for relative strength", IHSG_SYMBOL)
    ihsg_close = _load_ihsg_close()

    total_daily = 0
    total_fundamental = 0
    failures = []

    for code in tickers:
        try:
            with engine.begin() as conn:
                df = load_price_history(conn, code)
                if df.empty:
                    logger.warning("%s: no price_history yet — run pipeline.ingest_price first, skipping", code)
                    continue
                if len(df) < MIN_ROWS_FOR_TECHNICAL_FEATURES:
                    logger.warning(
                        "%s: only %d days of price_history (<%d minimum), skipping technical features for now",
                        code, len(df), MIN_ROWS_FOR_TECHNICAL_FEATURES,
                    )
                    continue

                features = build_technical_features(df)
                n = upsert_feature_daily(conn, code, features)
                total_daily += n
                logger.info("%s: upserted %d feature_daily rows", code, n)

                snapshot = fetch_fundamental_snapshot(to_yfinance_symbol(code))
                snapshot["relative_strength_20d_pct"] = compute_relative_strength(df["close"], ihsg_close)
                upsert_fundamental_snapshot(conn, code, snapshot)
                total_fundamental += 1
        except Exception as exc:
            logger.error("%s: feature build failed, skipping — %s", code, exc)
            failures.append(code)

    logger.info(
        "Done. feature_daily rows: %d, fundamental snapshots: %d. Failures: %s",
        total_daily, total_fundamental, failures or "none",
    )
    return {"feature_daily_rows": total_daily, "fundamental_snapshots": total_fundamental, "failures": failures}


if __name__ == "__main__":
    run()
