"""Incremental OHLCV ingestion from yfinance into `stocks` + `price_history`.

Usage:
    python -m pipeline.ingest_price
"""
import datetime as dt

import pandas as pd
from sqlalchemy import insert, select

from pipeline.db import get_engine, init_schema, price_history, stocks, upsert
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS, to_yfinance_symbol
from pipeline.yfinance_source import fetch_history, fetch_profile

logger = get_logger("pipeline.ingest_price")

SOURCE = "yfinance"


def load_trading_days() -> set[dt.date] | None:
    """IDX's own trading-day calendar, derived from IHSG (^JKSE)'s history
    rather than a maintained Indonesian public-holiday list (which would
    need yearly upkeep for variable religious holidays/cuti bersama).

    Confirmed empirically (2026-09-16), not hypothetical: yfinance
    sometimes returns a STALE phantom row for an individual ticker on a
    day IDX itself is closed -- e.g. ENRG's 2026-08-17 (Independence Day)
    row was open=high=low=close=1255 (2026-08-14's real close, copied
    verbatim) with volume=0, while ^JKSE itself correctly had no row at
    all that date. Confirmed across 5+ such dates in 2026 alone (Labor
    Day, Pancasila Day, a day near Idul Adha, Independence Day, a likely
    cuti-bersama day), each affecting a MINORITY of tickers (43-87 of
    ~900) rather than all of them -- the majority already correctly
    return nothing for a closed day, this is specifically about the ones
    that don't. See scripts/cleanup_holiday_phantom_rows.py for the
    retroactive cleanup of rows already ingested before this guard
    existed.

    Returns None (rather than raising) if the IHSG fetch itself fails --
    holiday-row filtering is a safety net, not something that should
    block ingesting the other ~900 tickers over a single transient
    network hiccup on this one extra call.
    """
    try:
        df = fetch_history("^JKSE", period="max")
        return set(df.index.date)
    except Exception as exc:
        logger.warning("Failed to load IHSG trading-day calendar -- holiday-row filtering disabled this run: %s", exc)
        return None


def _safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _safe_float(value) -> float | None:
    # NaN can appear on data gaps (e.g. suspended trading days) -- stored as
    # None/NULL rather than a literal NaN so it round-trips cleanly through
    # pandas/JSON downstream (app/data.py's `_notna` helper etc. already
    # assume SQL NULL, not float NaN, as the "missing" representation).
    if value is None or pd.isna(value):
        return None
    return float(value)


def upsert_stock(conn, code: str) -> None:
    existing = conn.execute(select(stocks.c.code).where(stocks.c.code == code)).first()
    if existing:
        return
    profile = fetch_profile(to_yfinance_symbol(code))
    conn.execute(
        insert(stocks).values(
            code=code, name=profile["name"], sector=profile["sector"], industry=profile["industry"],
        )
    )
    logger.info("Registered new stock %s (%s)", code, profile["name"])


def last_ingested_date(conn, code: str):
    row = conn.execute(
        select(price_history.c.date)
        .where(price_history.c.stock_code == code, price_history.c.source_provider == SOURCE)
        .order_by(price_history.c.date.desc())
        .limit(1)
    ).first()
    return row[0] if row else None


def ingest_one(conn, code: str, trading_days: set[dt.date] | None) -> int:
    symbol = to_yfinance_symbol(code)
    last_date = last_ingested_date(conn, code)

    if last_date is None:
        logger.info("%s: no existing data, fetching full history", code)
        df = fetch_history(symbol, period="5y")
    else:
        next_date = last_date + dt.timedelta(days=1)
        if next_date > dt.date.today():
            logger.info("%s: already up to date (last=%s)", code, last_date)
            return 0
        start = next_date.isoformat()
        logger.info("%s: incremental fetch from %s", code, start)
        df = fetch_history(symbol, start=start)

    if df.empty:
        logger.info("%s: no new rows", code)
        return 0

    if trading_days is not None:
        # See load_trading_days's docstring -- drop any row yfinance
        # returned for a day IDX itself was closed (a stale phantom
        # quote), before it ever reaches price_history.
        before = len(df)
        df = df[df.index.map(lambda ts: ts.date() in trading_days)]
        dropped = before - len(df)
        if dropped:
            logger.info("%s: dropped %d row(s) on IDX non-trading day(s) (holiday phantom data)", code, dropped)
        if df.empty:
            logger.info("%s: no new rows after holiday filtering", code)
            return 0

    rows = []
    for index, row in df.iterrows():
        rows.append(
            {
                "stock_code": code,
                "date": index.date(),
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": _safe_int(row.get("Volume")),
                "dividends": _safe_float(row.get("Dividends", 0)),
                "stock_splits": _safe_float(row.get("Stock Splits", 0)),
                "source_provider": SOURCE,
            }
        )

    upsert(
        conn, price_history, rows,
        update_columns=["open", "high", "low", "close", "volume", "dividends", "stock_splits"],
        index_elements=["stock_code", "date", "source_provider"],
    )
    logger.info("%s: upserted %d rows", code, len(rows))
    return len(rows)


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine = get_engine()
    init_schema(engine)

    # Fetched ONCE per run (not per ticker) -- a single small IHSG dataset,
    # negligible cost added to a run that's about to make ~900 of these
    # calls anyway. See load_trading_days's docstring for why this exists.
    trading_days = load_trading_days()
    if trading_days is not None:
        logger.info("Loaded %d IDX trading days (IHSG calendar) for holiday-row filtering", len(trading_days))

    total = 0
    failures = []
    for code in tickers:
        try:
            with engine.begin() as conn:
                upsert_stock(conn, code)
                total += ingest_one(conn, code, trading_days)
        except Exception as exc:
            logger.error("%s: ingestion failed, skipping — %s", code, exc)
            failures.append(code)

    logger.info("Done. %d rows upserted. Failures: %s", total, failures or "none")
    return {"rows_upserted": total, "failures": failures}


if __name__ == "__main__":
    run()
