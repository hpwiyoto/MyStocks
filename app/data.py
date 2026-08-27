"""Read-only data access for the Streamlit app. Only queries MySQL and reads
the committed model metadata file -- no pipeline/feature/training logic here
(that belongs in /pipeline, /features, /engine)."""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import inspect

from engine.predict import MODEL_VERSION
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from pipeline.tickers import to_yfinance_symbol

logger = get_logger("app.data")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

CACHE_TTL = 300  # seconds
LIVE_PRICE_TTL = 30  # seconds -- short on purpose, this is the "what's it doing right now" overlay


def _missing_tables(engine, required: list[str]) -> list[str]:
    """Which of `required` tables don't exist yet -- lets each page show a
    friendly "run this phase first" message instead of a raw SQL traceback
    when a Streamlit session is opened before pipeline/features/engine have
    ever populated the database (found via testing: a fresh DB otherwise
    surfaces pandas.errors.DatabaseError straight to the user)."""
    inspector = inspect(engine)
    return [t for t in required if not inspector.has_table(t)]


@st.cache_data(ttl=CACHE_TTL)
def load_latest_predictions() -> pd.DataFrame:
    engine = get_engine()
    if _missing_tables(engine, ["predictions", "stocks", "feature_daily"]):
        return pd.DataFrame()
    df = pd.read_sql(
        """
        SELECT p.stock_code, s.name, s.sector, p.date, p.probability, p.decision,
               p.entry_price, p.stop_loss_price, p.take_profit_price, p.risk_reward_ratio,
               fd.regime
        FROM predictions p
        LEFT JOIN stocks s ON p.stock_code = s.code
        LEFT JOIN feature_daily fd ON p.stock_code = fd.stock_code AND p.date = fd.date
        INNER JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM predictions
            WHERE model_version = %(model_version)s
            GROUP BY stock_code
        ) latest ON p.stock_code = latest.stock_code AND p.date = latest.max_date
        WHERE p.model_version = %(model_version)s
        ORDER BY p.probability DESC
        """,
        engine,
        params={"model_version": MODEL_VERSION},
    )
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_price_history(stock_code: str, days: int = 260) -> pd.DataFrame:
    engine = get_engine()
    if _missing_tables(engine, ["price_history"]):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_sql(
        """
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE stock_code = %(code)s AND source_provider = 'yfinance'
        ORDER BY date DESC
        LIMIT %(days)s
        """,
        engine,
        params={"code": stock_code, "days": days},
    )
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def load_latest_feature_row(stock_code: str) -> dict | None:
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily"]):
        return None
    df = pd.read_sql(
        """
        SELECT * FROM feature_daily
        WHERE stock_code = %(code)s
        ORDER BY date DESC LIMIT 1
        """,
        engine,
        params={"code": stock_code},
    )
    return df.iloc[0].to_dict() if not df.empty else None


@st.cache_data(ttl=CACHE_TTL)
def load_latest_fundamental(stock_code: str) -> dict | None:
    engine = get_engine()
    if _missing_tables(engine, ["feature_fundamental_snapshot"]):
        return None
    df = pd.read_sql(
        """
        SELECT * FROM feature_fundamental_snapshot
        WHERE stock_code = %(code)s
        ORDER BY snapshot_date DESC LIMIT 1
        """,
        engine,
        params={"code": stock_code},
    )
    return df.iloc[0].to_dict() if not df.empty else None


@st.cache_data(ttl=CACHE_TTL)
def load_stock_list() -> pd.DataFrame:
    engine = get_engine()
    if _missing_tables(engine, ["stocks"]):
        return pd.DataFrame(columns=["code", "name", "sector"])
    return pd.read_sql("SELECT code, name, sector FROM stocks ORDER BY code", engine)


@st.cache_data(ttl=CACHE_TTL)
def feature_daily_row_count() -> int:
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily"]):
        return 0
    df = pd.read_sql("SELECT COUNT(*) AS c FROM feature_daily", engine)
    return int(df["c"].iloc[0])


def load_model_metadata() -> dict:
    path = os.path.join(MODEL_DIR, f"{MODEL_VERSION}_metadata.json")
    with open(path) as f:
        return json.load(f)


def days_since(date_str: str) -> int:
    trained = dt.date.fromisoformat(date_str)
    return (dt.date.today() - trained).days


@st.cache_data(ttl=LIVE_PRICE_TTL)
def load_live_prices(codes: tuple[str, ...]) -> dict[str, float]:
    """Current/live market price for a SMALL set of tickers (the ones on
    screen right now -- e.g. the top 15 -- never the full ~900-ticker
    universe, that's what the daily pipeline is for). Deliberately separate
    from the DB-backed prediction data: this is a pure display overlay, no
    feature/prediction recompute, so it's cheap enough (~0.3s/ticker via
    yfinance's lightweight `fast_info`, not a full `.history()` fetch) to
    run on every page load without the multi-minute pipeline cost. Returns
    only the tickers that succeeded -- caller falls back to the
    (necessarily one-day-stale-at-most) `entry_price` from `predictions`
    for anything missing here, e.g. a transient yfinance hiccup.
    """
    prices = {}
    for code in codes:
        try:
            prices[code] = float(yf.Ticker(to_yfinance_symbol(code)).fast_info["last_price"])
        except Exception as exc:
            logger.warning("%s: live price fetch failed, falling back to last close — %s", code, exc)
    return prices
