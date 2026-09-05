"""Read-only data access for the Streamlit app. Only queries the database and
reads the committed model metadata file -- no pipeline/feature/training logic
here (that belongs in /pipeline, /features, /engine)."""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import bindparam, inspect, text, update

from engine.predict import MODEL_VERSION
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION
from features.db import feature_daily
from features.news import fetch_news_headlines
from pipeline.db import get_engine
from pipeline.idx_rapidapi_source import RAPIDAPI_KEY, fetch_foreign_flow_all
from pipeline.logging_config import get_logger
from pipeline.tickers import to_yfinance_symbol

logger = get_logger("app.data")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

CACHE_TTL = 300  # seconds
LIVE_PRICE_TTL = 30  # seconds -- short on purpose, this is the "what's it doing right now" overlay
NEWS_TTL = 1800  # seconds -- headlines don't need 30s freshness like price does, and this is one
                 # extra outbound HTTP call per Detail Saham page load, no need to repeat it often
FOREIGN_FLOW_TTL = 21600  # 6h -- IDX foreign flow only changes once per trading session, no reason
                          # to refetch more often than this; also keeps RapidAPI's free-tier monthly
                          # quota (see pipeline/idx_rapidapi_source.py's reserve_requests) safe from a
                          # single ticker being viewed repeatedly in one sitting.


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
        text("""
        SELECT p.stock_code, s.name, s.sector, p.date, p.probability, p.decision,
               p.entry_price, p.stop_loss_price, p.take_profit_price, p.risk_reward_ratio,
               fd.regime
        FROM predictions p
        LEFT JOIN stocks s ON p.stock_code = s.code
        LEFT JOIN feature_daily fd ON p.stock_code = fd.stock_code AND p.date = fd.date
        INNER JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM predictions
            WHERE model_version = :model_version
            GROUP BY stock_code
        ) latest ON p.stock_code = latest.stock_code AND p.date = latest.max_date
        WHERE p.model_version = :model_version
        ORDER BY
            -- Decision tier first (BUY, then WATCH, then AVOID), probability
            -- only as the tiebreaker within a tier -- NOT probability alone.
            -- A gocap-floor AVOID (engine.decision.GOCAP_PRICE_FLOOR) can
            -- have a higher raw model probability than a real WATCH pick
            -- (the override is about tradeability, not the score itself),
            -- so sorting by probability alone let those rows crowd out
            -- genuine opportunities at the top of Home's ranked table.
            CASE p.decision WHEN 'BUY' THEN 0 WHEN 'WATCH' THEN 1 ELSE 2 END,
            p.probability DESC
        """),
        engine,
        params={"model_version": MODEL_VERSION},
    )
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_latest_turnaround_predictions() -> pd.DataFrame:
    """Same shape/joins as load_latest_predictions, separate function (not
    a parameterized shared one) because the decision tiers are genuinely
    different -- POTENSIAL/BELUM here, not BUY/WATCH/AVOID -- and the
    turnaround model only ever scores tickers currently in bearish/
    bottoming regime (see engine.predict_turnaround), so this is always a
    small subset of the full universe, not a parallel ranking of everyone."""
    engine = get_engine()
    if _missing_tables(engine, ["predictions", "stocks", "feature_daily"]):
        return pd.DataFrame()
    df = pd.read_sql(
        text("""
        SELECT p.stock_code, s.name, s.sector, s.industry, p.date, p.probability, p.decision,
               p.entry_price, fd.regime
        FROM predictions p
        LEFT JOIN stocks s ON p.stock_code = s.code
        LEFT JOIN feature_daily fd ON p.stock_code = fd.stock_code AND p.date = fd.date
        INNER JOIN (
            SELECT stock_code, MAX(date) AS max_date
            FROM predictions
            WHERE model_version = :model_version
            GROUP BY stock_code
        ) latest ON p.stock_code = latest.stock_code AND p.date = latest.max_date
        WHERE p.model_version = :model_version
        ORDER BY CASE p.decision WHEN 'POTENSIAL' THEN 0 ELSE 1 END, p.probability DESC
        """),
        engine,
        params={"model_version": TURNAROUND_MODEL_VERSION},
    )
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_price_history(stock_code: str, days: int = 260) -> pd.DataFrame:
    engine = get_engine()
    if _missing_tables(engine, ["price_history"]):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_sql(
        text("""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE stock_code = :code AND source_provider = 'yfinance'
        ORDER BY date DESC
        LIMIT :days
        """),
        engine,
        params={"code": stock_code, "days": days},
    )
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=FOREIGN_FLOW_TTL)
def load_foreign_flow(stock_code: str) -> pd.DataFrame:
    """On-demand fetch (RapidAPI IDX, see pipeline/idx_rapidapi_source.py)
    for ONE ticker, triggered the first time its Detail Saham page loads
    each cache window -- not a bulk daily job, so quota only gets spent on
    tickers someone actually looks at. Persists into
    feature_daily.net_foreign_flow (same UPDATE-only, never-insert-a-bare-
    row convention as scripts/backfill_foreign_flow.py) so the data stays
    available for the empirical/analysis use the user asked for even
    though it deliberately does NOT feed any model (see
    scripts/test_foreign_flow_feature.py -- tested, didn't help Swing).
    Returns [] gracefully (never raises) if RAPIDAPI_KEY isn't configured
    or the fetch fails -- this is a best-effort display complement, same
    contract as load_news."""
    if not RAPIDAPI_KEY:
        return pd.DataFrame(columns=["date", "value"])
    flows = fetch_foreign_flow_all([stock_code], timeframe="1y")
    df = flows.get(stock_code, pd.DataFrame(columns=["date", "value"]))
    if not df.empty:
        engine = get_engine()
        with engine.begin() as conn:
            stmt = (
                update(feature_daily)
                .where(feature_daily.c.stock_code == bindparam("code"), feature_daily.c.date == bindparam("d"))
                .values(net_foreign_flow=bindparam("val"))
            )
            conn.execute(stmt, [{"code": stock_code, "d": r["date"], "val": float(r["value"])} for _, r in df.iterrows()])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def load_foreign_flow_history(stock_code: str, days: int = 260) -> pd.DataFrame:
    """Reads whatever's already stored in feature_daily.net_foreign_flow --
    the chart-ready counterpart to load_foreign_flow above, which is what
    actually keeps that column current. Deliberately separate (DB read vs
    API fetch+write) so the chart doesn't wait on a live API call every
    render -- load_foreign_flow already ran earlier in the same page load
    and cached its result for FOREIGN_FLOW_TTL."""
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily"]):
        return pd.DataFrame(columns=["date", "net_foreign_flow"])
    df = pd.read_sql(
        text("""
        SELECT date, net_foreign_flow
        FROM feature_daily
        WHERE stock_code = :code AND net_foreign_flow IS NOT NULL
        ORDER BY date DESC
        LIMIT :days
        """),
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
        text("""
        SELECT * FROM feature_daily
        WHERE stock_code = :code
        ORDER BY date DESC LIMIT 1
        """),
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
        text("""
        SELECT * FROM feature_fundamental_snapshot
        WHERE stock_code = :code
        ORDER BY snapshot_date DESC LIMIT 1
        """),
        engine,
        params={"code": stock_code},
    )
    return df.iloc[0].to_dict() if not df.empty else None


@st.cache_data(ttl=CACHE_TTL)
def load_stock_list() -> pd.DataFrame:
    engine = get_engine()
    if _missing_tables(engine, ["stocks"]):
        return pd.DataFrame(columns=["code", "name", "sector", "industry"])
    return pd.read_sql("SELECT code, name, sector, industry FROM stocks ORDER BY code", engine)


@st.cache_data(ttl=CACHE_TTL)
def feature_daily_row_count() -> int:
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily"]):
        return 0
    df = pd.read_sql("SELECT COUNT(*) AS c FROM feature_daily", engine)
    return int(df["c"].iloc[0])


def load_model_metadata(model_version: str = MODEL_VERSION) -> dict:
    path = os.path.join(MODEL_DIR, f"{model_version}_metadata.json")
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


@st.cache_data(ttl=NEWS_TTL)
def load_news(stock_code: str, stock_name: str = "") -> list[dict]:
    """Display-only headline panel (Detail Saham) -- see features.news for
    why this is never fed into the model. Query by code+name together
    ("BBCA Bank Central Asia saham") when a name is available -- narrower
    than code alone, which for a short/common code can pull in unrelated
    results."""
    query = f"{stock_code} {stock_name} saham".strip()
    return fetch_news_headlines(query)
