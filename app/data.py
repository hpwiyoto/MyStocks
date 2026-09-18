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
from engine.swing_configs import DEFAULT_CONFIG_ID, model_version_for
from features.db import feature_daily
from features.news import fetch_news_headlines
from pipeline.db import get_engine
from pipeline.idx_rapidapi_source import RAPIDAPI_KEY, fetch_foreign_flow_all
from pipeline.logging_config import get_logger
from pipeline.tickers import to_yfinance_symbol
from scripts.special_monitoring_board import ACTIVE_TICKERS as SPECIAL_MONITORING_TICKERS

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


def selected_swing_model_version() -> str:
    """Shared across every page that shows Swing results -- reads the same
    `persist_swing_config_id` session_state slot the Swing page's config
    toggle (engine.swing_configs) writes, so switching there stays
    consistent everywhere (Home/Detail Saham/Momentum Screener/Rekomendasi
    Emitten) instead of each page silently defaulting back to the shipped
    config on its own. Falls back to the default when the user hasn't
    opened the Swing page's toggle yet this session."""
    config_id = st.session_state.get("persist_swing_config_id", DEFAULT_CONFIG_ID)
    return model_version_for(config_id)


@st.cache_data(ttl=CACHE_TTL)
def load_latest_predictions(model_version: str = MODEL_VERSION) -> pd.DataFrame:
    """`model_version`: which Swing config's predictions to load -- see
    engine.swing_configs for the toggle between the top-5 target/stop/
    horizon configs. Defaults to the shipped model so every existing
    caller that doesn't pass it is unaffected."""
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
        params={"model_version": model_version},
    )
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_current_prediction_snapshot(stock_code: str, model_version: str) -> dict | None:
    """One ticker's LATEST probability/decision/regime for a specific
    config -- a lean single-row version of load_latest_predictions, so
    app.positions.load_positions_with_progress can look up each tracked
    position's CURRENT Swing read (for the entry-vs-now comparison the
    user asked for) without pulling the whole ~900-row predictions table
    per position in a loop."""
    engine = get_engine()
    if _missing_tables(engine, ["predictions", "feature_daily"]):
        return None
    df = pd.read_sql(
        text("""
        SELECT p.probability, p.decision, fd.regime
        FROM predictions p
        LEFT JOIN feature_daily fd ON p.stock_code = fd.stock_code AND p.date = fd.date
        WHERE p.stock_code = :code AND p.model_version = :model_version
        ORDER BY p.date DESC LIMIT 1
        """),
        engine,
        params={"code": stock_code, "model_version": model_version},
    )
    return df.iloc[0].to_dict() if not df.empty else None


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


@st.cache_data(ttl=CACHE_TTL)
def load_wyckoff_status(stock_code: str) -> dict | None:
    """Latest Wyckoff phase/spring/upthrust for one ticker -- computed
    on-demand from load_price_history's 260-day window, same on-the-fly
    pattern as features.support_resistance's chart-only S/R (not stored
    in feature_daily, since it isn't a model input -- see features/
    wyckoff.py's docstring). Shared by Detail Saham's display badge and
    the "Tandai Beli" entry-snapshot (app.positions.mark_position) so
    both read the exact same computation instead of two copies drifting
    apart. Returns None during the warmup period a ticker's own history
    hasn't cleared yet."""
    from features.wyckoff import compute_wyckoff_features

    price_df = load_price_history(stock_code, days=260)
    if price_df.empty:
        return None
    wyckoff_df = compute_wyckoff_features(price_df.assign(stock_code=stock_code))
    last = wyckoff_df.iloc[-1]
    if pd.isna(last["wyckoff_phase"]):
        return None
    return last.to_dict()


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
def load_data_freshness() -> dt.date | None:
    """Most recent date present in feature_daily -- i.e. how current the
    prices/RSI/MACD/CMF everyone reads actually are. Surfaced directly in
    the UI (rather than silently trusted) after a real incident: the
    scheduler process was asleep for 4 days (laptop off/asleep over that
    stretch, see scripts/scheduler_loop.py's catch-up fix) and every page
    kept quietly showing Sept-1 data with no indication anything was stale.
    Returns None if feature_daily doesn't exist yet or is empty.
    """
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily"]):
        return None
    df = pd.read_sql("SELECT MAX(date) AS max_date FROM feature_daily", engine)
    value = df["max_date"].iloc[0] if not df.empty else None
    if value is None or value != value:  # NaT/NaN guard, same pattern as safe_ratio's NaN check
        return None
    return pd.to_datetime(value).date()


@st.cache_data(ttl=CACHE_TTL)
def load_screener_raw_panel(lookback_days: int = 60) -> pd.DataFrame:
    """Bulk per-(ticker, date) panel across the WHOLE universe for the last
    ~`lookback_days` TRADING days, feeding features.momentum_screener's
    MACD-status classification, RSI/MACD divergence detection, and (via
    high/low, added for the Anchored VWAP validated-signal criterion --
    see scripts/test_strategy_6_criteria.py) the AVWAP-from-50-day-low
    check on the Momentum Screener page. Unlike load_price_history (one
    ticker), this scores the whole universe at once -- same shape as
    load_latest_predictions.

    Cutoff is a plain calendar-date WHERE clause (lookback_days*2 days back,
    a generous buffer for weekends/holidays) computed in Python rather than
    DATE_SUB/julianday SQL -- this project runs on SQLite locally and MySQL
    in production (see pipeline.db.get_engine), and a literal date string
    compares correctly on both without dialect-specific date arithmetic.
    """
    engine = get_engine()
    if _missing_tables(engine, ["feature_daily", "price_history"]):
        return pd.DataFrame()
    cutoff = (dt.date.today() - dt.timedelta(days=lookback_days * 2)).isoformat()
    df = pd.read_sql(
        text("""
        SELECT fd.stock_code, fd.date, ph.close, ph.high, ph.low, ph.volume,
               fd.rsi_14, fd.macd, fd.macd_signal, fd.macd_hist, fd.macd_hist_slope_3d,
               fd.cmf_20, fd.rvol_20, fd.regime
        FROM feature_daily fd
        JOIN price_history ph ON ph.stock_code = fd.stock_code AND ph.date = fd.date AND ph.source_provider = 'yfinance'
        WHERE fd.date >= :cutoff
        ORDER BY fd.stock_code, fd.date
        """),
        engine,
        params={"cutoff": cutoff},
    )
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_liquidity(window_days: int = 60) -> pd.DataFrame:
    """Average daily traded VALUE (close * volume, in Rupiah) over the last
    ~`window_days` trading days, one row per ticker -- a display-only
    liquidity gauge for the screener pages' optional "liquid names only"
    filter. Deliberately NOT a model feature: scripts/test_liquidity_
    filter.py backtested gating each screener's candidate pool on this and
    found it does NOT improve (and noticeably hurt Swing's) top-2 win
    rate, so it's surfaced as information + an opt-in filter, never a hard
    universe cut. Raw share `volume` alone would be misleading (10M shares
    of a Rp50 stock is Rp0.5b; 1M shares of a Rp10k stock is Rp10b) --
    traded value is the standard cross-stock-comparable measure.

    Same calendar-date-string WHERE clause style as load_screener_raw_panel
    (SQLite-and-MySQL-safe, no dialect date math); the exact trailing
    `window_days` count is taken in pandas after the fetch.
    """
    engine = get_engine()
    if _missing_tables(engine, ["price_history"]):
        return pd.DataFrame(columns=["stock_code", "avg_traded_value"])
    cutoff = (dt.date.today() - dt.timedelta(days=window_days * 2)).isoformat()
    df = pd.read_sql(
        text("""
        SELECT stock_code, date, close, volume
        FROM price_history
        WHERE date >= :cutoff AND source_provider = 'yfinance'
        ORDER BY stock_code, date
        """),
        engine,
        params={"cutoff": cutoff},
    )
    if df.empty:
        return pd.DataFrame(columns=["stock_code", "avg_traded_value"])
    df["traded_value"] = df["close"].astype(float) * df["volume"].astype(float)
    last_n = df.groupby("stock_code").tail(window_days)
    out = last_n.groupby("stock_code")["traded_value"].mean().reset_index()
    out.columns = ["stock_code", "avg_traded_value"]
    return out


SUSPENSION_FREEZE_DAYS = 2  # consecutive most-recent trading days of flat OHLC + zero volume
ARA_STREAK_DAYS = 1  # most-recent trading day(s) of flat OHLC + NONZERO volume (price-limit hit)


def _flatness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by load_suspended_tickers and untradeable_reason below --
    one flat-OHLC check, split into the two DISTINCT reasons a flat quote
    can mean "you cannot transact at the displayed price right now"."""
    df = df.copy()
    is_flat = (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"])
    df["frozen"] = is_flat & (df["volume"] == 0)
    df["ara_streak"] = is_flat & (df["volume"] > 0)
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_suspended_tickers() -> set[str]:
    """Stocks that are effectively UNTRADEABLE right now, for either of
    two distinct reasons -- both "actively misleading" for a screener to
    rank as a live opportunity, so both get excluded from every screener
    page's results the same way this function's name already implied for
    the first reason alone (confirmed via a real user report each time
    this gap was found):

    1. SUSPENDED: the most recent SUSPENSION_FREEZE_DAYS trading days all
       show identical open/high/low/close AND zero volume -- the
       signature yfinance/IDX actually produce for a halted symbol
       (confirmed real, not a guess: SAFE, Swing BUY 2026-09-01, then
       every session since was open=high=low=close=875 volume=0 -- see
       scripts/check_suspension_risk_v2.py). You cannot buy OR sell it at
       any price.

    2. ARA/ARB STREAK: flat open=high=low=close but NONZERO volume on the
       most recent ARA_STREAK_DAYS trading day(s) -- IDX's daily
       price-limit mechanism (Auto Reject Atas/Bawah): every matched
       trade that day cleared at exactly the limit price because the
       order book is almost entirely one-sided (a queue of buyers at the
       ceiling, or sellers at the floor, with essentially no one on the
       other side to fill a NEW order against). Confirmed real, not
       hypothetical: this SAME ticker (SAFE) reopened from the
       suspension above on 2026-09-15 and then hit this exact pattern 3
       trading days running (875->960->1055->1160, each ~+10%) -- the
       Swing model scored it BUY at 68-69% probability on all three,
       quoting an entry_price you very likely can't actually get filled
       at. 1 day is enough here (unlike the suspension case, which waits
       for 2) -- the FIRST day of a streak like this is exactly when a
       screener would otherwise flag it as a fresh opportunity, the
       precise failure mode this guards against.

    3. SPECIAL MONITORING BOARD (FCA): currently listed on IDX's own
       "Papan Pemantauan Khusus" (see scripts/special_monitoring_board.py)
       -- trades under Full Call Auction (periodic call-auction matching)
       instead of continuous trading, a DIFFERENT mechanism from 1/2 above
       and NOT detectable from OHLCV shape at all (confirmed real: TGUK
       reported un-buyable by the user despite a completely ordinary-
       looking continuous price chart). SAFE/TRUK/PACK below are also on
       this board independently of already being caught by reason 2.
    """
    engine = get_engine()
    untradeable = set(SPECIAL_MONITORING_TICKERS)
    if _missing_tables(engine, ["price_history"]):
        return untradeable
    cutoff = (dt.date.today() - dt.timedelta(days=15)).isoformat()
    df = pd.read_sql(
        text("""
        SELECT stock_code, date, open, high, low, close, volume
        FROM price_history
        WHERE date >= :cutoff AND source_provider = 'yfinance'
        ORDER BY stock_code, date
        """),
        engine,
        params={"cutoff": cutoff},
    )
    if df.empty:
        return untradeable
    df = _flatness_flags(df)
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date")
        if len(g) >= SUSPENSION_FREEZE_DAYS and g["frozen"].tail(SUSPENSION_FREEZE_DAYS).all():
            untradeable.add(code)
        elif len(g) >= ARA_STREAK_DAYS and g["ara_streak"].tail(ARA_STREAK_DAYS).all():
            untradeable.add(code)
    return untradeable


def untradeable_reason(code: str) -> str | None:
    """Which of load_suspended_tickers's three reasons applies to this ONE
    ticker (or None if it isn't currently flagged at all) -- for Detail
    Saham's banner, which needs to say WHICH one (they read very
    differently: frozen for days, hit a price limit yesterday, or on
    IDX's Special Monitoring Board) rather than reuse that function's flat
    exclusion set, which is deliberately just a set for its other callers
    that only ever check membership.
    """
    if code in SPECIAL_MONITORING_TICKERS:
        return "special_monitoring"
    engine = get_engine()
    if _missing_tables(engine, ["price_history"]):
        return None
    cutoff = (dt.date.today() - dt.timedelta(days=15)).isoformat()
    df = pd.read_sql(
        text("""
        SELECT date, open, high, low, close, volume FROM price_history
        WHERE stock_code = :code AND date >= :cutoff AND source_provider = 'yfinance'
        ORDER BY date
        """),
        engine,
        params={"code": code, "cutoff": cutoff},
    )
    if df.empty:
        return None
    df = _flatness_flags(df.sort_values("date"))
    if len(df) >= SUSPENSION_FREEZE_DAYS and df["frozen"].tail(SUSPENSION_FREEZE_DAYS).all():
        return "suspended"
    if len(df) >= ARA_STREAK_DAYS and df["ara_streak"].tail(ARA_STREAK_DAYS).all():
        return "ara_streak"
    return None


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


def best_swing_config_id() -> str:
    """Which of engine.swing_configs' 5 configs has the highest pooled
    Wilson LB (95%) -- direct user request for a star/marker on "the one
    that's numerically better". Wilson LB, not top5_lift (the metric
    scripts/search_swing_target.py's own ranking/`rank`/`label` use):
    top5_lift answers "how well does this config's own top-5%-of-
    predictions separate from ITS OWN null baseline", the right question
    for comparing configs whose underlying labels differ, but not the
    question a user picking a config to actually trade day-to-day is
    asking -- "if I follow every real BUY signal, what's my actual floor
    win rate". Deliberately recomputed live from each config's own
    metadata (not cached in engine.swing_configs) so this stays correct
    if any config gets retrained."""
    from engine.swing_configs import SWING_CONFIGS, DEFAULT_CONFIG_ID

    best_id, best_lb = DEFAULT_CONFIG_ID, -1.0
    for c in SWING_CONFIGS:
        try:
            meta = load_model_metadata(c["model_version"])
        except (OSError, json.JSONDecodeError):
            continue
        lb = (meta.get("walk_forward_validation") or {}).get("pooled", {}).get("wilson_lb_95")
        if lb is not None and lb > best_lb:
            best_id, best_lb = c["id"], lb
    return best_id


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


IHSG_TREND_TTL = 1800  # 30 min -- a slow-moving context indicator, not a live price; no need for LIVE_PRICE_TTL's 30s


@st.cache_data(ttl=IHSG_TREND_TTL)
def load_ihsg_trend() -> dict | None:
    """IHSG's own recent trend -- pure display context, not a model input.
    scripts/check_fold_drift.py found the Swing model's precision
    correlates with IHSG's direction (weaker when IHSG is declining), and
    scripts/test_ihsg_regime_feature.py found that feeding IHSG's trend
    INTO the model as a training feature makes things dramatically worse
    (see that script's docstring for why). This surfaces the same context
    to the HUMAN instead, who can apply judgment a raw model input
    couldn't. Returns None on fetch failure (e.g. no network) rather than
    raising -- this is a supplementary caution panel, never something that
    should block a page from rendering.
    """
    try:
        hist = yf.Ticker("^JKSE").history(period="6mo")
    except Exception as exc:
        logger.warning("IHSG trend fetch failed: %s", exc)
        return None
    if hist.empty or len(hist) < 21:
        return None
    close = hist["Close"]
    last = float(close.iloc[-1])
    return {
        "last": last,
        "ret_20d": (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None,
        "ret_50d": (last / float(close.iloc[-51]) - 1) * 100 if len(close) >= 51 else None,
        "as_of": close.index[-1].date(),
    }


@st.cache_data(ttl=IHSG_TREND_TTL)
def _fetch_ihsg_history(period: str) -> pd.DataFrame:
    """Cached raw fetch -- RAISES on failure/empty instead of catching, so
    only a genuinely successful fetch ever gets cached. If this caught its
    own exceptions and returned an empty DataFrame like load_ihsg_history()
    below used to, st.cache_data would store that empty result for the
    full IHSG_TREND_TTL (30 min) -- confirmed real risk on this Zscaler-
    proxied network (yfinance has logged real transient failures here
    before, e.g. 'Could not resolve host: query2.finance.yahoo.com'), and
    the symptom is exactly a chart that stays silently blank for half an
    hour after one bad blip, even once the network's fine again."""
    hist = yf.Ticker("^JKSE").history(period=period)
    if hist.empty:
        raise ValueError(f"yfinance returned no IHSG history for period={period!r}")
    return pd.DataFrame({"date": hist.index.tz_localize(None), "close": hist["Close"].astype(float)}).reset_index(drop=True)


def load_ihsg_history(period: str = "6mo") -> pd.DataFrame:
    """Plain IHSG close series for the Home page chart -- companion to
    load_ihsg_trend() above (which only returns the summary %/last-value
    dict, not the series a chart needs). Thin uncached wrapper around
    _fetch_ihsg_history() so a failure is retried on the next rerun rather
    than cached (see that function's docstring). Same fail-soft contract
    as load_ihsg_trend: an empty DataFrame on fetch failure, never an
    exception -- this is a supplementary context panel, not something
    that should block Home from rendering. `period` is any string
    yfinance's history() accepts (1mo/3mo/6mo/1y/...).
    """
    try:
        return _fetch_ihsg_history(period)
    except Exception as exc:
        logger.warning("IHSG history fetch failed: %s", exc)
        return pd.DataFrame(columns=["date", "close"])


@st.cache_data(ttl=NEWS_TTL)
def load_news(stock_code: str, stock_name: str = "") -> list[dict]:
    """Display-only headline panel (Detail Saham) -- see features.news for
    why this is never fed into the model. Query by code+name together
    ("BBCA Bank Central Asia saham") when a name is available -- narrower
    than code alone, which for a short/common code can pull in unrelated
    results."""
    query = f"{stock_code} {stock_name} saham".strip()
    return fetch_news_headlines(query)
