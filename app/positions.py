"""Read/write access for a user's tracked positions (app.db.tracked_
positions) -- kept SEPARATE from app/data.py, which is deliberately
read-only (see its own docstring); marking/closing a position is a
mutation, not a cached query, so it doesn't belong there.
"""
import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import select, update

from app.db import init_schema, tracked_positions
from engine.early_warning import check_position
from pipeline.db import get_engine

# Status shown per tracked position -- direct user request ("dtambah
# statusnya... supaya ada progress prediksinya"), a single at-a-glance
# read instead of making the user piece it together from the raw numbers.
# Priority order matters: checked top to bottom, first match wins (e.g. a
# position that's both past its horizon AND showing warning signals is
# reported as EXPIRED first -- the model's own forecast window has
# already lapsed, which matters more than a signal read within a window
# that's no longer valid).
STATUS_LABELS = {
    "target_hit": "🎯 Target Tercapai",
    "stop_hit": "🛑 Stop Loss Tercapai",
    "expired": "⏳ Prediksi Kadaluarsa",
    "warning": "⚠️ Peringatan Dini",
    "under_pressure": "🟡 Dalam Tekanan",
    "on_track": "✅ On Track",
}
# Trading-day horizons run on TRADING days; entry_date/today are calendar
# days. Same approximation scripts/check_suspension_risk_v2.py already
# uses elsewhere in this project (LOOKBACK_DAYS * 1.5) to convert one to
# the other without a second price_history query just for a day-count.
CALENDAR_TO_TRADING_DAY_BUFFER = 1.5


def _position_status(current_price: float | None, entry_price: float, stop_loss_price: float,
                      take_profit_price: float, days_held: int | None, horizon_days: int | None,
                      warning: bool) -> str:
    if current_price is None:
        return "on_track"
    if current_price >= take_profit_price:
        return "target_hit"
    if current_price <= stop_loss_price:
        return "stop_hit"
    if days_held is not None and horizon_days is not None and days_held > horizon_days * CALENDAR_TO_TRADING_DAY_BUFFER:
        return "expired"
    if warning:
        return "warning"
    if current_price < entry_price:
        return "under_pressure"
    return "on_track"


def has_active_position(user_email: str, stock_code: str) -> bool:
    engine = get_engine()
    init_schema(engine)
    with engine.connect() as conn:
        row = conn.execute(
            select(tracked_positions.c.id).where(
                tracked_positions.c.user_email == user_email,
                tracked_positions.c.stock_code == stock_code,
                tracked_positions.c.status == "active",
            )
        ).fetchone()
    return row is not None


def mark_position(
    user_email: str, stock_code: str, model_version: str, entry_date, entry_price: float,
    stop_loss_price: float, take_profit_price: float,
    entry_probability: float | None = None, entry_regime: str | None = None,
    entry_wyckoff_phase: str | None = None, entry_target_pct: float | None = None,
    entry_stop_pct: float | None = None, entry_horizon_days: int | None = None,
) -> None:
    """`entry_date`: accepts a plain `datetime.date`, a `pandas.Timestamp`
    (what a value read back from `predictions` via pd.read_sql actually
    is, NOT a plain date -- found the hard way: SQLite's Date column type
    rejects anything but a real `datetime.date`, so a Timestamp passed
    straight through raised at insert time), or an ISO date string --
    normalized here once so every caller (Swing's cards, Swing's quick-
    mark expander, Detail Saham) is protected the same way.

    The `entry_*` snapshot args (all optional, default None so old call
    sites keep working) record what the Swing recommendation actually
    said AT THE MOMENT this was marked -- direct user request ("direcord
    hasil rekomendasi swing nya apa saat di klik tandai beli"). See
    app/db.py's tracked_positions docstring for why these are snapshotted
    rather than looked up live later."""
    engine = get_engine()
    init_schema(engine)
    entry_date = pd.Timestamp(entry_date).date()
    with engine.begin() as conn:
        conn.execute(
            tracked_positions.insert().values(
                user_email=user_email, stock_code=stock_code, model_version=model_version,
                entry_date=entry_date, entry_price=entry_price,
                stop_loss_price=stop_loss_price, take_profit_price=take_profit_price,
                entry_probability=entry_probability, entry_regime=entry_regime,
                entry_wyckoff_phase=entry_wyckoff_phase, entry_target_pct=entry_target_pct,
                entry_stop_pct=entry_stop_pct, entry_horizon_days=entry_horizon_days,
                status="active",
            )
        )
    st.cache_data.clear()


def close_position(position_id: int, closed_price: float, closed_reason: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            update(tracked_positions)
            .where(tracked_positions.c.id == position_id)
            .values(closed_date=dt.date.today(), closed_price=closed_price, closed_reason=closed_reason, status="closed")
        )
    st.cache_data.clear()


def count_active_positions(user_email: str) -> int:
    """Cheap count for a card/badge (e.g. Home's shortcut) -- doesn't fetch
    live prices/features per position the way load_positions_with_progress
    does, so it's safe to call on every Home page load."""
    return len(_load_positions_raw(user_email, "active"))


@st.cache_data(ttl=60)
def _load_positions_raw(user_email: str, status: str | None) -> pd.DataFrame:
    engine = get_engine()
    init_schema(engine)
    query = select(tracked_positions).where(tracked_positions.c.user_email == user_email)
    if status:
        query = query.where(tracked_positions.c.status == status)
    query = query.order_by(tracked_positions.c.entry_date.desc())
    return pd.read_sql(query, engine)


def load_positions_with_progress(user_email: str, status: str | None = "active") -> pd.DataFrame:
    """Every active position, joined with the ticker's LATEST close +
    feature row so the caller doesn't have to fetch those separately per
    row -- adds current_price, pct_change_from_entry, pct_to_stop,
    pct_to_target, the early-warning breakdown from engine.early_warning.
    check_position, days_held, an overall `status` (STATUS_LABELS key),
    and a `current_` snapshot (probability/regime/wyckoff_phase) to
    compare against the `entry_` snapshot taken when the position was
    marked -- direct user request for an entry-vs-now comparison."""
    positions = _load_positions_raw(user_email, status)
    if positions.empty:
        return positions

    from app.data import load_current_prediction_snapshot, load_latest_feature_row, load_live_prices, load_model_metadata, load_price_history, load_wyckoff_status

    codes = tuple(positions["stock_code"].unique())
    live_prices = load_live_prices(codes)

    rows = []
    for _, pos in positions.iterrows():
        code = pos["stock_code"]
        feat_row = load_latest_feature_row(code) or {}
        current_price = live_prices.get(code)
        if current_price is None:
            # feature_daily has no `close` column -- fall back to the
            # last known daily close from price_history instead.
            price_df = load_price_history(code, days=1)
            current_price = float(price_df["close"].iloc[-1]) if not price_df.empty else None
        entry_price = float(pos["entry_price"])
        stop_loss_price = float(pos["stop_loss_price"])
        take_profit_price = float(pos["take_profit_price"])
        entry_date = pos["entry_date"]
        days_held = (dt.date.today() - entry_date).days if isinstance(entry_date, dt.date) else None

        warn = None
        if current_price is not None and feat_row:
            warn = check_position(entry_price, feat_row, float(current_price))
        warning = bool(warn and warn["warning"])

        # Prefer the SNAPSHOT taken at mark-time (immune to the default
        # config being retrained in place later, see app/db.py) -- fall
        # back to a live lookup only for positions marked before this
        # column existed.
        horizon_days = pos.get("entry_horizon_days")
        if horizon_days is None:
            try:
                horizon_days = load_model_metadata(pos["model_version"])["horizon_days"]
            except (OSError, KeyError):
                pass  # a since-retired/renamed config -- status falls back gracefully without it

        current_snapshot = load_current_prediction_snapshot(code, pos["model_version"]) or {}
        current_wyckoff = load_wyckoff_status(code) or {}

        row = pos.to_dict()
        row["current_price"] = current_price
        row["days_held"] = days_held
        row["pct_change_from_entry"] = (current_price - entry_price) / entry_price * 100 if current_price is not None else None
        row["pct_to_stop"] = (current_price - stop_loss_price) / current_price * 100 if current_price is not None else None
        row["pct_to_target"] = (take_profit_price - current_price) / current_price * 100 if current_price is not None else None
        row["warning"] = warning
        row["warning_detail"] = warn
        row["status"] = _position_status(current_price, entry_price, stop_loss_price, take_profit_price, days_held, horizon_days, warning)
        row["current_probability"] = current_snapshot.get("probability")
        row["current_decision"] = current_snapshot.get("decision")
        row["current_regime"] = current_snapshot.get("regime")
        row["current_wyckoff_phase"] = current_wyckoff.get("wyckoff_phase")
        rows.append(row)
    return pd.DataFrame(rows)
