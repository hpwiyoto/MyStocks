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
) -> None:
    engine = get_engine()
    init_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            tracked_positions.insert().values(
                user_email=user_email, stock_code=stock_code, model_version=model_version,
                entry_date=entry_date, entry_price=entry_price,
                stop_loss_price=stop_loss_price, take_profit_price=take_profit_price,
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
    pct_to_target, and the early-warning breakdown from engine.early_
    warning.check_position (days_held too, for display)."""
    positions = _load_positions_raw(user_email, status)
    if positions.empty:
        return positions

    from app.data import load_latest_feature_row, load_live_prices, load_price_history

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
        entry_date = pos["entry_date"]
        days_held = (dt.date.today() - entry_date).days if isinstance(entry_date, dt.date) else None

        warn = None
        if current_price is not None and feat_row:
            warn = check_position(entry_price, feat_row, float(current_price))

        row = pos.to_dict()
        row["current_price"] = current_price
        row["days_held"] = days_held
        row["pct_change_from_entry"] = (current_price - entry_price) / entry_price * 100 if current_price is not None else None
        row["pct_to_stop"] = (current_price - float(pos["stop_loss_price"])) / current_price * 100 if current_price is not None else None
        row["pct_to_target"] = (float(pos["take_profit_price"]) - current_price) / current_price * 100 if current_price is not None else None
        row["warning"] = bool(warn and warn["warning"])
        row["warning_detail"] = warn
        rows.append(row)
    return pd.DataFrame(rows)
