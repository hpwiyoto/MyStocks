"""Detects newly-appeared BUY signals across all 5 Swing configs after
the daily pipeline runs, and emails the admin -- direct user request.

"Newly-appeared" (not "every ticker currently BUY") is a deliberate
choice: notifying every day about a ticker that's been BUY for a week
would bury the one thing worth acting on (a fresh signal) in repeats of
things already seen and already decided about. A ticker counts as new
if its MOST RECENT prediction row for a config is BUY and either it has
no earlier row, or that earlier row wasn't BUY.

Per-ticker latest-vs-previous (not a single global "today vs yesterday"
date) because engine.predict.run isolates failures per ticker -- a
handful of tickers can legitimately have an older "latest" row than the
rest after a run with partial failures. Comparing each ticker against
its OWN prior row is correct regardless; comparing everyone against one
global calendar date is not (see find_new_buy_signals's overall_latest_date
guard below for the other half of this: skip a ticker whose "latest" row
ISN'T from today's run at all -- that's stale data, not a real "new"
signal, even though it'd otherwise look BUY-with-no-recent-BUY-before-it).

Usage:
    python -m scripts.notify_buy_signals
"""
import pandas as pd
from sqlalchemy import text

from app.data import load_suspended_tickers
from app.email_notify import send_buy_signal_notification
from engine.swing_configs import SWING_CONFIGS
from pipeline.db import get_engine
from pipeline.logging_config import get_logger

logger = get_logger("scripts.notify_buy_signals")


def _new_buys_for_config(engine, model_version: str, excluded: set[str]) -> pd.DataFrame:
    df = pd.read_sql(
        text("""
        SELECT stock_code, date, decision, probability, entry_price, take_profit_price, stop_loss_price
        FROM predictions
        WHERE model_version = :mv
        ORDER BY stock_code, date DESC
        """),
        engine,
        params={"mv": model_version},
    )
    if df.empty:
        return df

    overall_latest_date = df["date"].max()
    rows = []
    for code, g in df.groupby("stock_code", sort=False):
        g = g.reset_index(drop=True)
        latest = g.iloc[0]
        if latest["decision"] != "BUY":
            continue
        if latest["date"] != overall_latest_date:
            continue  # this ticker's "latest" is stale, not part of today's run
        if code in excluded:
            continue
        previous_was_buy = len(g) > 1 and g.iloc[1]["decision"] == "BUY"
        if previous_was_buy:
            continue
        rows.append(latest)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=df.columns)


def find_new_buy_signals() -> list[dict]:
    engine = get_engine()
    excluded = load_suspended_tickers()
    new_buys = []
    for cfg in SWING_CONFIGS:
        hits = _new_buys_for_config(engine, cfg["model_version"], excluded)
        for _, r in hits.iterrows():
            new_buys.append({
                "config_label": cfg["label"],
                "stock_code": r["stock_code"],
                "probability": float(r["probability"]),
                "entry_price": float(r["entry_price"]),
                "take_profit_price": float(r["take_profit_price"]),
                "stop_loss_price": float(r["stop_loss_price"]),
            })
        logger.info("%s: %d new BUY signal(s)", cfg["id"], len(hits))
    return new_buys


def check_and_notify() -> None:
    new_buys = find_new_buy_signals()
    if not new_buys:
        logger.info("No new BUY signals across any Swing config -- no email sent.")
        return
    logger.info("%d new BUY signal(s) total -- sending notification email.", len(new_buys))
    send_buy_signal_notification(new_buys)


if __name__ == "__main__":
    check_and_notify()
