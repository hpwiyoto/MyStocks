"""Fase 6 monitoring: detect two distinct failure modes and alert.

1. Explicit failures -- pipeline.ingest_price already retries + isolates
   per-ticker, but if a ticker still ends up in its `failures` list, that's
   worth surfacing.
2. Silently-stale data -- ingest can "succeed" (no exception) yet the latest
   price_history date hasn't advanced, e.g. yfinance quietly serving cached
   data. This can't be detected from exceptions alone, only by checking the
   actual data.

Alerting is pluggable: always logs (console + data/logs/pipeline.log via
pipeline.logging_config), and ADDITIONALLY sends a Telegram message if
TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set in the environment. Without
those, it's log-only -- a real notification channel can be wired in later
without touching the detection logic.
"""
import datetime as dt
import os

import requests
from sqlalchemy import text

from pipeline.db import coerce_date, get_engine
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS

logger = get_logger("scripts.monitor")

STALE_AFTER_DAYS = 4  # tolerates a weekend + one holiday


def send_alert(message: str) -> None:
    logger.warning("ALERT: %s", message)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": f"[MyStocks] {message}"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to send Telegram alert (falling back to log-only): %s", exc)


def check_stale_data(tickers: list[str] | None = None) -> list[str]:
    tickers = tickers or SEED_TICKERS
    engine = get_engine()
    today = dt.date.today()
    stale = []
    with engine.connect() as conn:
        for code in tickers:
            row = conn.execute(
                text("SELECT MAX(date) FROM price_history WHERE stock_code = :code"),
                {"code": code},
            ).first()
            latest = coerce_date(row[0]) if row else None
            if latest is None:
                stale.append(f"{code} (belum ada data sama sekali)")
                continue
            days_behind = (today - latest).days
            if days_behind > STALE_AFTER_DAYS:
                stale.append(f"{code} (terakhir {latest}, {days_behind} hari lalu)")
    return stale


def check_and_alert(ingest_failures: list[str] | None = None) -> None:
    problems = []

    if ingest_failures:
        problems.append(f"Gagal ingest eksplisit: {', '.join(ingest_failures)}")

    stale = check_stale_data()
    if stale:
        problems.append(f"Data basi (>{STALE_AFTER_DAYS} hari tanpa update): {'; '.join(stale)}")

    if problems:
        send_alert(" | ".join(problems))
    else:
        logger.info("Monitor: semua ticker up to date, tidak ada masalah.")


if __name__ == "__main__":
    check_and_alert()
