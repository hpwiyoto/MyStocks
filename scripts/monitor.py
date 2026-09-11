"""Fase 6 monitoring: detect distinct failure modes and alert.

1. Explicit failures -- pipeline.ingest_price / engine.predict already
   retry + isolate per-ticker, but if a ticker still ends up in one of
   their `failures` lists, that's worth surfacing.
2. Silently-stale price data -- ingest can "succeed" (no exception) yet the
   latest price_history date hasn't advanced, e.g. yfinance quietly serving
   cached data. Can't be detected from exceptions alone, only from the data.
3. Silently-stale (or silently-broken) predictions -- confirmed via a real
   incident: a schema mismatch (missing column) made EVERY prediction fail
   with an exception, which per-ticker isolation dutifully caught and logged
   -- but ingest itself had succeeded (price_history was current), and this
   file originally only ever checked price_history's freshness, so
   check_and_alert() reported "semua up to date" the whole time predictions
   were completely broken. Checking the predictions table's own freshness
   independently of price_history closes that blind spot; a caller also
   passing predict failures surfaces the exact cause immediately rather
   than waiting for the staleness threshold to trip.

(Turnaround dropped from both PREDICTION_MODELS and check_and_alert's
signature after the Turnaround page/daily step were retired -- see
scripts/run_daily.py's docstring.)

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


# (label, model_version) -- checked independently of price_history so a
# schema mismatch or any other predict-only failure that leaves
# price_history current still gets caught (see module docstring, point 3).
PREDICTION_MODELS = [
    ("swing", "direction_xgboost_v5"),
]


def check_stale_predictions() -> list[str]:
    engine = get_engine()
    today = dt.date.today()
    stale = []
    with engine.connect() as conn:
        for label, model_version in PREDICTION_MODELS:
            row = conn.execute(
                text("SELECT MAX(date) FROM predictions WHERE model_version = :mv"),
                {"mv": model_version},
            ).first()
            latest = coerce_date(row[0]) if row else None
            if latest is None:
                stale.append(f"{label} (belum pernah ada prediksi)")
                continue
            days_behind = (today - latest).days
            if days_behind > STALE_AFTER_DAYS:
                stale.append(f"{label} (terakhir {latest}, {days_behind} hari lalu)")
    return stale


def _format_failures(label: str, failures: list[str] | None, limit: int = 10) -> str | None:
    """A schema-wide bug (the real incident this guards against) fails EVERY
    ticker, not a handful -- dumping hundreds of codes into one alert line
    is unreadable, so this truncates while still saying how many total."""
    if not failures:
        return None
    shown = ", ".join(failures[:limit])
    more = f" (+{len(failures) - limit} lainnya)" if len(failures) > limit else ""
    return f"{label}: {shown}{more}"


def check_and_alert(
    ingest_failures: list[str] | None = None,
    predict_failures: list[str] | None = None,
) -> None:
    problems = []

    for text_line in [
        _format_failures("Gagal ingest eksplisit", ingest_failures),
        _format_failures("Gagal prediksi swing", predict_failures),
    ]:
        if text_line:
            problems.append(text_line)

    stale = check_stale_data()
    if stale:
        problems.append(f"Data harga basi (>{STALE_AFTER_DAYS} hari tanpa update): {'; '.join(stale)}")

    stale_pred = check_stale_predictions()
    if stale_pred:
        problems.append(f"Prediksi basi (>{STALE_AFTER_DAYS} hari tanpa update): {'; '.join(stale_pred)}")

    if problems:
        send_alert(" | ".join(problems))
    else:
        logger.info("Monitor: semua ticker & prediksi up to date, tidak ada masalah.")


if __name__ == "__main__":
    check_and_alert()
