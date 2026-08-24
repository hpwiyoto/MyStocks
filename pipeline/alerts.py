"""Freshness alerts for data sources that require a manual step.

`foreign_flow` can only be updated by a human downloading a CSV from IDX's
website (see import_foreign_flow.py — automated scraping is blocked and
intentionally not bypassed). Nothing else in this pipeline will ever refresh
that table, so this module checks how stale/missing it is and surfaces a loud,
impossible-to-miss warning. It is called automatically at the end of every
`ingest_price.run()` — not something you have to remember to run separately.
"""
import datetime as dt

from sqlalchemy import select

from pipeline.db import foreign_flow, get_engine, init_schema, price_history
from pipeline.logging_config import get_logger

logger = get_logger("pipeline.alerts")

STALE_AFTER_DAYS = 3  # calendar days; covers weekends without over-alerting

ALERT_BANNER = "=" * 70


def check_foreign_flow_freshness(conn, tickers: list[str]) -> list[dict]:
    alerts = []
    today = dt.date.today()

    for code in tickers:
        latest_price = conn.execute(
            select(price_history.c.date)
            .where(price_history.c.stock_code == code)
            .order_by(price_history.c.date.desc())
            .limit(1)
        ).scalar()
        latest_flow = conn.execute(
            select(foreign_flow.c.date)
            .where(foreign_flow.c.stock_code == code)
            .order_by(foreign_flow.c.date.desc())
            .limit(1)
        ).scalar()

        if latest_price is None:
            continue  # no price data yet either; not a foreign_flow-specific issue

        if latest_flow is None:
            alerts.append(
                {
                    "stock_code": code,
                    "status": "missing",
                    "latest_price_date": latest_price,
                    "latest_flow_date": None,
                    "days_stale": None,
                }
            )
            continue

        days_stale = (today - latest_flow).days
        if days_stale > STALE_AFTER_DAYS:
            alerts.append(
                {
                    "stock_code": code,
                    "status": "stale",
                    "latest_price_date": latest_price,
                    "latest_flow_date": latest_flow,
                    "days_stale": days_stale,
                }
            )

    return alerts


def report(alerts: list[dict]) -> None:
    if not alerts:
        logger.info("foreign_flow up to date for all tracked stocks — no manual import needed.")
        return

    missing = [a["stock_code"] for a in alerts if a["status"] == "missing"]
    stale = [a for a in alerts if a["status"] == "stale"]

    logger.warning(ALERT_BANNER)
    logger.warning("MANUAL DATA REMINDER — foreign_flow perlu diimpor ulang")
    if missing:
        logger.warning("Belum pernah diimpor sama sekali (%d saham): %s", len(missing), ", ".join(missing))
    if stale:
        detail = ", ".join(f"{a['stock_code']} (terakhir {a['latest_flow_date']}, {a['days_stale']} hari lalu)" for a in stale)
        logger.warning("Sudah lewat %d hari sejak data terakhir (%d saham): %s", STALE_AFTER_DAYS, len(stale), detail)
    logger.warning("Cara update: unduh CSV dari idx.co.id lalu jalankan:")
    logger.warning("  python -m pipeline.import_foreign_flow path/to/file.csv")
    logger.warning(ALERT_BANNER)


def run(tickers: list[str] | None = None) -> list[dict]:
    from pipeline.tickers import SEED_TICKERS

    tickers = tickers or SEED_TICKERS
    engine = get_engine()
    init_schema(engine)
    with engine.connect() as conn:
        alerts = check_foreign_flow_freshness(conn, tickers)
    report(alerts)
    return alerts


if __name__ == "__main__":
    run()
