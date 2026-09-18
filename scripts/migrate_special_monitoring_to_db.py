"""One-off: seed the app's new `manual_ticker_exclusion` table (see
app/db.py) from the hardcoded ACTIVE_TICKERS snapshot in
scripts/special_monitoring_board.py, so switching load_suspended_tickers
over to reading the DB (admin-editable from the Admin page, no more
code-change + app-restart to refresh this list) doesn't lose the 160
tickers already transcribed from IDX's PDF.

After running this once, scripts/special_monitoring_board.py is reference/
history only -- the DB table is what the app actually reads. Safe to
re-run (upsert, not append -- see pipeline.db.upsert).

Usage:
    python -m scripts.migrate_special_monitoring_to_db
"""
from app.db import init_schema, manual_ticker_exclusion
from pipeline.db import get_engine, upsert
from pipeline.logging_config import get_logger
from scripts.special_monitoring_board import ACTIVE_TICKERS

logger = get_logger("scripts.migrate_special_monitoring_to_db")


def run():
    engine = get_engine()
    init_schema(engine)
    rows = [
        {"stock_code": code, "reason": "special_monitoring", "note": "Papan Pemantauan Khusus BEI"}
        for code in sorted(ACTIVE_TICKERS)
    ]
    with engine.begin() as conn:
        upsert(conn, manual_ticker_exclusion, rows, update_columns=["reason", "note"], index_elements=["stock_code"])
    logger.info("Seeded/updated %d ticker(s) into manual_ticker_exclusion", len(rows))


if __name__ == "__main__":
    run()
