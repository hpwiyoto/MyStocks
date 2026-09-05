"""Fase 6: daily orchestration -- ingest -> features -> predict (swing +
turnaround) -> monitor.

Meant to be triggered once per day (after IDX market close) by the
scheduler service in docker-compose.yml. Each step is isolated: a failure
in one step is logged and does NOT prevent later steps from attempting to
run, since e.g. slightly stale features are still more useful than no
predictions at all.

Usage:
    python -m scripts.run_daily
"""
import datetime as dt
import os
import zoneinfo

from pipeline.logging_config import get_logger

logger = get_logger("scripts.run_daily")

WIB = zoneinfo.ZoneInfo("Asia/Jakarta")
# Read by scripts.scheduler_loop on startup to decide whether a day (or
# several -- e.g. a laptop left off/asleep over a weekend, the exact scenario
# that surfaced this) was missed entirely and needs an immediate catch-up run
# rather than silently waiting for tomorrow's scheduled slot. WIB date, not
# server-local, to stay consistent with the scheduler's own WIB-based clock.
LAST_RUN_MARKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "last_daily_run.txt")


def run():
    logger.info("=== Daily run: start ===")

    ingest_result = {"failures": []}
    try:
        from pipeline.ingest_price import run as ingest_run
        ingest_result = ingest_run()
    except Exception:
        logger.exception("ingest_price step raised unexpectedly")

    try:
        from features.build_features import run as features_run
        features_run()
    except Exception:
        logger.exception("build_features step raised unexpectedly")

    try:
        from engine.predict import run as predict_run
        predict_run()
    except Exception:
        logger.exception("predict step raised unexpectedly")

    # Was missing entirely until found via a user report ("Turnaround
    # kosong") -- the scheduler ran ingest/features/swing-predict daily but
    # never re-scored the turnaround model, so its predictions table only
    # ever reflected whichever candidates were bearish/bottoming on
    # whatever date someone last ran engine.predict_turnaround by hand.
    try:
        from engine.predict_turnaround import run as predict_turnaround_run
        predict_turnaround_run()
    except Exception:
        logger.exception("predict_turnaround step raised unexpectedly")

    try:
        from scripts.monitor import check_and_alert
        check_and_alert(ingest_failures=ingest_result.get("failures"))
    except Exception:
        logger.exception("monitor step raised unexpectedly")

    try:
        with open(LAST_RUN_MARKER, "w") as f:
            f.write(dt.datetime.now(WIB).date().isoformat())
    except OSError:
        logger.warning("Could not write last-run marker (non-fatal, only affects scheduler catch-up detection)")

    logger.info("=== Daily run: done ===")


if __name__ == "__main__":
    run()
