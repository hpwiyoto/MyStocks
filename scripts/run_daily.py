"""Fase 6: daily orchestration -- ingest -> features -> predict (swing) ->
monitor.

Meant to be triggered once per day (after IDX market close) by the
scheduler service in docker-compose.yml. Each step is isolated: a failure
in one step is logged and does NOT prevent later steps from attempting to
run, since e.g. slightly stale features are still more useful than no
predictions at all.

The Turnaround model's daily scoring step was removed after the
Turnaround page itself was retired (its top-2 real-world performance,
and even its proposed 3-month replacement's, came in well below Swing's
-- see scripts/compare_turnaround_v2_top2.py). engine/predict_turnaround.py
and its training pipeline (scripts/train_turnaround.py etc.) are kept in
the repo as historical/reusable research, just no longer wired into the
daily run.

Usage:
    python -m scripts.run_daily
"""
import datetime as dt
import os
import zoneinfo

from pipeline.logging_config import get_logger

logger = get_logger("scripts.run_daily")

WIB = zoneinfo.ZoneInfo("Asia/Jakarta")
# Read by scripts.scheduler_loop to decide whether a run is still due today.
# Stores a full WIB timestamp, not just a date -- confirmed real incident
# (2026-09-11): a bare date let a morning catch-up run (fetching only
# YESTERDAY's close, since the market hadn't opened yet) satisfy "already
# ran today" and permanently mask that same evening's real post-16:30 run,
# which would have fetched TODAY's actual close. scripts.scheduler_loop's
# run_is_due() compares this timestamp against the most recently PASSED
# 16:30 WIB slot, not just the calendar date.
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

    predict_result = {"failures": []}
    try:
        from engine.predict import run as predict_run
        predict_result = predict_run()
    except Exception:
        logger.exception("predict step raised unexpectedly")

    try:
        from scripts.monitor import check_and_alert
        # predict failures weren't wired in until a real incident (a missing
        # DB column failed every prediction, but check_and_alert only ever
        # looked at ingest_failures + price_history staleness -- both fine,
        # since ingest itself worked -- so this reported "all good" the
        # whole time predictions were completely broken).
        check_and_alert(
            ingest_failures=ingest_result.get("failures"),
            predict_failures=predict_result.get("failures"),
        )
    except Exception:
        logger.exception("monitor step raised unexpectedly")

    try:
        with open(LAST_RUN_MARKER, "w") as f:
            f.write(dt.datetime.now(WIB).isoformat())
    except OSError:
        logger.warning("Could not write last-run marker (non-fatal, only affects scheduler catch-up detection)")

    logger.info("=== Daily run: done ===")


if __name__ == "__main__":
    run()
