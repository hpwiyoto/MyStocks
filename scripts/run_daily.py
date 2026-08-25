"""Fase 6: daily orchestration -- ingest -> features -> predict -> monitor.

Meant to be triggered once per day (after IDX market close) by the
scheduler service in docker-compose.yml. Each step is isolated: a failure
in one step is logged and does NOT prevent later steps from attempting to
run, since e.g. slightly stale features are still more useful than no
predictions at all.

Usage:
    python -m scripts.run_daily
"""
from pipeline.logging_config import get_logger

logger = get_logger("scripts.run_daily")


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

    try:
        from scripts.monitor import check_and_alert
        check_and_alert(ingest_failures=ingest_result.get("failures"))
    except Exception:
        logger.exception("monitor step raised unexpectedly")

    logger.info("=== Daily run: done ===")


if __name__ == "__main__":
    run()
