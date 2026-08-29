"""Prediction engine: load the trained Direction model, score every tracked
ticker's latest feature_daily row, run it through the decision engine, and
upsert the result into `predictions`.

Usage:
    python -m engine.predict
"""
from sqlalchemy import inspect, select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from engine.db import init_schema, predictions
from engine.decision import decide
from engine.model import build_feature_row, load_model_and_metadata, predict_probability
from features.db import FEATURE_VERSION, feature_daily
from pipeline.db import get_engine, price_history
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS

logger = get_logger("engine.predict")

MODEL_VERSION = "direction_xgboost_v5"


def latest_feature_row(conn, code: str):
    row = conn.execute(
        select(feature_daily)
        .where(feature_daily.c.stock_code == code, feature_daily.c.feature_version == FEATURE_VERSION)
        .order_by(feature_daily.c.date.desc())
        .limit(1)
    ).mappings().first()
    return dict(row) if row else None


def price_on_date(conn, code: str, date):
    row = conn.execute(
        select(price_history.c.close)
        .where(price_history.c.stock_code == code, price_history.c.date == date)
        .limit(1)
    ).first()
    return float(row[0]) if row else None


def upsert_prediction(conn, record: dict) -> None:
    stmt = mysql_insert(predictions).values([record])
    update_cols = {
        c: stmt.inserted[c]
        for c in record
        if c not in ("stock_code", "date", "model_version")
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine_db = get_engine()
    init_schema(engine_db)

    inspector = inspect(engine_db)
    missing_tables = [t for t in ("feature_daily", "price_history") if not inspector.has_table(t)]
    if missing_tables:
        logger.error(
            "Table(s) %s don't exist yet. Run `python -m pipeline.ingest_price` "
            "and `python -m features.build_features` first.", missing_tables,
        )
        return {"scored": [], "skipped": [], "failures": [], "error": f"missing tables: {missing_tables}"}

    booster, meta = load_model_and_metadata(MODEL_VERSION)
    feature_cols = meta["feature_cols"]
    base_rate = meta["base_rate"]
    target_pct = meta["target_pct"]
    stop_pct = meta["stop_pct"]

    scored = []
    skipped = []
    failures = []

    for code in tickers:
        try:
            with engine_db.begin() as conn:
                feat_row = latest_feature_row(conn, code)
                if feat_row is None:
                    logger.warning("%s: no feature_daily row (run features.build_features first), skipping", code)
                    skipped.append(code)
                    continue

                entry_price = price_on_date(conn, code, feat_row["date"])
                if entry_price is None:
                    logger.warning("%s: no price_history row for %s, skipping", code, feat_row["date"])
                    skipped.append(code)
                    continue

                X, missing = build_feature_row(feat_row, feature_cols)
                if missing:
                    # A genuinely NULL ratio (e.g. trailing_pe for a
                    # persistently loss-making company -- ~34% of rows,
                    # confirmed by checking the DB directly) is not a
                    # reason to drop the whole ticker: X already has NaN in
                    # `missing`'s columns, and XGBoost handles that natively.
                    # Logged for visibility, not treated as a skip.
                    logger.info("%s: %s NULL, predicting with the rest of the feature row anyway", code, missing)

                probability = predict_probability(booster, X)
                decision_result = decide(probability, base_rate, entry_price, target_pct, stop_pct)

                record = {
                    "stock_code": code,
                    "date": feat_row["date"],
                    "model_version": MODEL_VERSION,
                    "probability": round(probability, 4),
                    **decision_result,
                }
                upsert_prediction(conn, record)
                scored.append(record)
                logger.info(
                    "%s (%s): prob=%.3f -> %s (entry=%.2f sl=%.2f tp=%.2f)",
                    code, feat_row["date"], probability, decision_result["decision"],
                    entry_price, decision_result["stop_loss_price"], decision_result["take_profit_price"],
                )
        except Exception as exc:
            logger.error("%s: prediction failed, skipping — %s", code, exc)
            failures.append(code)

    logger.info("Done. Scored %d, skipped %d, failed %d.", len(scored), len(skipped), len(failures))
    return {"scored": scored, "skipped": skipped, "failures": failures}


if __name__ == "__main__":
    run()
