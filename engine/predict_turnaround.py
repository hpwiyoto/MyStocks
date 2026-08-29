"""Turnaround prediction engine: scores tickers CURRENTLY in bearish/
bottoming regime (the only candidates the model was trained on -- see
scripts/turnaround_labels.py) against the turnaround model, upserts into
the same `predictions` table as the swing model with model_version=
"turnaround_xgboost_v1".

Deliberately reuses engine.predict's table/upsert plumbing rather than a
separate schema: same shape of row (stock_code, date, model_version,
probability, decision, entry_price + nullable trade-structure columns),
just without a target_pct/stop_pct trade to derive stop_loss/take_profit/
risk_reward_ratio from -- this model's target is "reaches and holds a
regime", not a price level, so those three columns are left NULL for
turnaround rows (the app's UI treats them as "not applicable" rather than
missing data).

Usage:
    python -m engine.predict_turnaround
"""
from sqlalchemy import inspect, select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from engine.db import init_schema, predictions
from engine.model import build_feature_row, load_model_and_metadata, predict_probability
from engine.predict import price_on_date
from features.db import FEATURE_VERSION, feature_daily
from features.regime import BAD_REGIMES
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS

logger = get_logger("engine.predict_turnaround")

MODEL_VERSION = "turnaround_xgboost_v1"


def latest_feature_row(conn, code: str):
    row = conn.execute(
        select(feature_daily)
        .where(feature_daily.c.stock_code == code, feature_daily.c.feature_version == FEATURE_VERSION)
        .order_by(feature_daily.c.date.desc())
        .limit(1)
    ).mappings().first()
    return dict(row) if row else None


def upsert_prediction(conn, record: dict) -> None:
    stmt = mysql_insert(predictions).values([record])
    update_cols = {c: stmt.inserted[c] for c in record if c not in ("stock_code", "date", "model_version")}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine_db = get_engine()
    init_schema(engine_db)

    inspector = inspect(engine_db)
    missing_tables = [t for t in ("feature_daily", "price_history") if not inspector.has_table(t)]
    if missing_tables:
        logger.error("Table(s) %s don't exist yet.", missing_tables)
        return {"scored": [], "skipped": [], "not_candidate": [], "failures": [], "error": f"missing tables: {missing_tables}"}

    booster, meta = load_model_and_metadata(MODEL_VERSION)
    feature_cols = meta["feature_cols"]
    threshold = meta["walk_forward_validation"]["buy_threshold"]

    scored, skipped, not_candidate, failures = [], [], [], []

    for code in tickers:
        try:
            with engine_db.begin() as conn:
                feat_row = latest_feature_row(conn, code)
                if feat_row is None:
                    skipped.append(code)
                    continue

                if feat_row.get("regime") not in BAD_REGIMES:
                    # Not a candidate -- the model was only ever trained on
                    # rows starting in bearish/bottoming, scoring anything
                    # else would be extrapolating outside what it learned.
                    not_candidate.append(code)
                    continue

                entry_price = price_on_date(conn, code, feat_row["date"])
                if entry_price is None:
                    skipped.append(code)
                    continue

                X, missing = build_feature_row(feat_row, feature_cols)
                if missing:
                    logger.info("%s: %s NULL, predicting with the rest of the feature row anyway", code, missing)

                probability = predict_probability(booster, X)
                decision = "POTENSIAL" if probability >= threshold else "BELUM"

                record = {
                    "stock_code": code, "date": feat_row["date"], "model_version": MODEL_VERSION,
                    "probability": round(probability, 4), "decision": decision, "entry_price": entry_price,
                    "stop_loss_price": None, "take_profit_price": None, "risk_reward_ratio": None,
                }
                upsert_prediction(conn, record)
                scored.append(record)
                logger.info("%s (%s, regime=%s): prob=%.3f -> %s",
                            code, feat_row["date"], feat_row.get("regime"), probability, decision)
        except Exception as exc:
            logger.error("%s: turnaround prediction failed, skipping -- %s", code, exc)
            failures.append(code)

    logger.info("Done. Scored %d, not-candidate (not bearish/bottoming) %d, skipped %d, failed %d.",
                len(scored), len(not_candidate), len(skipped), len(failures))
    return {"scored": scored, "skipped": skipped, "not_candidate": not_candidate, "failures": failures}


if __name__ == "__main__":
    run()
