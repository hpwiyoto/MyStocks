"""Trains the 4 RUNNER-UP configs from scripts/search_swing_target.py's
20-config search as full sibling models, alongside the already-shipped
winner (target=10%/stop=5%/horizon=5d, trained in place as
models/direction_xgboost_v5.json by scripts/train_v5.py).

Direct user request: a toggle in the app to switch between the top-5
configs by top5_lift, not just the single winner. That means each of the
other 4 needs to be a real, independently walk-forward-validated and
threshold-tuned model -- not the winner's model reused with different
labels, and not the winner's threshold blindly copied (already learned
from search_swing_target.py's own methodology note: a new label needs its
OWN threshold sweep, the winner's 0.60 has no reason to be right for a
different target/horizon).

Top 5 by top5_lift (from data/search_swing_target_results.csv):
  1. target=10% stop=5%  horizon=5d  lift=+0.228  (shipped separately, NOT retrained here)
  2. target=7%  stop=3.5% horizon=5d  lift=+0.203
  3. target=10% stop=5%  horizon=10d lift=+0.189
  4. target=7%  stop=3.5% horizon=10d lift=+0.184
  5. target=5%  stop=2.5% horizon=5d  lift=+0.176

Same XGB_PARAMS/NUM_BOOST_ROUND as scripts/train_v5.py for all 5 -- kept
identical to what search_swing_target.py itself used, so every variant
stays consistent with the research that ranked it, no re-tuned
hyperparameters introducing a second uncontrolled variable per config.

Threshold picked automatically per config (not eyeballed per curve, so
this stays reproducible across configs): among thresholds in the pooled
walk-forward sweep with an estimated >=1.0 BUY signals/day (the same
usable-frequency floor scripts/tune_v5_new_target_threshold.py used to
justify picking 0.60 for the winner over sparser 0.65+), pick the one
with the highest Wilson LB; if none clears that floor, relax to >=0.5,
then >=0.2, then fall back to the single best Wilson LB regardless of
frequency.

Usage:
    python -m scripts.train_v5_variants
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.search_swing_target import build_labels
from scripts.train_v5 import (
    FEATURES_PATH,
    MODEL_DIR,
    NUM_BOOST_ROUND,
    PRICES_PATH,
    XGB_PARAMS,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)

logger = get_logger("scripts.train_v5_variants")

THRESHOLD_RANGE = np.arange(0.30, 0.91, 0.05)
MIN_SIGNALS_PER_DAY_TIERS = [1.0, 0.5, 0.2, 0.0]

# (target_pct, stop_pct, horizon, model_version, rank/lift -- for the notes field)
VARIANTS = [
    (0.07, 0.035, 5, "direction_xgboost_v5_t7_h5", 2, 0.202931),
    (0.10, 0.050, 10, "direction_xgboost_v5_t10_h10", 3, 0.189419),
    (0.07, 0.035, 10, "direction_xgboost_v5_t7_h10", 4, 0.183642),
    (0.05, 0.025, 5, "direction_xgboost_v5_t5_h5", 5, 0.176246),
]


def _fold_data(df, feature_cols, splits):
    folds = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("fold %d: skipped (insufficient data)", fold_i)
            continue
        folds.append({"fold": fold_i, "dtrain": xgb.DMatrix(X_train, label=y_train),
                       "dtest": xgb.DMatrix(X_test), "y_test": y_test})
    return folds


def _threshold_sweep_pooled(folds, total_test_days):
    pooled_y, pooled_prob = [], []
    for f in folds:
        booster = xgb.train(XGB_PARAMS, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(f["dtest"])
        pooled_y.append(f["y_test"])
        pooled_prob.append(prob)
    y, prob = np.concatenate(pooled_y), np.concatenate(pooled_prob)

    rows = []
    for t in THRESHOLD_RANGE:
        m = ml_metrics(y, prob, t)
        taken = prob >= t
        n = int(taken.sum())
        wins = int(y[taken].sum()) if n else 0
        precision = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else 0.0
        rows.append({
            "threshold": round(float(t), 2), "precision": precision, "n_trades": n, "wins": wins,
            "wilson_lb": lb, "roc_auc": m["roc_auc"], "signals_per_day": n / total_test_days,
        })
    return pd.DataFrame(rows)


def _pick_threshold(sweep: pd.DataFrame) -> dict:
    for floor in MIN_SIGNALS_PER_DAY_TIERS:
        candidates = sweep[sweep["signals_per_day"] >= floor]
        if not candidates.empty:
            best = candidates.loc[candidates["wilson_lb"].idxmax()]
            return best.to_dict()
    return sweep.iloc[0].to_dict()


def train_variant(target_pct, stop_pct, horizon, model_version, features, prices):
    logger.info("=" * 100)
    logger.info("VARIANT %s: target=%.1f%% stop=%.2f%% horizon=%dd", model_version, target_pct * 100, stop_pct * 100, horizon)
    logger.info("=" * 100)

    labels = build_labels(prices, horizon, target_pct, stop_pct)
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features", len(df), len(feature_cols))

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
    folds = _fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    test_dates = set()
    for split in splits:
        mask = (dates >= split["test_start_date"]) & (dates <= split["test_end_date"])
        test_dates.update(dates[mask].tolist())
    total_test_days = len(test_dates)

    sweep = _threshold_sweep_pooled(folds, total_test_days)
    logger.info("Threshold sweep:\n%s", sweep.to_string(index=False))
    chosen = _pick_threshold(sweep)
    buy_threshold = float(chosen["threshold"])
    logger.info("Chosen threshold=%.2f: precision=%.4f wilson_lb=%.4f n_trades=%d signals/day=%.2f",
                buy_threshold, chosen["precision"], chosen["wilson_lb"], chosen["n_trades"], chosen["signals_per_day"])

    logger.info("Training final model on all %d rows...", len(df))
    dall = xgb.DMatrix(df[feature_cols], label=df["label"])
    final_booster = xgb.train(XGB_PARAMS, dall, num_boost_round=NUM_BOOST_ROUND)

    base_rate = float(df["label"].mean())
    model_path = f"{MODEL_DIR}/{model_version}.json"
    meta_path = f"{MODEL_DIR}/{model_version}_metadata.json"
    final_booster.save_model(model_path)

    metadata = {
        "model_version": model_version,
        "trained_at": dt.date.today().isoformat(),
        "feature_cols": feature_cols,
        "base_rate": base_rate,
        "horizon_days": horizon,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "n_training_rows": len(df),
        "tickers": sorted(df["stock_code"].unique().tolist()),
        "hyperparameters": {**XGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND},
        "walk_forward_validation": {
            "n_folds": len(folds),
            "buy_threshold": buy_threshold,
            "avg_ml_metrics": {"precision": chosen["precision"], "roc_auc": chosen["roc_auc"]},
            "pooled": {
                "note": ("Trade-weighted across all folds -- same methodology as "
                         "models/direction_xgboost_v5_metadata.json's pooled block."),
                "threshold": buy_threshold, "n_trades": int(chosen["n_trades"]), "wins": int(chosen["wins"]),
                "precision": round(float(chosen["precision"]), 4), "wilson_lb_95": round(float(chosen["wilson_lb"]), 4),
            },
        },
        "notes": (
            f"One of 4 runner-up configs (rank #{VARIANTS_BY_VERSION[model_version][4]} of 20 by top5_lift "
            f"={VARIANTS_BY_VERSION[model_version][5]:+.4f}) trained alongside the shipped winner "
            "(direction_xgboost_v5, target=10%/stop=5%/horizon=5d) from scripts/search_swing_target.py's "
            "search -- direct user request for a toggle between the top-5 configs, not just the single "
            "winner. Same feature set/hyperparameters as the winner; threshold independently re-tuned for "
            "THIS target via scripts/train_v5_variants.py's own pooled walk-forward sweep, picked to keep "
            "an estimated >=1 BUY signal/day where the sweep allows it."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved %s + %s", model_path, meta_path)
    return metadata


VARIANTS_BY_VERSION = {v[3]: v for v in VARIANTS}


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    for target_pct, stop_pct, horizon, model_version, rank, lift in VARIANTS:
        train_variant(target_pct, stop_pct, horizon, model_version, features, prices)

    logger.info("=" * 100)
    logger.info("All %d variants trained.", len(VARIANTS))


if __name__ == "__main__":
    run()
