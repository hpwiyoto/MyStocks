"""Adds a POOLED (trade-weighted) walk-forward precision + Wilson LB to
the shipped Swing model's metadata, alongside the existing walk_forward_
validation.avg_ml_metrics.precision.

That existing number (82.5%) is an UNWEIGHTED AVERAGE across the 5 walk-
forward folds -- scripts/tune_v5_extended.py's research (2026-09-11)
found this aggregation can genuinely mislead: a near-empty fold (fold 4
in these splits often has ~0 BUY-zone signals) gets the same weight as a
fold with hundreds of real trades, letting it swing the average away
from what a trader actually pooling every real signal would experience.
Two strong-looking candidate hyperparameter configs that beat the
shipped model on THIS metric both lost to it once checked pooled -- the
shipped config itself was never re-verified pooled anywhere the app
actually displays a number, which is the gap this script closes.

Writes the result into models/direction_xgboost_v5_metadata.json under
walk_forward_validation.pooled -- read by app/pages/6_Info_Model.py.
Does NOT touch/replace avg_ml_metrics or avg_trading_metrics (kept as
the original training-time record, for a transparent before/after).

Usage:
    python -m scripts.add_pooled_walk_forward_metrics
"""
import json

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import BUY_THRESHOLD, HORIZON, XGB_PARAMS, walk_forward_splits
from scripts.tune_v5 import _fold_data as fold_data
from scripts.tune_v5 import _load_panel as load_panel
from scripts.tune_v5 import threshold_sweep

logger = get_logger("scripts.add_pooled_walk_forward_metrics")

METADATA_PATH = "models/direction_xgboost_v5_metadata.json"


def run():
    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    folds = fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    shipped_cfg = {k: XGB_PARAMS[k] for k in ("max_depth", "eta", "min_child_weight", "subsample", "colsample_bytree")}
    sweep = threshold_sweep(folds, shipped_cfg)
    match = sweep[sweep["threshold"].round(2) == round(BUY_THRESHOLD, 2)]
    if match.empty:
        logger.error("threshold %.2f not found in sweep -- aborting, metadata NOT touched", BUY_THRESHOLD)
        return
    row = match.iloc[0]
    n = int(row["n_trades"])
    precision = float(row["precision"])
    wins = round(precision * n)
    lb = wilson_lower_bound(wins, n)
    logger.info("Pooled @ threshold=%.2f: n=%d wins=%d precision=%.4f wilson_lb=%.4f", BUY_THRESHOLD, n, wins, precision, lb)

    with open(METADATA_PATH) as f:
        meta = json.load(f)
    meta["walk_forward_validation"]["pooled"] = {
        "note": ("Trade-weighted across all folds, NOT a simple average of each fold's own precision "
                 "(see avg_ml_metrics above, and scripts/tune_v5_extended.py's docstring for why that can "
                 "mislead). This is the honest 'if you followed every real BUY signal across the whole "
                 "test period' number."),
        "threshold": BUY_THRESHOLD, "n_trades": n, "wins": wins,
        "precision": round(precision, 4), "wilson_lb_95": round(lb, 4),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Updated %s", METADATA_PATH)


if __name__ == "__main__":
    run()
