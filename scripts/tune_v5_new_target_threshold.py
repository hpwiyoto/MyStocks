"""Threshold re-tuning for Swing's new target definition (10%/5%/5 trading
days, adopted 2026-09-13 via scripts/search_swing_target.py -- see that
script's docstring for why the target itself changed).

BUY_THRESHOLD=0.65 was tuned specifically for the OLD 5%/2.5%/10d label.
scripts/search_swing_target.py deliberately used a threshold-agnostic
top-N%-of-predictions metric so the 20 target/horizon configs could be
compared fairly -- it does NOT tell us the right absolute probability
cutoff for the new label now that it's actually being adopted. This script
closes that gap: same pooled (trade-weighted, not fold-averaged --
scripts/tune_v5_extended.py's finding that fold-averaging can mislead)
walk-forward threshold sweep methodology as scripts/tune_v5.py, but wider
range (0.30->0.90) since a more extreme target can shift where the useful
range sits, and reports an estimated BUY signals/day figure alongside
precision/profit-factor/drawdown -- the same practical constraint that
got the OLD threshold moved from 0.65 down to 0.60 (0.65 meant sparse
enough signals that several consecutive days landed at zero, which the
user explicitly flagged; see engine/decision.py's docstring).

Keeps the CURRENT XGB_PARAMS (max_depth=3 etc, unchanged) rather than
re-running a hyperparameter grid search -- scripts/search_swing_target.py's
20-config result (the basis for adopting this target at all) was itself
produced with these exact hyperparameters, so re-picking hyperparameters
here would make the final shipped model inconsistent with the research
that justified the target change.

Usage:
    python -m scripts.tune_v5_new_target_threshold
"""
import json

import numpy as np

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import HORIZON, XGB_PARAMS, walk_forward_splits
from scripts.tune_v5 import _fold_data as fold_data
from scripts.tune_v5 import _load_panel as load_panel
from scripts.tune_v5 import threshold_sweep

logger = get_logger("scripts.tune_v5_new_target_threshold")

THRESHOLD_RANGE = np.arange(0.30, 0.91, 0.05)
RESULTS_JSON = "data/tune_v5_new_target_threshold_results.json"


def run():
    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    folds = fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    # Total pooled test-period trading days (across all folds combined) --
    # same denominator engine/decision.py's docstring uses for "average
    # signals/day across ~900 tickers" (n_trades already sums across every
    # ticker within that period, so dividing by calendar-of-trading-days,
    # not by ticker count, gives the right per-day figure).
    test_dates = set()
    for split in splits:
        mask = (dates >= split["test_start_date"]) & (dates <= split["test_end_date"])
        test_dates.update(dates[mask].tolist())
    total_test_days = len(test_dates)
    logger.info("Total pooled test-period trading days: %d", total_test_days)

    shipped_cfg = {k: XGB_PARAMS[k] for k in ("max_depth", "eta", "min_child_weight", "subsample", "colsample_bytree")}
    orig_range = threshold_sweep.__globals__["THRESHOLD_RANGE"]
    threshold_sweep.__globals__["THRESHOLD_RANGE"] = THRESHOLD_RANGE
    try:
        sweep = threshold_sweep(folds, shipped_cfg)
    finally:
        threshold_sweep.__globals__["THRESHOLD_RANGE"] = orig_range

    sweep["wilson_lb"] = [
        wilson_lower_bound(round(row.precision * row.n_trades), int(row.n_trades)) if row.n_trades else 0.0
        for row in sweep.itertuples()
    ]
    sweep["signals_per_day"] = sweep["n_trades"] / total_test_days

    logger.info("=" * 100)
    logger.info("FULL THRESHOLD SWEEP (pooled across %d folds):", len(folds))
    logger.info("=" * 100)
    logger.info("\n%s", sweep.to_string(index=False))

    sweep.to_json(RESULTS_JSON, orient="records", indent=2)
    logger.info("Saved %s", RESULTS_JSON)


if __name__ == "__main__":
    run()
