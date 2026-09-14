"""Finishes the loose end flagged in scripts/tune_v5_variants_hyperparams.py:
t7_h10 (+0.85pp) and t5_h5 (+0.89pp) showed apparent hyperparameter
improvements in that script's 10-config grid, but -- unlike the default
config's max_depth=4/eta=0.03 finding (scripts/tune_v5_new_target_
hyperparams.py) and t10_h10's own finding, neither got a fine-grained
sweep around its winning point to confirm a smooth peak rather than an
isolated noisy spike. Direct user request ("lakukan dulu yang no 1") to
close that gap before any adoption decision.

Each winning config differs from the SAME max_depth=4/eta=0.05/
min_child_weight=1 "middle" config on exactly one axis, so each gets a
FINE sweep on THAT axis specifically, holding everything else fixed --
same diagnostic the default config's own eta=0.03 finding used:
  - t7_h10's winner (config #4 in the original 10-config grid) changed
    eta: 0.05 -> 0.03. Swept eta in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05].
  - t5_h5's winner (config #6) changed min_child_weight: 1 -> 5. Swept
    min_child_weight in [1, 3, 4, 5, 6, 7, 8, 10].

A smooth peak at the winning value (rising into it, falling away on both
sides) means the finding is real; an isolated spike with no consistent
neighbors means it was noise, the same distinction that separated the
default config's real finding from the slope-window/MA/regime-threshold
research's rejected ones.

RESULT (2026-09-14): split verdict -- t7_h10 CONFIRMED, t5_h5 NOT.

t7_h10's eta sweep traced a clean, interpretable peak at 0.03 -- 76.78%
(0.02) -> 77.65% (0.025) -> 78.10% (0.03, peak) -> 77.71% (0.035) ->
76.65% (0.04), rising smoothly into the peak and falling smoothly away
on both immediate neighbors, the same signature that confirmed the
default config's own finding. (0.05 ticks back up to 77.37% -- a minor
wiggle far from the peak, doesn't undermine the local peak shape around
0.03.) ADOPTED below.

t5_h5's min_child_weight sweep did NOT show a clean peak -- values
bounce within a narrow ~1.1pp band with no consistent direction: 78.50%
(1) -> 77.97% (3, a dip) -> 78.82% (4) -> 79.11% (5, nominal peak) ->
78.38% (6, a dip right after the peak) -> 78.68%/78.74%/78.65% (7/8/10,
flat). This reads as noise fluctuating around a roughly flat line, not
an interpretable single optimum -- min_child_weight also lacks eta's
clean monotonic-regularization interpretation, so a noisy curve here is
plausible even with no real underlying effect. NOT adopted -- t5_h5
keeps its original shared hyperparameters (LEGACY_HYPERPARAMS in
scripts/train_v5_variants.py).

Usage:
    python -m scripts.verify_variants_hyperparams_finesweep
"""
import json

import pandas as pd

from pipeline.logging_config import get_logger
from scripts.search_swing_target import build_labels
from scripts.train_v5 import FEATURES_PATH, MODEL_DIR, PRICES_PATH, prepare_panel, walk_forward_splits
from scripts.tune_v5_variants_hyperparams import evaluate_config_pooled, fold_data

logger = get_logger("scripts.verify_variants_hyperparams_finesweep")


def load_folds_for(target_pct, stop_pct, horizon, features, prices):
    labels = build_labels(prices, horizon, target_pct, stop_pct)
    df, feature_cols = prepare_panel(features, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
    return fold_data(df, feature_cols, splits)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)

    # --- t7_h10: fine eta sweep -------------------------------------------
    with open(f"{MODEL_DIR}/direction_xgboost_v5_t7_h10_metadata.json") as f:
        t7_h10_threshold = json.load(f)["walk_forward_validation"]["buy_threshold"]
    logger.info("#" * 90)
    logger.info("t7_h10 (target=7%%/stop=3.5%%/horizon=10d, threshold=%.2f): fine eta sweep", t7_h10_threshold)
    logger.info("#" * 90)
    folds_t7_h10 = load_folds_for(0.07, 0.035, 10, features, prices)
    logger.info("%d usable folds", len(folds_t7_h10))
    for eta in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
        cfg = {"max_depth": 4, "eta": eta, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8}
        m = evaluate_config_pooled(folds_t7_h10, cfg, t7_h10_threshold)
        logger.info("eta=%.3f: roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                     eta, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"])

    # --- t5_h5: fine min_child_weight sweep --------------------------------
    with open(f"{MODEL_DIR}/direction_xgboost_v5_t5_h5_metadata.json") as f:
        t5_h5_threshold = json.load(f)["walk_forward_validation"]["buy_threshold"]
    logger.info("#" * 90)
    logger.info("t5_h5 (target=5%%/stop=2.5%%/horizon=5d, threshold=%.2f): fine min_child_weight sweep", t5_h5_threshold)
    logger.info("#" * 90)
    folds_t5_h5 = load_folds_for(0.05, 0.025, 5, features, prices)
    logger.info("%d usable folds", len(folds_t5_h5))
    for mcw in [1, 3, 4, 5, 6, 7, 8, 10]:
        cfg = {"max_depth": 4, "eta": 0.05, "min_child_weight": mcw, "subsample": 0.8, "colsample_bytree": 0.8}
        m = evaluate_config_pooled(folds_t5_h5, cfg, t5_h5_threshold)
        logger.info("min_child_weight=%d: roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                     mcw, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"])


if __name__ == "__main__":
    run()
