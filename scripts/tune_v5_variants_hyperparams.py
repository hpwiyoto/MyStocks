"""Do the 4 runner-up Swing configs (engine.swing_configs -- 7%/5d,
10%/10d, 7%/10d, 5%/5d) need their OWN hyperparameters, the way the
default config turned out to (scripts/tune_v5_new_target_hyperparams.py
found max_depth=4/eta=0.03 beats the shipped max_depth=3/eta=0.05 by
+2.63pp Wilson LB there)? Direct user follow-up: "ya tolong cek" to my
own offer to check this after reporting that finding.

scripts/train_v5_variants.py trained all 4 with hyperparameters
"kept identical to what search_swing_target.py itself used" -- i.e.
never independently tuned for THEIR OWN labels either, same gap the
default config had. This applies scripts/tune_v5.py's 10-config GRID to
each of the 4, using search_swing_target.build_labels (already
parameterized by target/stop/horizon) to build each config's own label,
and each config's OWN shipped threshold (read from its metadata) as the
fixed evaluation point -- same pooled (not fold-averaged) methodology as
the default config's check.

RESULT (2026-09-14): 3 of 4 sibling configs show an apparent improvement,
1 doesn't -- magnitudes vary enough that they carry different confidence:
  - t7_h5 (7%/-3.5%/5d): shipped (max_depth=3/eta=0.05) is ALREADY the
    pooled winner, no change found.
  - t10_h10 (10%/-5%/10d): max_depth=4/eta=0.05/min_child_weight=5 beats
    shipped by +2.19pp Wilson LB (72.54% vs 70.35%) -- comparable
    magnitude to the default config's own verified +2.63pp finding
    (scripts/tune_v5_new_target_hyperparams.py), high confidence.
  - t7_h10 (7%/-3.5%/10d): max_depth=4/eta=0.03 beats shipped by only
    +0.85pp (78.10% vs 77.24%).
  - t5_h5 (5%/-2.5%/5d): max_depth=4/eta=0.05/min_child_weight=5 beats
    shipped by only +0.89pp (79.11% vs 78.22%).

The t7_h10/t5_h5 gaps (+0.85pp, +0.89pp) are the SAME magnitude as the
slope-window and MA "wins" that turned out to be sampling noise once
combined-rechecked (scripts/test_slope_windows.py, scripts/test_ma_
windows.py both saw 0.65-1.42pp individual "improvements" that reversed
under a robustness check) -- NOT verified further here (unlike the
default config's eta=0.03 finding, which got its own fine-grained sweep
confirming a smooth peak, not an isolated spike). Treat t10_h10's
finding with the same confidence as the default's; treat t7_h10's and
t5_h5's as unconfirmed leads, not established results, until they get
the same fine-sweep verification before any adoption decision.

Usage:
    python -m scripts.tune_v5_variants_hyperparams
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.search_swing_target import build_labels
from scripts.train_v5 import FEATURES_PATH, MODEL_DIR, NUM_BOOST_ROUND, PRICES_PATH, ml_metrics, prepare_panel, walk_forward_splits
from scripts.tune_v5 import BASE_PARAMS, GRID

logger = get_logger("scripts.tune_v5_variants_hyperparams")

# (target_pct, stop_pct, horizon, model_version)
VARIANTS = [
    (0.07, 0.035, 5, "direction_xgboost_v5_t7_h5"),
    (0.10, 0.050, 10, "direction_xgboost_v5_t10_h10"),
    (0.07, 0.035, 10, "direction_xgboost_v5_t7_h10"),
    (0.05, 0.025, 5, "direction_xgboost_v5_t5_h5"),
]
SHIPPED_CFG = {"max_depth": 3, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8}


def fold_data(df, feature_cols, splits):
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


def evaluate_config_pooled(folds, cfg, threshold):
    params = {**BASE_PARAMS, **cfg}
    pooled_y, pooled_prob = [], []
    for f in folds:
        booster = xgb.train(params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(f["dtest"])
        pooled_y.append(f["y_test"])
        pooled_prob.append(prob)
    y, prob = np.concatenate(pooled_y), np.concatenate(pooled_prob)
    roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"]
    buy_mask = prob >= threshold
    n = int(buy_mask.sum())
    wins = int(y[buy_mask].sum()) if n else 0
    precision = wins / n if n else float("nan")
    lb = wilson_lower_bound(wins, n) if n else 0.0
    return {"roc_auc": roc_auc, "n": n, "wins": wins, "precision": precision, "wilson_lb": lb}


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    all_results = []
    for target_pct, stop_pct, horizon, model_version in VARIANTS:
        with open(f"{MODEL_DIR}/{model_version}_metadata.json") as f:
            meta = json.load(f)
        threshold = meta["walk_forward_validation"]["buy_threshold"]

        logger.info("#" * 100)
        logger.info("VARIANT %s: target=%.1f%% stop=%.2f%% horizon=%dd threshold=%.2f",
                     model_version, target_pct * 100, stop_pct * 100, horizon, threshold)
        logger.info("#" * 100)

        labels = build_labels(prices, horizon, target_pct, stop_pct)
        df, feature_cols = prepare_panel(features, labels)
        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
        folds = fold_data(df, feature_cols, splits)
        logger.info("%d usable folds", len(folds))

        variant_results = []
        for cfg_i, cfg in enumerate(GRID):
            m = evaluate_config_pooled(folds, cfg, threshold)
            is_shipped = cfg == SHIPPED_CFG
            variant_results.append({"model_version": model_version, "config_id": cfg_i, **cfg, "is_shipped": is_shipped, **m})
            logger.info("[%s] config %d/%d %s%s: roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                         model_version, cfg_i + 1, len(GRID), cfg, " [SHIPPED]" if is_shipped else "",
                         m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"])

        vdf = pd.DataFrame(variant_results).sort_values("wilson_lb", ascending=False)
        best = vdf.iloc[0]
        shipped_lb = vdf[vdf["is_shipped"]]["wilson_lb"].iloc[0]
        if best["is_shipped"]:
            logger.info("[%s] Shipped config is ALREADY the pooled winner -- no change needed.", model_version)
        else:
            logger.info("[%s] Config #%d beats shipped by %+.4f Wilson LB (max_depth=%d, eta=%.2f) -- candidate for adoption.",
                         model_version, int(best["config_id"]), best["wilson_lb"] - shipped_lb, best["max_depth"], best["eta"])
        all_results.extend(variant_results)

    rdf = pd.DataFrame(all_results)
    rdf.to_csv("data/tune_v5_variants_hyperparams_results.csv", index=False)
    logger.info("=" * 100)
    logger.info("Saved data/tune_v5_variants_hyperparams_results.csv")
    logger.info("=" * 100)
    logger.info("SUMMARY -- best config per variant:")
    for model_version in rdf["model_version"].unique():
        sub = rdf[rdf["model_version"] == model_version].sort_values("wilson_lb", ascending=False)
        best = sub.iloc[0]
        shipped_lb = sub[sub["is_shipped"]]["wilson_lb"].iloc[0]
        logger.info("  %s: best=#%d (max_depth=%d eta=%.2f) wilson_lb=%.4f vs shipped=%.4f (%+.4f)",
                     model_version, int(best["config_id"]), best["max_depth"], best["eta"],
                     best["wilson_lb"], shipped_lb, best["wilson_lb"] - shipped_lb)


if __name__ == "__main__":
    run()
