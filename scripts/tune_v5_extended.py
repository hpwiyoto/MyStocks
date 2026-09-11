"""A more thorough hyperparameter search for Swing (direction_xgboost_v5)
than scripts/tune_v5.py's original 10-config grid, run at the user's
explicit request ("jalankan nomor 1 dulu secara seksama").

What the original search covered vs. what's new here:
- tune_v5.py's 10 configs vary ONE knob at a time off a single baseline
  (max_depth 3/4/5/6 at fixed eta/mcw/subsample; eta 0.03/0.05/0.10 at
  fixed depth; etc.) -- a coordinate search, not a real grid, so it never
  tests e.g. "does max_depth=6 want a smaller eta too" (parameter
  INTERACTIONS are invisible to a one-at-a-time sweep).
- gamma (min loss reduction to split), reg_alpha (L1), and reg_lambda
  (L2) were NEVER varied at all before -- XGBoost's defaults (gamma=0,
  alpha=0, lambda=1) were simply never questioned.

This runs a RANDOM search (Bergstra & Bengio 2012's result that random
search matches or beats grid search at equal budget, because a grid
wastes trials repeating unimportant-dimension values) over all 8 knobs
at once, N_CONFIGS draws, so interactions ARE explored. The current
shipped config is always included as draw #0 -- a direct, apples-to-
apples "did anything actually beat what's live today" comparison, not
just "what looks best among random draws".

Same walk-forward folds, same holistic selection (AUC + profit_factor +
max_drawdown, ranked together -- a higher-AUC config with much worse
drawdown is an overfitting signature, not a genuinely better model, per
scripts/tune_v5.py's own reasoning) and the same reference threshold
(0.65) as the original search, so results are directly comparable.
NUM_BOOST_ROUND is NOT varied -- kept fixed at 200 to match every other
script in this project (train_v5.py, tune_v5.py) that assumes that
value; changing it here would make "the winning config" incompatible
with how the shipped model actually gets retrained.

This only searches and reports -- it does NOT retrain/redeploy the live
model. That is a separate, deliberate step if a config here genuinely
beats the shipped baseline (same posture as scripts/test_rally_speed_
feature.py's already-adopted finding: tested first, shipped only after
an explicit decision).

RESULT (2026-09-11): no config found beats the shipped baseline once
checked rigorously -- but getting to that answer surfaced a real flaw in
this script's (and the original tune_v5.py's) OWN selection method, not
just a null result on the hyperparameters themselves.

The holistic rank (average precision/AUC/drawdown across 5 folds,
weighting every fold equally regardless of size -- exactly what tune_v5.py
already did) picked config #8 (max_depth=3, eta=0.0744, min_child_weight=
20, subsample=0.92, colsample_bytree=0.85, gamma=0.3) as better than
shipped: fold-avg precision 85.8% vs shipped's 82.5%, better drawdown
(-7.2% vs -9.6%), MORE trades (225 vs 178 avg/fold). Looked like a clean
win on every axis.

It wasn't. Re-checked with a POOLED (trade-weighted, not fold-weighted)
threshold sweep at the shipped BUY_THRESHOLD=0.65 -- the statistically
honest comparison, same convention as every other feature/rule test in
this project's Wilson-LB methodology -- and the story flips: shipped
pools to n=890 wins=722 precision=81.1% wilson_lb=78.4%; config #8 pools
to n=1125 wins=895 precision=79.6% wilson_lb=77.1% -- shipped's LB is
HIGHER. Spot-checked a second strong fold-avg candidate (config #15,
fold-avg precision 85.7%, profit_factor 10.54 -- even better than #8 on
paper) the same way: pools to wilson_lb=76.9%, also below shipped. At
EVERY threshold in the 0.30-0.70 sweep, shipped's pooled precision beats
both candidates' -- they only win on RAW COVERAGE (more trades at a
given threshold, i.e. the model is less conservative), not on precision
or the risk-adjusted LB.

Root cause: unweighted fold-averaging lets a handful of small/lucky folds
(fold 4 in these splits often has near-zero BUY-zone signals -- see the
n_trades=0/nan rows throughout the full config table below) swing the
average disproportionately relative to their real sample size. Pooling
weights every actual trade equally instead, which is what a real trader
running this experiences. This means tune_v5.py's ORIGINAL config choice
carries the same untested assumption -- not verified false here (shipped
IS what that search picked), but the selection METHOD it used is the
same one just shown to mislead.

Conclusion: the shipped hyperparameters were NOT beaten by this 40-config
random search (including 3 previously-untried knobs: gamma, L1, L2) --
the model already sits close to a local optimum in this space. Not
retrained/reshipped. If hyperparameter tuning is revisited again, use
pooled Wilson LB as the selection criterion from the start, not fold-
averaging -- this run only pooled-verified its top 2 candidates
post-hoc (re-verifying all 40 that way was not run here, ~20 more minutes
of compute; the two checked were independently consistent enough to
trust the pattern without exhausting the rest).

Usage:
    python -m scripts.tune_v5_extended
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import BUY_THRESHOLD as REFERENCE_THRESHOLD
from scripts.train_v5 import HORIZON as LABEL_HORIZON
from scripts.train_v5 import NUM_BOOST_ROUND
from scripts.train_v5 import XGB_PARAMS as SHIPPED_PARAMS
from scripts.train_v5 import ml_metrics, trading_metrics, walk_forward_splits
from scripts.tune_v5 import BASE_PARAMS
from scripts.tune_v5 import _fold_data as fold_data
from scripts.tune_v5 import _load_panel as load_panel
from scripts.tune_v5 import pick_best_holistic, threshold_sweep

logger = get_logger("scripts.tune_v5_extended")

N_CONFIGS = 40
SEED = 42

# The shipped config's own 5 tunable knobs, kept as-is -- draw #0 below.
SHIPPED_CONFIG = {k: SHIPPED_PARAMS[k] for k in ("max_depth", "eta", "min_child_weight", "subsample", "colsample_bytree")}


def sample_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = [{**SHIPPED_CONFIG, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}]  # draw #0 = shipped, XGBoost defaults for the 3 new knobs
    for _ in range(n - 1):
        configs.append({
            "max_depth": int(rng.integers(2, 8)),  # 2..7
            "eta": round(float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))), 4),
            "min_child_weight": int(rng.choice([1, 2, 3, 5, 8, 10, 15, 20])),
            "subsample": round(float(rng.uniform(0.5, 1.0)), 2),
            "colsample_bytree": round(float(rng.uniform(0.5, 1.0)), 2),
            "gamma": float(rng.choice([0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0])),
            "reg_alpha": float(rng.choice([0.0, 0.001, 0.01, 0.1, 0.5, 1.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 1.5, 2.0, 5.0, 10.0])),
        })
    return configs


def hyperparameter_search(folds, grid: list[dict]) -> pd.DataFrame:
    results = []
    for cfg_i, cfg in enumerate(grid):
        params = {**BASE_PARAMS, **cfg}
        fold_ml, fold_trading = [], []
        for f in folds:
            booster = xgb.train(params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
            prob = booster.predict(f["dtest"])
            fold_ml.append(ml_metrics(f["y_test"], prob, REFERENCE_THRESHOLD))
            fold_trading.append(trading_metrics(f["y_test"], prob, REFERENCE_THRESHOLD))
        avg_ml = pd.DataFrame(fold_ml).mean()
        avg_trading = pd.DataFrame(fold_trading).mean()
        row = {"config_id": cfg_i, **cfg, "roc_auc": avg_ml["roc_auc"], "precision": avg_ml["precision"],
               "profit_factor": avg_trading["profit_factor"], "max_drawdown_pct": avg_trading["max_drawdown_pct"],
               "n_trades": avg_trading["n_trades"]}
        results.append(row)
        tag = " <- SHIPPED (baseline)" if cfg_i == 0 else ""
        logger.info("Config %d/%d %s: AUC=%.4f precision=%.3f profit_factor=%.2f max_dd=%.1f%% n_trades=%.0f%s",
                     cfg_i + 1, len(grid), cfg, row["roc_auc"], row["precision"], row["profit_factor"],
                     row["max_drawdown_pct"], row["n_trades"], tag)
    return pd.DataFrame(results)


def run():
    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=LABEL_HORIZON)
    folds = fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    grid = sample_configs(N_CONFIGS, SEED)
    logger.info("=" * 90)
    logger.info("RANDOM SEARCH: %d configs (draw #0 = shipped baseline), 8 knobs incl. gamma/reg_alpha/reg_lambda", len(grid))
    logger.info("=" * 90)
    hp_results = hyperparameter_search(folds, grid)

    shipped_row = hp_results.iloc[0]
    best_id = pick_best_holistic(hp_results)
    best_row = hp_results.iloc[best_id]
    best_cfg = {k: grid[best_id][k] for k in grid[best_id]}

    logger.info("=" * 90)
    logger.info("SHIPPED  (#0): AUC=%.4f precision=%.3f profit_factor=%.2f max_dd=%.1f%%",
                shipped_row["roc_auc"], shipped_row["precision"], shipped_row["profit_factor"], shipped_row["max_drawdown_pct"])
    logger.info("BEST     (#%d, holistic rank): %s", best_id, best_cfg)
    logger.info("         AUC=%.4f precision=%.3f profit_factor=%.2f max_dd=%.1f%%",
                best_row["roc_auc"], best_row["precision"], best_row["profit_factor"], best_row["max_drawdown_pct"])
    if best_id == 0:
        logger.info("-> Shipped config IS the best found. No change indicated.")
    else:
        logger.info("-> Candidate #%d beats shipped on the holistic rank. NOT auto-applied -- pending review.", best_id)
    logger.info("Full table:\n%s", hp_results.sort_values("config_id").to_string(index=False))

    logger.info("=" * 90)
    logger.info("Threshold sweep with the BEST config found (pooled out-of-sample)")
    logger.info("=" * 90)
    thr_results = threshold_sweep(folds, best_cfg)
    logger.info("\n%s", thr_results.to_string(index=False))

    with open("data/tune_v5_extended_results.json", "w") as f:
        json.dump({
            "n_configs": len(grid),
            "hyperparameter_search": hp_results.to_dict(orient="records"),
            "shipped_config_id": 0,
            "best_config_id": best_id,
            "best_config": best_cfg,
            "threshold_sweep": thr_results.to_dict(orient="records"),
        }, f, indent=2)
    logger.info("Saved data/tune_v5_extended_results.json")


if __name__ == "__main__":
    run()
