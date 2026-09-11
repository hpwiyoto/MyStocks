"""Does excluding BUY-zone signals that fire on an already-overextended,
deeply-overbought stock (a "blow-off top" -- buying right as a fast rally
is peaking, not while it's still building) improve Swing's real pooled
precision?

Prompted directly by a real case: LUCY, BUY 2026-09-04 at RSI 76.6 /
regime=overextended (already +144% over the prior month, +44% in just the
3 trading days before the signal), crashed -33.7% in the 5 trading days
after -- the stop (-2.5%) was blown through on day 1 (low -9.4%), not a
mild loss. The question: is "overextended + very high RSI at BUY time" a
GENUINE, generalizable risk flag across the whole 5-year history, or does
this one case just look that way in hindsight (the classic trap this
project's whole methodology exists to avoid -- see e.g. the grid-search
overfitting note in features/momentum_screener.py).

Same walk-forward methodology as scripts/tune_v5.py, POOLED (trade-
weighted) precision + Wilson LB as the metric, not fold-averaged -- per
scripts/tune_v5_extended.py's finding that fold-averaging can mislead.

Splits the BUY zone (prob>=BUY_THRESHOLD) into "blow-off top" (regime==
overextended AND rsi_14 > cutoff, swept at 70/75/80) vs everything else,
plus regime==overextended ALONE (no RSI condition) as a simpler
alternative filter.

Usage:
    python -m scripts.test_overextended_buy_filter
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import BUY_THRESHOLD, HORIZON, NUM_BOOST_ROUND, XGB_PARAMS, walk_forward_splits
from scripts.tune_v5 import _load_panel as load_panel

logger = get_logger("scripts.test_overextended_buy_filter")

RSI_CUTOFFS = [70, 75, 80]
EXTRA_COLS = ["rsi_14", "regime_overextended"]


def run_walk_forward_pooled(df, feature_cols, extra_cols, splits, xgb_params) -> pd.DataFrame:
    pooled = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("fold %d: skipped (insufficient data)", fold_i)
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        fold_extra = df.loc[test_mask, extra_cols].reset_index(drop=True)
        fold_df = pd.DataFrame({"fold": fold_i, "y_true": y_test, "prob": prob})
        pooled.append(pd.concat([fold_df, fold_extra], axis=1))
        logger.info("fold %d done: n_test=%d", fold_i, len(X_test))
    return pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()


def _stat(sub: pd.DataFrame) -> tuple[int, float, float]:
    n = len(sub)
    if n == 0:
        return 0, float("nan"), 0.0
    w = int(sub["y_true"].sum())
    return n, w / n, wilson_lower_bound(w, n)


def run():
    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    pooled = run_walk_forward_pooled(df, feature_cols, EXTRA_COLS, splits, XGB_PARAMS)
    logger.info("Pooled OOS rows (all folds, all probabilities): %d", len(pooled))

    buy_zone = pooled[pooled["prob"] >= BUY_THRESHOLD].copy()
    n0, wr0, lb0 = _stat(buy_zone)
    logger.info("=" * 90)
    logger.info("FULL BUY ZONE (prob>=%.2f, no filter): n=%d win_rate=%.4f wilson_lb=%.4f", BUY_THRESHOLD, n0, wr0, lb0)
    logger.info("=" * 90)

    overext_mask = buy_zone["regime_overextended"] == 1
    n1, wr1, lb1 = _stat(buy_zone[overext_mask])
    n1b, wr1b, lb1b = _stat(buy_zone[~overext_mask])
    logger.info("regime==overextended ALONE (no RSI condition):")
    logger.info("  overextended subset:     n=%-5d win_rate=%.4f wilson_lb=%.4f  (%.1f%% of BUY zone)",
                n1, wr1, lb1, n1 / n0 * 100 if n0 else 0)
    logger.info("  rest of BUY zone (kept): n=%-5d win_rate=%.4f wilson_lb=%.4f  (%+.4f vs unfiltered)",
                n1b, wr1b, lb1b, lb1b - lb0)

    logger.info("-" * 90)
    logger.info("'Blow-off top' = regime==overextended AND rsi_14 > cutoff:")
    for cutoff in RSI_CUTOFFS:
        blowoff = overext_mask & (buy_zone["rsi_14"] > cutoff)
        n2, wr2, lb2 = _stat(buy_zone[blowoff])
        n2b, wr2b, lb2b = _stat(buy_zone[~blowoff])
        logger.info("  RSI>%d: blow-off n=%-5d win_rate=%.4f wilson_lb=%.4f (%.1f%% of BUY zone) | "
                    "rest (kept) n=%-5d win_rate=%.4f wilson_lb=%.4f (%+.4f vs unfiltered)",
                    cutoff, n2, wr2, lb2, n2 / n0 * 100 if n0 else 0, n2b, wr2b, lb2b, lb2b - lb0)


if __name__ == "__main__":
    run()
