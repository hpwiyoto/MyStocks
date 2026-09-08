"""Follow-up to scripts/check_fold_drift.py + test_ihsg_regime_feature.py,
inspired by a genuinely-applicable idea from a paper the user shared
(Zhao et al. 2026, MSIF-OEM -- an alpha-factor framework that clusters
samples by market operating mode and keeps a SEPARATE regressor per
regime, rather than feeding a raw market-index feature into one global
model). We already found that feeding IHSG's trend in as a training
FEATURE backfires badly (zero cross-sectional variance lets a tree split
by calendar date instead of learning a real pattern). This script tests
the right-sized version of their actual insight instead: does the
model's live BUY_THRESHOLD need to be regime-conditional -- i.e., does a
BUY signal issued while IHSG is currently declining have a worse hit
rate than one issued while IHSG is flat/rising, using the SAME trained
model's raw probability, no retraining at all?

This is a DECISION-LAYER question (engine/decision.py), not a training
question, so it can't fall into the overfitting trap that sank the raw
feature: IHSG's trend is looked up AT INFERENCE TIME from the already-
trained model's output, exactly like scripts/check_fold_drift.py's own
per-fold breakdown, never seen by the model during training.

Usage:
    python -m scripts.test_regime_conditional_threshold
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from pipeline.yfinance_source import fetch_history
from scripts.train_v5 import (
    BUY_THRESHOLD,
    FEATURES_PATH,
    NUM_BOOST_ROUND,
    PRICES_PATH,
    XGB_PARAMS,
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    trading_metrics,
    walk_forward_splits,
)
from scripts.search_momentum_rules import wilson_lower_bound

logger = get_logger("scripts.test_regime_conditional_threshold")

LIVE_THRESHOLD = 0.60
RAISED_THRESHOLD = 0.65  # engine/decision.py's OLD threshold, before it was lowered -- reused
                          # here as the candidate "stricter bar during a declining market" value,
                          # not a new number invented for this test.
IHSG_SYMBOL = "^JKSE"


def _load_ihsg_trailing_return() -> pd.DataFrame:
    """IHSG's trailing 20-trading-day return AS OF each date -- strictly
    backward-looking (no lookahead: uses only price data up to and
    including that date), unlike scripts/test_ihsg_regime_feature.py this
    is never fed to the model, just used here to split ALREADY-COMPUTED
    predictions into two buckets after the fact."""
    raw = fetch_history(IHSG_SYMBOL, period="5y").reset_index()
    raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    close = raw.set_index("date")["Close"].sort_index()
    ret_20d = (close / close.shift(20) - 1) * 100
    return ret_20d.rename("ihsg_ret_20d").reset_index()


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    ihsg = _load_ihsg_trailing_return()
    df = df.merge(ihsg, on="date", how="left")
    logger.info("IHSG trailing-return null rate: %.1f%%", df["ihsg_ret_20d"].isna().mean() * 100)

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)

    all_scored = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        test_rows = df.loc[test_mask]
        X_test, y_test = test_rows[feature_cols], test_rows["label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data)", fold_i)
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        scored = test_rows[["date", "ihsg_ret_20d"]].copy()
        scored["prob"] = prob
        scored["label"] = y_test
        scored["fold"] = fold_i
        all_scored.append(scored)

    pooled = pd.concat(all_scored, ignore_index=True)
    buy = pooled[pooled["prob"] >= LIVE_THRESHOLD].dropna(subset=["ihsg_ret_20d"])
    logger.info("=" * 70)
    logger.info("Pooled BUY signals (prob >= %.2f) across all folds: n=%d", LIVE_THRESHOLD, len(buy))
    logger.info("=" * 70)

    declining = buy[buy["ihsg_ret_20d"] < 0]
    not_declining = buy[buy["ihsg_ret_20d"] >= 0]

    def report(label, sub):
        n = len(sub)
        if n == 0:
            logger.info("[%s] n=0", label)
            return
        wins = int(sub["label"].sum())
        wr = wins / n
        logger.info("[%s] n=%d win_rate=%.4f wilson_lb=%.4f", label, n, wr, wilson_lower_bound(wins, n))

    report("ALL BUY signals (baseline, current live behavior)", buy)
    report("BUY issued while IHSG declining (ret_20d < 0)", declining)
    report("BUY issued while IHSG flat/rising (ret_20d >= 0)", not_declining)

    logger.info("-" * 70)
    logger.info("Counterfactual: raise bar to %.2f ONLY during IHSG decline, keep %.2f otherwise",
                RAISED_THRESHOLD, LIVE_THRESHOLD)
    counterfactual = pd.concat([
        not_declining,  # unchanged: still gated at 0.60
        declining[declining["prob"] >= RAISED_THRESHOLD],  # stricter bar only here
    ])
    report(f"Counterfactual combined signal set", counterfactual)
    logger.info("Counterfactual signal count vs baseline: %d vs %d (%.1f%% of original volume kept)",
                len(counterfactual), len(buy), len(counterfactual) / len(buy) * 100 if len(buy) else 0)


if __name__ == "__main__":
    run()
