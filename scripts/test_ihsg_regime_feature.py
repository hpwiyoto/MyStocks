"""Does adding an IHSG-level (market-wide) trend feature help the Swing
model specifically during the folds that scripts/check_fold_drift.py found
to be weakest (fold 2, fold 3 -- both coinciding with IHSG declining, per
the user's own domain knowledge: IHSG bullish May 2025 -> 27 Jan 2026, then
sharply bearish through mid-2026)? The model already has
relative_strength_20d_pct (stock return MINUS IHSG's) but nothing capturing
IHSG's OWN absolute trend -- a stock outperforming IHSG by the same margin
means something very different in a rising market vs a falling one, and
the model currently can't tell those apart.

Same A/B methodology as scripts/test_foreign_flow_feature.py and
scripts/test_macd_zscore_feature.py: same walk-forward folds, same
hyperparameters, same everything -- ONLY the feature set differs between
the "baseline" and "with IHSG" runs, and both are reported PER FOLD (not
just averaged) since that's the whole point of this test.

RESULT (2026-09-07): makes things WORSE, badly, specifically in the two
folds it was meant to fix -- fold 2 precision@0.60 dropped 71.8%->52.6%,
fold 3 dropped 71.4%->24.6% (n_buy exploded 49->4398, i.e. the model started
firing BUY on almost everything). NOT adopted. Likely cause: these IHSG
columns have ZERO cross-sectional variance (identical value for all ~900
tickers on a given day) -- a tree can exploit that to effectively split by
calendar date rather than learn a genuine per-stock pattern, which looks
fine in-sample and collapses out-of-sample the moment the test fold's IHSG
regime differs from what training saw. The underlying premise (IHSG's
broad trend correlates with when the model struggles) was correct and
confirmed separately by checking IHSG's real return per fold; the fix
attempted here just doesn't work.

Usage:
    python -m scripts.test_ihsg_regime_feature
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
from scripts.check_fold_drift import LIVE_THRESHOLD

logger = get_logger("scripts.test_ihsg_regime_feature")

IHSG_SYMBOL = "^JKSE"


def _load_ihsg_features() -> pd.DataFrame:
    """IHSG's own absolute trend, computed directly off its close series --
    deliberately simple (rolling returns + SMA position), not the full
    per-stock regime classifier, since the hypothesis here is specifically
    about the BROAD market's direction, not a categorical regime label."""
    raw = fetch_history(IHSG_SYMBOL, period="5y").reset_index()
    raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    close = raw.set_index("date")["Close"].sort_index()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out = pd.DataFrame({
        "ihsg_return_20d": close.pct_change(20) * 100,
        "ihsg_return_50d": close.pct_change(50) * 100,
        "ihsg_above_sma50": (close > sma50).astype(float),
        "ihsg_sma20_above_sma50": (sma20 > sma50).astype(float),
    })
    out.index.name = "date"
    return out.reset_index()


def _train_eval_fold(df, feature_cols, split):
    train_mask = df["date"] <= split["train_embargo_end_date"]
    test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
    if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
        return None
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    prob = booster.predict(dtest)
    m65, tr65 = ml_metrics(y_test, prob, BUY_THRESHOLD), trading_metrics(y_test, prob, BUY_THRESHOLD)
    m60, tr60 = ml_metrics(y_test, prob, LIVE_THRESHOLD), trading_metrics(y_test, prob, LIVE_THRESHOLD)
    return {
        "precision@0.65": m65["precision"], "n_buy@0.65": tr65["n_trades"],
        "precision@0.60": m60["precision"], "n_buy@0.60": tr60["n_trades"],
        "roc_auc": m65["roc_auc"],
    }


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Baseline panel: %d rows, %d features", len(df), len(feature_cols))

    ihsg_feats = _load_ihsg_features()
    ihsg_cols = [c for c in ihsg_feats.columns if c != "date"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(ihsg_feats, on="date", how="left")
    logger.info("After merging IHSG features, null rate per column: %s",
                {c: f"{df[c].isna().mean():.1%}" for c in ihsg_cols})

    feature_cols_with_ihsg = feature_cols + ihsg_cols

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)

    logger.info("=" * 70)
    logger.info("A/B per fold: baseline (42 features) vs baseline + IHSG trend (%d features)", len(feature_cols_with_ihsg))
    logger.info("=" * 70)
    for fold_i, split in enumerate(splits):
        base = _train_eval_fold(df, feature_cols, split)
        with_ihsg = _train_eval_fold(df, feature_cols_with_ihsg, split)
        if base is None or with_ihsg is None:
            logger.info("Fold %d: skipped (insufficient data)", fold_i)
            continue
        logger.info(
            "Fold %d [%s -> %s]:\n"
            "  baseline : precision@0.65=%.3f (n=%d) precision@0.60=%.3f (n=%d) AUC=%.3f\n"
            "  +IHSG    : precision@0.65=%.3f (n=%d) precision@0.60=%.3f (n=%d) AUC=%.3f",
            fold_i, str(split["test_start_date"])[:10], str(split["test_end_date"])[:10],
            base["precision@0.65"], base["n_buy@0.65"], base["precision@0.60"], base["n_buy@0.60"], base["roc_auc"],
            with_ihsg["precision@0.65"], with_ihsg["n_buy@0.65"], with_ihsg["precision@0.60"], with_ihsg["n_buy@0.60"], with_ihsg["roc_auc"],
        )


if __name__ == "__main__":
    run()
