"""Does a Lasso -> MLP -> XGBoost SERIES cascade (each model trained on
the residual/prediction of the one before it) beat plain XGBoost for
Swing's own binary classification task? Directly testing a claim the
user brought from Feng et al. (2025, Pacific-Basin Finance Journal,
"Global stock market forecasting: Insights from series and parallel
combination of machine learning models"): that a SERIES cascade of
Lasso-MLP-XGBoost outperforms individual models and a PARALLEL
combination, across 9 global stock INDICES over 2018-2024.

Why this needs its own test rather than being taken on faith (the whole
point of the user's question): that paper predicts continuous INDEX
returns with regression models over ~6 years across 9 large, liquid
indices -- a structurally different problem from classifying whether
ONE stock hits +5% before -2.5% within 10 days across ~900 IDX tickers.
Their own robustness section admits the Series-over-Parallel edge is
market-structure-dependent (strong in the US, "relatively neutral" in
Europe/Asia) -- so there is no a priori reason to assume it transfers
to IDX individual-stock classification without checking.

Adaptation to a classification target (their pipeline is built for
continuous returns, ours is binary): "Lasso" -> L1-penalized Logistic
Regression, "MLP" -> MLPClassifier, "XGBoost" unchanged. Each stage's
predicted probability on the TRAINING set is appended as an extra
feature for the next stage (a lightweight, in-sample-only approximation
of proper k-fold stacking -- imperfect for training, but the metric
that matters, TEST-set precision on truly held-out folds, is unaffected
by that approximation since no test information leaks backward).

Same walk-forward folds as scripts/train_v5.py, scripts/check_fold_drift.py.

RESULT (2026-09-08): NO -- the claim does not hold here. The cascade lost
to plain XGBoost in ALL 4 folds, by a wide and consistent margin:
  Fold 0: 88.1% -> 74.4% (-13.7pp)
  Fold 1: 74.1% -> 70.0% (-4.1pp)
  Fold 2: 71.8% -> 62.3% (-9.5pp)
  Fold 3: 71.4% -> 58.7% (-12.7pp)
AUC stayed roughly flat or slightly worse in every fold (discriminative
power essentially unchanged), while n_buy@0.60 exploded 3-4x in every
fold (151->422, 456->1321, 671->1254, 49->138) -- the cascade's
probability output is far LESS calibrated at the 0.60 cutoff, letting
many more low-quality signals cross the same bar without a matching
rise in true positives. NOT adopted. Consistent with this project's
running finding: techniques validated on INDEX-level regression tasks
(continuous returns, large diversified baskets, different noise
characteristics) do not automatically transfer to single-stock binary
classification on IDX -- the paper's own robustness section already
flagged the effect as market-structure-dependent even within its own
domain, and this is a second, larger step outside that domain
(individual emerging-market stocks, not developed-market indices).

Usage:
    python -m scripts.test_series_cascade
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
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

logger = get_logger("scripts.test_series_cascade")

LIVE_THRESHOLD = 0.60


def _baseline_xgb(X_train, y_train, X_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster.predict(dtest)


def _series_cascade(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train.fillna(0))
    Xs_test = scaler.transform(X_test.fillna(0))

    # Stage 1: "Lasso" -> L1-penalized logistic regression.
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000)
    lasso.fit(Xs_train, y_train)
    p1_train = lasso.predict_proba(Xs_train)[:, 1]
    p1_test = lasso.predict_proba(Xs_test)[:, 1]

    # Stage 2: MLP, features + stage-1 probability.
    mlp_train_X = np.column_stack([Xs_train, p1_train])
    mlp_test_X = np.column_stack([Xs_test, p1_test])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(mlp_train_X, y_train)
    p2_train = mlp.predict_proba(mlp_train_X)[:, 1]
    p2_test = mlp.predict_proba(mlp_test_X)[:, 1]

    # Stage 3: XGBoost, ORIGINAL (unscaled) features + both prior stages'
    # probabilities -- XGBoost doesn't need scaling, and giving it the
    # unscaled features alongside the cascade signals lets it use whichever
    # representation helps.
    xgb_train_X = X_train.copy()
    xgb_train_X["_lasso_prob"] = p1_train
    xgb_train_X["_mlp_prob"] = p2_train
    xgb_test_X = X_test.copy()
    xgb_test_X["_lasso_prob"] = p1_test
    xgb_test_X["_mlp_prob"] = p2_test

    dtrain = xgb.DMatrix(xgb_train_X, label=y_train)
    dtest = xgb.DMatrix(xgb_test_X)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster.predict(dtest)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)

    logger.info("=" * 70)
    logger.info("Per-fold: plain XGBoost (baseline) vs Lasso->MLP->XGBoost series cascade")
    logger.info("=" * 70)
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data)", fold_i)
            continue

        prob_baseline = _baseline_xgb(X_train, y_train, X_test)
        prob_cascade = _series_cascade(X_train, y_train, X_test)

        m_base = ml_metrics(y_test, prob_baseline, LIVE_THRESHOLD)
        tr_base = trading_metrics(y_test, prob_baseline, LIVE_THRESHOLD)
        m_casc = ml_metrics(y_test, prob_cascade, LIVE_THRESHOLD)
        tr_casc = trading_metrics(y_test, prob_cascade, LIVE_THRESHOLD)

        logger.info(
            "Fold %d [%s -> %s]:\n"
            "  baseline XGBoost    : precision@0.60=%.3f n_buy=%d AUC=%.3f\n"
            "  Lasso->MLP->XGBoost : precision@0.60=%.3f n_buy=%d AUC=%.3f",
            fold_i, str(split["test_start_date"])[:10], str(split["test_end_date"])[:10],
            m_base["precision"], tr_base["n_trades"], m_base["roc_auc"],
            m_casc["precision"], tr_casc["n_trades"], m_casc["roc_auc"],
        )


if __name__ == "__main__":
    run()
