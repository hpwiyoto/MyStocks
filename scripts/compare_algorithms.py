"""Is XGBoost actually the best algorithm for the Swing model, or would a
different one do better on the CURRENT (v5, 44-feature, rally-speed-
included) dataset and label?

LightGBM was compared once before (Fase 3, notebooks/fase3_ml_research
.ipynb) but against the OLD v1 feature set/label -- stale relative to
today's feature set (rally-speed features landed 2026-09-08, several
feature rounds after that comparison). This reruns LightGBM plus four
algorithms never tried on this project's data before (Random Forest,
Logistic Regression, Linear SVM, CatBoost), on the EXACT SAME
walk-forward folds/features/label as the shipped model -- same
build_panel_labels/prepare_panel/walk_forward_splits imported directly
from scripts.train_v5, not re-derived, so there is no risk of a subtle
methodology drift making the comparison unfair.

Primary comparison metric: ROC-AUC (threshold-independent, so it's
comparable across algorithms whose probability calibration differs
wildly -- a raw SVM decision_function score and a calibrated XGBoost
probability aren't on the same scale, but their RANKING quality is
directly comparable via AUC). Precision/recall at the shipped threshold
(0.65) is reported too for context, but that threshold was tuned FOR
XGBoost specifically (scripts/tune_v5.py) -- it is not a fair "each
algorithm's own best operating point," which would need its own sweep per
algorithm (out of scope here; only worth doing for whichever algorithm,
if any, actually beats XGBoost on AUC first).

Hyperparameters: the three boosting algorithms (XGBoost/LightGBM/
CatBoost) use matching depth/learning-rate/subsample settings, since
boosting hyperparameters transfer directly between them and this keeps
that three-way comparison apples-to-apples. Random Forest, Logistic
Regression and Linear SVM are fundamentally different model families --
forcing XGBoost's shallow depth=3 onto Random Forest would badly undersell
it (RF's whole design leans on many deeper, decorrelated trees, not
shallow boosted ones), so each gets its own reasonable, NOT exhaustively
tuned, off-the-shelf config. This is a "does a different algorithm show
enough promise to be worth real tuning investment" pass, not a final
verdict on any algorithm's ceiling -- same first-pass-then-invest-if-
promising principle as every other empirical test in this project.

sklearn models (Random Forest, Logistic Regression, Linear SVM) don't
accept NaN natively, unlike XGBoost/LightGBM/CatBoost which all handle
missing values internally -- median imputation (fit on TRAIN fold only,
to avoid leakage) is applied just for those three. Logistic Regression
and Linear SVM also get StandardScaler (fit on train only) since they're
scale-sensitive; Random Forest doesn't need it (scale-invariant).

SVM: a literal RBF-kernel SVC does not scale to ~600k-row folds (O(n^2)
to O(n^3) -- would take hours to days per fold). LinearSVC is used
instead (near-linear scaling), a deliberate, disclosed substitution, not
a silent shortcut. LinearSVC has no predict_proba; its decision_function()
(a ranking score, not a calibrated probability) is enough for ROC-AUC,
which only needs relative ordering.

Usage:
    python -m scripts.compare_algorithms
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    BUY_THRESHOLD,
    FEATURES_PATH,
    NUM_BOOST_ROUND,
    PRICES_PATH,
    XGB_PARAMS,
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)

logger = get_logger("scripts.compare_algorithms")

# Matches XGBoost's own tuned depth/eta/subsample (scripts/tune_v5.py) --
# apples-to-apples for the three boosting algorithms specifically.
LGBM_PARAMS = {
    "objective": "binary", "max_depth": 3, "learning_rate": 0.05,
    "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
    "verbosity": -1, "seed": 42,
}
CATBOOST_PARAMS = dict(
    iterations=NUM_BOOST_ROUND, depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bylevel=0.8, min_data_in_leaf=1,
    verbose=False, random_seed=42, allow_writing_files=False,
)
# Not depth-matched to the boosters -- see module docstring on why RF gets
# its own reasonable config instead of XGBoost's shallow depth=3.
RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42)
LOGREG_PARAMS = dict(max_iter=1000, random_state=42)
LINEAR_SVM_PARAMS = dict(random_state=42, max_iter=2000, dual="auto")

ALGORITHMS = ["xgboost", "lightgbm", "catboost", "random_forest", "logistic_regression", "linear_svm"]


def _impute_scale(X_train: pd.DataFrame, X_test: pd.DataFrame, scale: bool):
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    if not scale:
        return X_train_i, X_test_i
    scaler = StandardScaler()
    return scaler.fit_transform(X_train_i), scaler.transform(X_test_i)


def score_fold(algo: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Returns a per-row score for X_test -- a calibrated probability for
    every algorithm except linear_svm (decision_function, ranking-only),
    which is fine since ROC-AUC (the primary metric here) only needs
    relative ordering, not calibration."""
    if algo == "xgboost":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        return booster.predict(dtest)

    if algo == "lightgbm":
        import lightgbm as lgb
        dtrain = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        return booster.predict(X_test)

    if algo == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    if algo == "random_forest":
        X_train_i, X_test_i = _impute_scale(X_train, X_test, scale=False)
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train_i, y_train)
        return model.predict_proba(X_test_i)[:, 1]

    if algo == "logistic_regression":
        X_train_s, X_test_s = _impute_scale(X_train, X_test, scale=True)
        model = LogisticRegression(**LOGREG_PARAMS)
        model.fit(X_train_s, y_train)
        return model.predict_proba(X_test_s)[:, 1]

    if algo == "linear_svm":
        X_train_s, X_test_s = _impute_scale(X_train, X_test, scale=True)
        model = LinearSVC(**LINEAR_SVM_PARAMS)
        model.fit(X_train_s, y_train)
        return model.decision_function(X_test_s)

    raise ValueError(f"unknown algorithm: {algo}")


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features", len(df), len(feature_cols))

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)
    logger.info("%d walk-forward folds generated", len(splits))

    results = {algo: [] for algo in ALGORITHMS}
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()

        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data or single-class train set)", fold_i)
            continue

        logger.info("Fold %d: train=%d test=%d train_pos_rate=%.3f test_pos_rate=%.3f",
                    fold_i, len(X_train), len(X_test), y_train.mean(), y_test.mean())

        for algo in ALGORITHMS:
            t0 = time.time()
            score = score_fold(algo, X_train, y_train, X_test)
            elapsed = time.time() - t0
            # Only the calibrated-probability algorithms get a meaningful
            # precision/recall at the SHIPPED threshold (0.65) -- linear_svm's
            # decision_function lives on a totally different scale, so its
            # threshold-based numbers would be meaningless noise, not a real
            # comparison point. ROC-AUC is unaffected either way.
            if algo == "linear_svm":
                auc_only = ml_metrics(y_test, score, threshold=0.0)  # threshold irrelevant to roc_auc
                m = {"precision": float("nan"), "recall": float("nan"), "roc_auc": auc_only["roc_auc"], "n_buy_signals": None}
            else:
                m = ml_metrics(y_test, score, BUY_THRESHOLD)
            m["elapsed_sec"] = round(elapsed, 1)
            results[algo].append(m)
            logger.info("  [%s] roc_auc=%.4f precision=%s recall=%s (%.1fs)",
                        algo, m["roc_auc"],
                        f"{m['precision']:.3f}" if m["precision"] == m["precision"] else "n/a",
                        f"{m['recall']:.3f}" if m["recall"] == m["recall"] else "n/a",
                        elapsed)

    logger.info("=" * 70)
    logger.info("SUMMARY (average across folds)")
    logger.info("=" * 70)
    summary_rows = []
    for algo in ALGORITHMS:
        if not results[algo]:
            continue
        avg = pd.DataFrame(results[algo]).mean(numeric_only=True)
        summary_rows.append({
            "algorithm": algo, "n_folds": len(results[algo]),
            "avg_roc_auc": round(float(avg["roc_auc"]), 4),
            "avg_precision@0.65": round(float(avg["precision"]), 4) if avg["precision"] == avg["precision"] else None,
            "avg_recall@0.65": round(float(avg["recall"]), 4) if avg["recall"] == avg["recall"] else None,
            "avg_train_time_sec": round(float(avg["elapsed_sec"]), 1),
        })
    summary = pd.DataFrame(summary_rows).sort_values("avg_roc_auc", ascending=False)
    pd.set_option("display.width", 200)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
