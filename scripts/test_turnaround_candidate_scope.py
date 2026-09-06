"""One-off empirical test of a candidate-scope idea raised by the user:
instead of admitting BOTH bearish and bottoming as turnaround candidates,
what if only "bottoming" is admitted -- the regime that already encodes "RSI
recovering from oversold, downtrend losing steam" (features/regime.py),
i.e. a stock showing a small early sign of turning, vs "bearish" which is a
straight downtrend with no recovery signal at all? Hypothesis: narrowing to
bottoming-only (a) is a cleaner/more learnable signal (higher precision/
ROC-AUC) and (b) resolves faster (shorter empirical days-to-turnaround),
since the stock is already closer to the finish line when picked up.

Tests THREE variants against the identical walk-forward pipeline used for
the shipped model, so results are directly comparable to the numbers
already recorded for turnaround_xgboost_v1:
  A. baseline (shipped): candidates = {bearish, bottoming}, horizon=125
  B. bottoming-only, SAME horizon=125 (isolates the "cleaner pool" effect
     alone, no horizon change yet)
  C. bottoming-only, SHORTER horizon=65 (~3 months) (tests the full
     hypothesis: narrower pool + the shorter horizon it should support)

Also reports empirical days-to-turnaround for resolved-positive rows in
each variant, to directly check the "waits less time" half of the
hypothesis rather than assuming it from the horizon parameter alone.

Usage:
    python -m scripts.test_turnaround_candidate_scope
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from features.regime import BAD_REGIMES, GOOD_REGIMES
from pipeline.logging_config import get_logger
from scripts.train_turnaround import XGB_PARAMS
from scripts.train_v5 import FEATURES_PATH, NUM_BOOST_ROUND, ml_metrics, prepare_panel, walk_forward_splits
from scripts.turnaround_labels import HOLD_DISQUALIFYING_REGIMES, HOLD_TRADING_DAYS

logger = get_logger("scripts.test_turnaround_candidate_scope")

THRESHOLDS_TO_REPORT = [0.70, 0.80, 0.85, 0.90]


def label_and_days(regimes: np.ndarray, starting_regimes: set, horizon: int, hold: int = HOLD_TRADING_DAYS):
    """Same resolution logic as scripts.turnaround_labels.label_turnaround_series,
    parameterized by starting_regimes/horizon (not importable as-is since the
    shipped version hardcodes BAD_REGIMES/HORIZON_TRADING_DAYS at call time),
    plus tracks days-to-turnaround (t - i) for resolved-positive rows."""
    n = len(regimes)
    labels = np.full(n, np.nan)
    days = np.full(n, np.nan)
    is_start = np.array([r in starting_regimes for r in regimes])
    is_good = np.array([r in GOOD_REGIMES for r in regimes])

    for i in range(n):
        if not is_start[i]:
            continue
        window_end = i + horizon
        if window_end >= n:
            continue
        resolved = False
        had_unverifiable_attempt = False
        for t in range(i + 1, window_end + 1):
            if not is_good[t]:
                continue
            hold_end = t + hold
            if hold_end >= n:
                had_unverifiable_attempt = True
                continue
            held = all(regimes[k] not in HOLD_DISQUALIFYING_REGIMES for k in range(t + 1, hold_end + 1))
            if held:
                labels[i] = 1.0
                days[i] = t - i
                resolved = True
                break
        if not resolved and labels[i] != 1.0 and not had_unverifiable_attempt:
            labels[i] = 0.0
    return labels, days


def build_labels(features: pd.DataFrame, starting_regimes: set, horizon: int):
    all_labels, all_days = [], []
    for code, g in features.groupby("stock_code"):
        g = g.sort_values("date")
        regimes = g["regime"].fillna("").to_numpy()
        labels, days = label_and_days(regimes, starting_regimes, horizon)
        all_labels.append(pd.DataFrame({"stock_code": code, "date": g["date"].to_numpy(), "turnaround_label": labels}))
        all_days.append(days)
    return pd.concat(all_labels, ignore_index=True), np.concatenate(all_days)


def run_walk_forward(df, feature_cols, splits, label_name):
    fold_metrics = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("[%s] fold %d: skipped (insufficient data or single-class train set)", label_name, fold_i)
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        row = {"fold": fold_i, "n_test": len(X_test), "test_pos_rate": float(y_test.mean())}
        for thr in THRESHOLDS_TO_REPORT:
            m = ml_metrics(y_test, prob, thr)
            row[f"precision@{thr}"] = m["precision"]
            row[f"recall@{thr}"] = m["recall"]
            row[f"n_signals@{thr}"] = m["n_buy_signals"]
        row["roc_auc"] = ml_metrics(y_test, prob, 0.5)["roc_auc"]
        fold_metrics.append(row)
        logger.info("[%s] fold %d: n_test=%d roc_auc=%.3f precision@0.85=%.3f n_signals@0.85=%d",
                    label_name, fold_i, len(X_test), row["roc_auc"], row["precision@0.85"], row["n_signals@0.85"])
    return pd.DataFrame(fold_metrics)


def test_variant(features, name, starting_regimes, horizon):
    logger.info("=" * 70)
    logger.info("VARIANT %s: starting_regimes=%s horizon=%d", name, sorted(starting_regimes), horizon)
    logger.info("=" * 70)

    labels, days = build_labels(features, starting_regimes, horizon)
    resolved = labels["turnaround_label"].notna()
    n_resolved = int(resolved.sum())
    base_rate = labels.loc[resolved, "turnaround_label"].mean() if n_resolved else float("nan")
    days_pos = days[~np.isnan(days)]
    logger.info("Candidate rows (started in %s) resolved: %d, base_rate=%.1f%%, "
                "days-to-turnaround: mean=%.1f median=%.1f (n_positive=%d)",
                sorted(starting_regimes), n_resolved, base_rate * 100,
                float(np.mean(days_pos)) if len(days_pos) else float("nan"),
                float(np.median(days_pos)) if len(days_pos) else float("nan"),
                len(days_pos))

    labels = labels.rename(columns={"turnaround_label": "label"})
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel: %d rows, %d features", len(df), len(feature_cols))

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
    fold_df = run_walk_forward(df, feature_cols, splits, name)

    if fold_df.empty:
        logger.warning("[%s] no usable folds", name)
        return {"name": name, "n_resolved": n_resolved, "base_rate": base_rate,
                "days_mean": float(np.mean(days_pos)) if len(days_pos) else None,
                "days_median": float(np.median(days_pos)) if len(days_pos) else None,
                "n_folds": 0}

    avg = fold_df.mean(numeric_only=True)
    logger.info("[%s] AVERAGE across %d folds: roc_auc=%.3f", name, len(fold_df), avg["roc_auc"])
    for thr in THRESHOLDS_TO_REPORT:
        logger.info("  @%.2f: precision=%.3f recall=%.3f n_signals=%.0f",
                    thr, avg[f"precision@{thr}"], avg[f"recall@{thr}"], avg[f"n_signals@{thr}"])

    return {
        "name": name, "n_resolved": n_resolved, "base_rate": base_rate,
        "days_mean": float(np.mean(days_pos)) if len(days_pos) else None,
        "days_median": float(np.median(days_pos)) if len(days_pos) else None,
        "n_folds": len(fold_df), "roc_auc": float(avg["roc_auc"]),
        **{f"precision@{thr}": float(avg[f"precision@{thr}"]) for thr in THRESHOLDS_TO_REPORT},
        **{f"recall@{thr}": float(avg[f"recall@{thr}"]) for thr in THRESHOLDS_TO_REPORT},
    }


if __name__ == "__main__":
    logger.info("Loading %s...", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    logger.info("Loaded %d feature rows", len(features))

    results = [
        test_variant(features, "A_baseline_bearish+bottoming_h125", BAD_REGIMES, 125),
        test_variant(features, "B_bottoming_only_h125", {"bottoming"}, 125),
        test_variant(features, "C_bottoming_only_h65", {"bottoming"}, 65),
    ]

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    summary = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    logger.info("\n%s", summary.to_string(index=False))
