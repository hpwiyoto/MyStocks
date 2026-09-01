"""v5 training, run directly in Codespaces (not Colab -- sklearn isn't
installed here, so this uses xgboost's low-level Booster API throughout,
same as engine/model.py's serving path). Ports notebooks/fase3_ml_research
.ipynb's methodology (triple-barrier labeling, walk-forward split with
embargo) to numpy/xgboost only, no sklearn.

Reads the already-exported data/export_for_colab_{features,prices}.parquet
(already excludes gocap-floor rows, already has the full v5 feature set)
rather than re-querying MySQL -- that data is already correct and this
avoids repeating the OOM-prone full-table query export_for_colab.py itself
had to be rewritten to avoid.

Usage:
    python -m scripts.train_v5
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger

logger = get_logger("scripts.train_v5")

MODEL_DIR = "models"
FEATURES_PATH = "data/export_for_colab_features.parquet"
PRICES_PATH = "data/export_for_colab_prices.parquet"

HORIZON = 10
TARGET_PCT = 0.05
STOP_PCT = 0.025
BUY_THRESHOLD = 0.65  # tuned via scripts/tune_v5.py's pooled out-of-sample sweep: 0.65 beat both
                      # 0.60 (precision 75.0%->81.7%, profit_factor 6.00->8.92, max_dd -16.2%->-9.8%)
                      # AND 0.70 (which regressed slightly on profit_factor/drawdown despite fewer
                      # trades) -- unlike v4's sweep, which improved monotonically all the way to
                      # 0.70, v5's peaks at 0.65.

NON_FEATURE_COLS = {"id", "stock_code", "date", "feature_version", "created_at"}
BOOL_COLS = ["higher_high_20d", "higher_low_20d", "lower_high_20d", "lower_low_20d"]

# Absolute price/volume-level columns -- not comparable across stocks with
# different price levels (same exclusion the fase3 notebook established;
# none of v5's 6 new columns belong here -- obv_zscore_20 is a z-score,
# price_vs_vwap20_pct/sector_relative_strength_20d_pct are already %,
# trailing_pe/price_to_book are ratios, market_cap_log is log-compressed).
# net_foreign_flow (RapidAPI IDX, pipeline/idx_rapidapi_source.py) is the
# same category -- raw Rupiah, hundreds of billions for a mega-cap vs
# millions for a small-cap, not comparable across stocks. Its normalized
# form (net_foreign_flow_zscore_20, same rolling-zscore treatment as
# obv_zscore_20) is what actually gets used as a feature -- see
# scripts/test_foreign_flow_feature.py for the empirical A/B test this was
# built for before committing it here permanently.
ABSOLUTE_SCALE_COLS = {
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_20", "ema_50",
    "ema20_slope_5d", "ema20_accel_5d", "sma50_slope_10d",
    "macd", "macd_signal", "macd_hist", "macd_hist_slope_3d", "macd_hist_accel_3d",
    "volume_slope_5d", "obv", "obv_slope_5d", "net_foreign_flow",
}

XGB_PARAMS = {
    # max_depth=3 (not v4's 4): selected via scripts/tune_v5.py's 10-config
    # walk-forward grid search, holistic pick (AUC + profit_factor +
    # max_drawdown together) -- max_depth 5/6 had marginally higher AUC
    # (0.587-0.588 vs 0.581) but noticeably worse profit_factor (6.6-6.9 vs
    # 8.2) and drawdown (-17% to -18% vs -9.6%), the same overfitting
    # signature v4's own hyperparameter search rejected.
    "max_depth": 3, "eta": 0.05, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "objective": "binary:logistic", "eval_metric": "logloss", "seed": 42,
}
NUM_BOOST_ROUND = 200


def triple_barrier_label(high, low, close, entry_idx, horizon=10, target_pct=0.05, stop_pct=0.025):
    n = len(close)
    if entry_idx + horizon >= n:
        return np.nan
    entry_price = close[entry_idx]
    target_price = entry_price * (1 + target_pct)
    stop_price = entry_price * (1 - stop_pct)
    for i in range(entry_idx + 1, entry_idx + horizon + 1):
        if low[i] <= stop_price:
            return 0
        if high[i] >= target_price:
            return 1
    return np.nan


def build_panel_labels(prices: pd.DataFrame) -> pd.DataFrame:
    all_labels = []
    for code, g in prices.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        labels = [
            triple_barrier_label(high, low, close, i, HORIZON, TARGET_PCT, STOP_PCT)
            for i in range(len(g))
        ]
        all_labels.append(pd.DataFrame({"stock_code": code, "date": g["date"], "label": labels}))
    return pd.concat(all_labels, ignore_index=True)


def walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10):
    unique_dates = np.sort(np.unique(dates))
    n = len(unique_dates)
    splits = []
    test_start_pos = min_train_days
    for _ in range(n_splits):
        if test_start_pos >= n:
            break
        test_end_pos = min(test_start_pos + test_size_days, n)
        embargo_end_pos = test_start_pos - 1 - label_horizon
        if embargo_end_pos < 0:
            test_start_pos = test_end_pos
            continue
        splits.append({
            "train_embargo_end_date": unique_dates[embargo_end_pos],
            "test_start_date": unique_dates[test_start_pos],
            "test_end_date": unique_dates[test_end_pos - 1],
        })
        test_start_pos = test_end_pos
    return splits


def prepare_panel(features: pd.DataFrame, labels: pd.DataFrame):
    df = features.merge(labels, on=["stock_code", "date"], how="inner")
    df = df[df["sma_200"].notna()].copy()  # warmup filter only, sma_200 itself excluded as a feature

    df["has_similar_pattern"] = (df["similar_pattern_count"].fillna(0) > 0).astype(int)
    df["historical_win_rate"] = df["historical_win_rate"].fillna(0.5)

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    for c in BOOL_COLS:
        df[c] = df[c].astype(float)

    regime_dummies = pd.get_dummies(df["regime"], prefix="regime", dtype=float)
    df = pd.concat([df.drop(columns=["regime"]), regime_dummies], axis=1)

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != "label" and c not in ABSOLUTE_SCALE_COLS
    ]

    df = df.sort_values("date").reset_index(drop=True)
    return df, feature_cols


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mann-Whitney U / rank-sum AUC -- no sklearn here, so implemented
    directly. Standard formula: AUC = (sum of positive-class ranks -
    n_pos*(n_pos+1)/2) / (n_pos*n_neg)."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob))
    ranks[order] = np.arange(1, len(y_prob) + 1)
    # average ranks for ties
    sorted_prob = y_prob[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_prob):
        j = i
        while j + 1 < len(sorted_prob) and sorted_prob[j + 1] == sorted_prob[i]:
            j += 1
        if j > i:
            sorted_ranks[i:j + 1] = sorted_ranks[i:j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ml_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "precision": precision, "recall": recall,
        "roc_auc": roc_auc(y_true, y_prob),
        "n_buy_signals": int(y_pred.sum()),
    }


def trading_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    taken = y_prob >= threshold
    outcomes = y_true[taken]
    returns = np.where(outcomes == 1, TARGET_PCT, -STOP_PCT)
    if len(returns) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"), "max_drawdown_pct": float("nan")}
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("nan")
    return {
        "n_trades": len(returns),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(drawdown.min() * 100),
    }


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    labels = build_panel_labels(prices)
    logger.info("Label distribution: %s", labels["label"].value_counts(dropna=False).to_dict())

    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features", len(df), len(feature_cols))
    logger.info("Feature list: %s", feature_cols)

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    logger.info("%d walk-forward folds generated", len(splits))

    fold_ml_metrics = []
    fold_trading_metrics = []
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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        m = ml_metrics(y_test, prob, BUY_THRESHOLD)
        t = trading_metrics(y_test, prob, BUY_THRESHOLD)
        fold_ml_metrics.append(m)
        fold_trading_metrics.append(t)
        logger.info("Fold %d @ threshold=%.2f: precision=%.3f recall=%.3f roc_auc=%.3f | "
                    "n_trades=%d win_rate=%.3f profit_factor=%.2f max_dd=%.1f%%",
                    fold_i, BUY_THRESHOLD, m["precision"], m["recall"], m["roc_auc"],
                    t["n_trades"], t["win_rate"], t["profit_factor"], t["max_drawdown_pct"])

    logger.info("=" * 70)
    logger.info("AVERAGE ACROSS %d FOLDS (threshold=%.2f)", len(fold_ml_metrics), BUY_THRESHOLD)
    avg_ml = pd.DataFrame(fold_ml_metrics).mean().to_dict()
    avg_trading = pd.DataFrame(fold_trading_metrics).mean().to_dict()
    logger.info("ML: %s", {k: round(v, 4) for k, v in avg_ml.items()})
    logger.info("Trading: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in avg_trading.items()})
    logger.info("=" * 70)

    # Final model: trained on ALL data (walk-forward folds above are purely
    # for validation/reporting), same as v1-v4's convention.
    logger.info("Training final model on all %d rows...", len(df))
    dall = xgb.DMatrix(df[feature_cols], label=df["label"])
    final_booster = xgb.train(XGB_PARAMS, dall, num_boost_round=NUM_BOOST_ROUND)

    base_rate = float(df["label"].mean())
    model_path = f"{MODEL_DIR}/direction_xgboost_v5.json"
    meta_path = f"{MODEL_DIR}/direction_xgboost_v5_metadata.json"
    final_booster.save_model(model_path)

    metadata = {
        "model_version": "direction_xgboost_v5",
        "trained_at": dt.date.today().isoformat(),
        "feature_cols": feature_cols,
        "base_rate": base_rate,
        "horizon_days": HORIZON,
        "target_pct": TARGET_PCT,
        "stop_pct": STOP_PCT,
        "n_training_rows": len(df),
        "tickers": sorted(df["stock_code"].unique().tolist()),
        "hyperparameters": {**XGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND},
        "walk_forward_validation": {
            "n_folds": len(fold_ml_metrics),
            "buy_threshold": BUY_THRESHOLD,
            "avg_ml_metrics": {k: round(float(v), 4) for k, v in avg_ml.items()},
            "avg_trading_metrics": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in avg_trading.items()},
        },
        "notes": (
            "v5 feature set: v4's features + obv_zscore_20, price_vs_vwap20_pct, "
            "sector_relative_strength_20d_pct, trailing_pe, price_to_book, market_cap_log. "
            "Gocap-floor rows (close<=Rp50) excluded from training -- see engine.decision."
            "GOCAP_PRICE_FLOOR. Trained directly in Codespaces (xgb.train Booster API, no "
            "sklearn), same hyperparameters as v4 (no re-tuning yet this round)."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved %s + %s", model_path, meta_path)
    return metadata


if __name__ == "__main__":
    run()
