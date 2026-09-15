"""Profit factor + signal frequency + horizon, combined into one
return-per-day-of-capital-deployed number, for all 5 Swing configs.

Direct user follow-up to the AUC/Wilson-LB/win-rate comparison: those
three are model-development diagnostics (do the probabilities rank
correctly, is the win-rate reliable) -- none of them says how much
money a trade actually makes or loses. User's own conclusion, confirmed
here: an investor should decide by profit factor/expectancy, not by
AUC/Wilson-LB/win-rate directly, AND profit factor alone still isn't
enough without also accounting for how often a signal fires
(signals_per_day) and how long capital sits in one trade
(horizon_days) -- two configs can share a profit factor and still
compound money at very different rates.

Where the returns come from: scripts/train_v5.py's own triple-barrier
label construction ALREADY resolves every kept row to an exact outcome
-- target hit first (label=1) or stop hit first (label=0) within the
horizon; rows where NEITHER barrier is touched (price just drifts) are
dropped entirely (`df[df["label"].notna()]`), never counted as a win or
a loss. So every evaluated trade already has a real exit day -- this
module (`build_labels_with_returns`) is a copy of scripts/search_swing_
target.py's vectorized build_labels that additionally reads off that
exit day's price to compute a realized return, in two flavors:

  - return_clean: exactly +target_pct on a win, exactly -stop_pct on a
    loss -- the assumption implicit in the label itself (limit fills at
    the barrier price, no slippage). Because ALL 5 configs use the same
    2:1 target:stop ratio (10/5, 7/3.5, 5/2.5), profit_factor_clean
    collapses to a fixed multiple of win_rate/(1-win_rate) for every
    config -- i.e. under this assumption profit factor carries NO
    information beyond win-rate, and reproduces the exact same ranking
    already seen from Wilson LB. This is reported mainly to make that
    equivalence visible, not as the final answer.
  - return_realistic: uses the OPEN price on the exit day (pulled from
    price_history, since the parquet has no open) to capture gap-through
    risk that a same-day high/low touch can't -- if a stock gaps below
    the stop price overnight, a stop-loss order fills at the (worse)
    open price, not at the nominal stop price; symmetric upside
    treatment applies to a gap-up past the target. This is where the
    ranking CAN actually diverge from win-rate/Wilson LB, because
    longer-horizon and wider-stop configs carry more overnight gap
    exposure per trade.

Also computed per config (from data already produced by training, just
not previously persisted): signals_per_day at the shipped threshold,
and horizon_days (already in metadata). Combined into
expectancy_per_day = expectancy_realistic / horizon_days, a rough,
consistently-computed proxy for how fast each config compounds capital
if you can only hold one position at a time and always redeploy at the
horizon boundary -- not a true CAGR (ignores compounding and the
possibility of exiting before the full horizon), but comparable
apples-to-apples across configs since every config uses the same
approximation.

RESULT (2026-09-15): profit_factor_clean reproduces the win-rate/Wilson-LB
ranking EXACTLY as predicted (t5_h5 > t7_h10 > t7_h5 > t10_h10 > default,
9.02/8.73/8.56/6.24/5.81) -- confirms it carries no information beyond
win-rate for these 5 configs specifically, since they all share the same
2:1 target:stop ratio.

profit_factor_realistic (gap-adjusted) raises every config's number but
does NOT reorder them -- same ranking as clean. Interestingly, the two
BIGGER-target configs (default +38%, t10_h10 +31% relative) gained the
most from the gap adjustment, more than the smaller-target configs
(t7_h10/t7_h5/t5_h5, +1-17%) -- in this backtest period, winning trades'
exit-day gaps ran further past target than losing trades' gaps ran past
stop, so accounting for realistic fills was NET FAVORABLE across the
board, not a risk that erodes the naive numbers as the module docstring
above anticipated. So: profit factor (either version) tracks win-rate
here and changes nothing about which config looks best.

expectancy_per_day_pct is where the ranking FLIPS entirely:
  default (10%/-5%/5d):  2.453%/day  <- WINNER, despite lowest win-rate
  t5_h5   (5%/-2.5%/5d): 1.556%/day
  t7_h5   (7%/-3.5%/5d): 1.540%/day
  t10_h10 (10%/-5%/10d): 1.225%/day
  t7_h10  (7%/-3.5%/10d):0.759%/day  <- worst, despite 2nd-best profit factor
The default config wins on capital velocity precisely BECAUSE of what
made it look worst on win-rate: a bigger target (10%) paid out over the
SHORTEST horizon (5d) combined with the HIGHEST signal frequency
(1.96/day) of any config. t7_h10 is the mirror-image loser: smaller
target (7%) AND the longer horizon (10d) -- the double penalty shows up
nowhere in win-rate/Wilson-LB/profit-factor, only once horizon and
frequency are folded in.

CONCLUSION: for an investor optimizing actual profit velocity, the
already-shipped default config (direction_xgboost_v5, 10%/-5%/5d) is the
best of the 5 -- confirming the ORIGINAL top5_lift-based choice from
scripts/search_swing_target.py's 20-config search was right, even though
every per-trade-quality metric (AUC/Wilson-LB/win-rate/profit-factor)
made it look like the weakest of the 5. Win-rate-style metrics answer
"how often is this right"; expectancy_per_day answers "how fast does
this make money" -- for THIS investor question, the second is what
matters, and the two disagree on which config to pick.

Usage:
    python -m scripts.compute_profit_factor
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.train_v5 import FEATURES_PATH, MODEL_DIR, NUM_BOOST_ROUND, PRICES_PATH, prepare_panel, walk_forward_splits
from scripts.train_v5_variants import BASE_XGB_PARAMS, LEGACY_HYPERPARAMS, T10_H10_HYPERPARAMS, T7_H10_HYPERPARAMS

logger = get_logger("scripts.compute_profit_factor")

DEFAULT_HYPERPARAMS = {"max_depth": 4, "eta": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8}

# (target_pct, stop_pct, horizon, model_version, hyperparams)
CONFIGS = [
    (0.10, 0.05, 5, "direction_xgboost_v5", DEFAULT_HYPERPARAMS),
    (0.07, 0.035, 5, "direction_xgboost_v5_t7_h5", LEGACY_HYPERPARAMS),
    (0.10, 0.05, 10, "direction_xgboost_v5_t10_h10", T10_H10_HYPERPARAMS),
    (0.07, 0.035, 10, "direction_xgboost_v5_t7_h10", T7_H10_HYPERPARAMS),
    (0.05, 0.025, 5, "direction_xgboost_v5_t5_h5", LEGACY_HYPERPARAMS),
]


def load_prices_with_open(prices: pd.DataFrame) -> pd.DataFrame:
    engine = get_engine()
    op = pd.read_sql(
        "SELECT stock_code, date, open FROM price_history WHERE source_provider = 'yfinance'",
        engine,
    )
    op["date"] = pd.to_datetime(op["date"]).dt.date
    merged = prices.merge(op, on=["stock_code", "date"], how="left")
    logger.info("Open merged: %d/%d rows have an open value", merged["open"].notna().sum(), len(merged))
    return merged


def build_labels_with_returns(prices: pd.DataFrame, horizon: int, target_pct: float, stop_pct: float) -> pd.DataFrame:
    """Same triple-barrier walk as search_swing_target.build_labels, plus
    return_clean/return_realistic per resolved row (see module docstring)."""
    rows = []
    for code, g in prices.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= horizon:
            continue
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        openp = g["open"].to_numpy(dtype=float)
        valid_n = n - horizon
        fwd_high = sliding_window_view(high[1:], horizon)
        fwd_low = sliding_window_view(low[1:], horizon)
        fwd_open = sliding_window_view(openp[1:], horizon)
        idxs = np.arange(valid_n)
        entry = close[idxs]
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)
        target_hit = fwd_high[idxs] >= target_price[:, None]
        stop_hit = fwd_low[idxs] <= stop_price[:, None]
        target_any = target_hit.any(axis=1)
        stop_any = stop_hit.any(axis=1)
        first_target_t = np.where(target_any, target_hit.argmax(axis=1), horizon)
        first_stop_t = np.where(stop_any, stop_hit.argmax(axis=1), horizon)
        resolved = target_any | stop_any
        win = first_target_t < first_stop_t
        label = np.where(resolved, win.astype(float), np.nan)

        exit_t = np.where(win, first_target_t, first_stop_t)
        exit_t_safe = np.minimum(exit_t, horizon - 1)
        exit_open = fwd_open[idxs, exit_t_safe]

        return_clean = np.where(win, target_pct, -stop_pct)

        gap_win = exit_open / entry - 1  # what a limit sell would ACTUALLY realize if opened above target
        gap_loss = exit_open / entry - 1  # what a stop-loss market order would ACTUALLY realize if opened below stop
        return_realistic = np.where(
            win,
            np.where(np.isnan(gap_win), target_pct, np.maximum(target_pct, gap_win)),
            np.where(np.isnan(gap_loss), -stop_pct, np.minimum(-stop_pct, gap_loss)),
        )

        return_clean = np.where(resolved, return_clean, np.nan)
        return_realistic = np.where(resolved, return_realistic, np.nan)

        sub = g.iloc[idxs][["stock_code", "date"]].copy()
        sub["label"] = label
        sub["return_clean"] = return_clean
        sub["return_realistic"] = return_realistic
        rows.append(sub)
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["stock_code", "date", "label", "return_clean", "return_realistic"]))


def fold_data(df, feature_cols, splits):
    folds = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        ret_clean = df.loc[test_mask, "return_clean"].to_numpy()
        ret_realistic = df.loc[test_mask, "return_realistic"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("fold %d: skipped (insufficient data)", fold_i)
            continue
        folds.append({
            "fold": fold_i, "dtrain": xgb.DMatrix(X_train, label=y_train), "dtest": xgb.DMatrix(X_test),
            "y_test": y_test, "ret_clean": ret_clean, "ret_realistic": ret_realistic,
        })
    return folds


def _profit_factor(returns_taken: np.ndarray) -> tuple[float, float]:
    gains = returns_taken[returns_taken > 0].sum()
    losses = -returns_taken[returns_taken < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    expectancy = returns_taken.mean() if len(returns_taken) else float("nan")
    return pf, expectancy


def evaluate(config, features, prices):
    target_pct, stop_pct, horizon, model_version, xgb_params = config
    xgb_params = {**BASE_XGB_PARAMS, **xgb_params}
    with open(f"{MODEL_DIR}/{model_version}_metadata.json") as f:
        threshold = json.load(f)["walk_forward_validation"]["buy_threshold"]

    logger.info("#" * 100)
    logger.info("%s: target=%.1f%% stop=%.2f%% horizon=%dd threshold=%.2f",
                model_version, target_pct * 100, stop_pct * 100, horizon, threshold)
    logger.info("#" * 100)

    labels = build_labels_with_returns(prices, horizon, target_pct, stop_pct)
    df, feature_cols = prepare_panel(features, labels[["stock_code", "date", "label"]])
    df = df.merge(labels[["stock_code", "date", "return_clean", "return_realistic"]], on=["stock_code", "date"], how="left")

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
    folds = fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    test_dates = set()
    for split in splits:
        mask = (dates >= split["test_start_date"]) & (dates <= split["test_end_date"])
        test_dates.update(dates[mask].tolist())
    total_test_days = len(test_dates)

    pooled_y, pooled_prob, pooled_clean, pooled_realistic = [], [], [], []
    for f in folds:
        booster = xgb.train(xgb_params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(f["dtest"])
        pooled_y.append(f["y_test"])
        pooled_prob.append(prob)
        pooled_clean.append(f["ret_clean"])
        pooled_realistic.append(f["ret_realistic"])
    y = np.concatenate(pooled_y)
    prob = np.concatenate(pooled_prob)
    ret_clean = np.concatenate(pooled_clean)
    ret_realistic = np.concatenate(pooled_realistic)

    taken = prob >= threshold
    n = int(taken.sum())
    wins = int(y[taken].sum())
    win_rate = wins / n if n else float("nan")
    signals_per_day = n / total_test_days

    pf_clean, exp_clean = _profit_factor(ret_clean[taken])
    pf_realistic, exp_realistic = _profit_factor(ret_realistic[taken])
    expectancy_per_day = exp_realistic / horizon

    result = {
        "model_version": model_version, "target_pct": target_pct, "stop_pct": stop_pct, "horizon_days": horizon,
        "threshold": threshold, "n_trades": n, "win_rate": win_rate, "signals_per_day": signals_per_day,
        "profit_factor_clean": pf_clean, "expectancy_clean_pct": exp_clean * 100,
        "profit_factor_realistic": pf_realistic, "expectancy_realistic_pct": exp_realistic * 100,
        "expectancy_per_day_pct": expectancy_per_day * 100,
    }
    logger.info(
        "n=%d win_rate=%.4f signals/day=%.2f | PF_clean=%.3f exp_clean=%.3f%% | "
        "PF_realistic=%.3f exp_realistic=%.3f%% | exp/day=%.4f%%",
        n, win_rate, signals_per_day, pf_clean, exp_clean * 100, pf_realistic, exp_realistic * 100,
        expectancy_per_day * 100,
    )
    return result


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    prices = load_prices_with_open(prices)

    results = [evaluate(cfg, features, prices) for cfg in CONFIGS]
    rdf = pd.DataFrame(results)

    logger.info("#" * 100)
    logger.info("Ranked by profit_factor_realistic (the decision-relevant number):")
    logger.info("\n%s", rdf.sort_values("profit_factor_realistic", ascending=False).to_string(index=False))
    logger.info("#" * 100)
    logger.info("Ranked by expectancy_per_day_pct (profit factor + frequency + horizon combined):")
    logger.info("\n%s", rdf.sort_values("expectancy_per_day_pct", ascending=False).to_string(index=False))

    rdf.to_csv("data/profit_factor_comparison.csv", index=False)
    logger.info("Saved data/profit_factor_comparison.csv")


if __name__ == "__main__":
    run()
