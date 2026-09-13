"""Search for the most optimal Swing target definition -- direct user
request: instead of assuming the current +5% target / -2.5% stop / 10-
trading-day horizon is optimal, search across magnitude AND horizon. Same
"search the target itself, don't assume it" spirit as scripts/
search_turnaround_v2_target.py + scripts/tune_v5_magnitude.py's earlier
Turnaround v2 work, applied here to Swing for the first time.

Grid: 4 target/stop pairs (keeping the CURRENT 2:1 reward:risk ratio
fixed -- that ratio itself has never been separately investigated; a
natural follow-up if this search turns something up, not folded in here
to keep the search 1-dimensional-in-magnitude x 1-dimensional-in-horizon
instead of a 3D grid) x 5 horizons = 20 configs. Each is a DIFFERENT
label, so unlike every earlier feature-only test in this project, each
config needs its own full walk-forward RETRAIN -- there's no way around
that when the target itself changes.

Fair cross-config comparison: a bigger target and/or longer horizon
mechanically shifts the unconditional null win rate (confirmed in the
Turnaround v2 magnitude sweep -- null rose from 32.7% at +10% to 36.8% at
+30%), so comparing raw precision at a FIXED probability threshold like
0.65 across configs isn't apples-to-apples -- that threshold was tuned
specifically for the CURRENT target. Instead, for every config, take the
TOP TOP_PCT of that config's own pooled out-of-sample probability
distribution (rank-based selection, same idea as this project's various
"top-N/day" backtests) and report the LIFT of its Wilson LB over that
SAME config's own null baseline -- the lift is what's comparable across
configs, not the raw win rate.

RESULT (2026-09-13): the CURRENTLY SHIPPED target (5%/2.5%/10d) is NOT
the best in this grid -- it ranks 8th of 20 by top5_lift. The winner:
target=10% / stop=5% / horizon=5 trading days -- top5_lb=51.7%
(lift+0.228 over its own null 28.9%) vs shipped's top5_lb=46.8%
(lift+0.168 over null 30.1%). Runner-up: target=7%/stop=3.5%/horizon=5d
(lift+0.203). Clear pattern across the whole grid: SHORTER horizons
consistently beat longer ones at every target size (ROC-AUC degrades
monotonically from ~0.59 at 5d to ~0.55-0.58 at 30d for every target),
and at short horizons, BIGGER targets (7-10%) separate winners better
than the current 5% -- the model can tell "about to make a big, fast
move" apart from noise more easily than "about to make a modest move",
counterintuitive but consistent (a sharper underlying event is easier to
classify even though it's individually rarer).

Real caveats, not yet resolved, before this could ship:
1. This changes what Swing fundamentally IS -- a "10-day swing, +5%/
   -2.5%" tool becomes a "5-day fast trade, +10%/-5%" one. Same 2:1
   ratio, but double the absolute risk per trade and half the holding
   period -- a product-identity decision, not a tuning knob, and needs an
   explicit go-ahead as such.
2. Coverage: the winning config's resolved panel is smaller (340,829 rows
   vs shipped's 618,559) -- a big move within only 5 days resolves (hits
   target or stop) less often than a modest move within 10, so fewer
   candidates ever produce a definitive resolved label at all. The lift
   numbers above are real, but come from a narrower base.
3. BUY_THRESHOLD=0.65 was tuned for the CURRENT label. This search used
   top-5%/10%-of-predictions as a threshold-agnostic ranking metric
   specifically so configs could be compared fairly -- it does NOT tell
   us what the right absolute probability cutoff would be for the new
   label; that would need its own scripts/tune_v5.py-style threshold
   sweep before shipping.

Full 20-row table (all target/stop/horizon combos, sorted by top5_lift)
saved to data/search_swing_target_results.csv.

Usage:
    python -m scripts.search_swing_target
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import FEATURES_PATH, NUM_BOOST_ROUND, PRICES_PATH, XGB_PARAMS
from scripts.train_v5 import ml_metrics, prepare_panel, walk_forward_splits

logger = get_logger("scripts.search_swing_target")

TARGET_STOP_PAIRS = [(0.03, 0.015), (0.05, 0.025), (0.07, 0.035), (0.10, 0.05)]  # keeps current 2:1 ratio
HORIZONS = [5, 10, 15, 20, 30]
TOP_PCTS = [0.05, 0.10]
RESULTS_CSV = "data/search_swing_target_results.csv"


def build_labels(prices: pd.DataFrame, horizon: int, target_pct: float, stop_pct: float) -> pd.DataFrame:
    """Vectorized triple-barrier label -- same semantics as scripts/
    train_v5.py's triple_barrier_label (stop checked before target on a
    same-day tie, first-touch-wins across the horizon), just done with
    sliding-window numpy ops instead of a pure-Python per-row loop so a
    20-config sweep with horizons up to 30 stays fast."""
    rows = []
    for code, g in prices.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= horizon:
            continue
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        valid_n = n - horizon
        fwd_high = sliding_window_view(high[1:], horizon)  # row i -> high[i+1 : i+1+horizon]
        fwd_low = sliding_window_view(low[1:], horizon)
        idxs = np.arange(valid_n)
        entry = close[idxs]
        target_hit = fwd_high[idxs] >= entry[:, None] * (1 + target_pct)
        stop_hit = fwd_low[idxs] <= entry[:, None] * (1 - stop_pct)
        target_any = target_hit.any(axis=1)
        stop_any = stop_hit.any(axis=1)
        first_target_t = np.where(target_any, target_hit.argmax(axis=1), horizon)
        first_stop_t = np.where(stop_any, stop_hit.argmax(axis=1), horizon)
        resolved = target_any | stop_any
        win = first_target_t < first_stop_t  # same-day tie -> stop wins, matches train_v5.py's convention
        label = np.where(resolved, win.astype(float), np.nan)
        sub = g.iloc[idxs][["stock_code", "date"]].copy()
        sub["label"] = label
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stock_code", "date", "label"])


def run_walk_forward_pooled(df, feature_cols, splits, xgb_params):
    pooled_y, pooled_prob = [], []
    for split in splits:
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        pooled_y.append(y_test)
        pooled_prob.append(prob)
    if not pooled_y:
        return np.array([]), np.array([])
    return np.concatenate(pooled_y), np.concatenate(pooled_prob)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    results = []
    for target_pct, stop_pct in TARGET_STOP_PAIRS:
        for horizon in HORIZONS:
            label_name = f"target={target_pct*100:.1f}%/stop={stop_pct*100:.2f}%/horizon={horizon}d"
            labels = build_labels(prices, horizon, target_pct, stop_pct)
            df, feature_cols = prepare_panel(features, labels)
            if df.empty:
                logger.info("[%s] empty panel, skipped", label_name)
                continue
            null_win_rate = float(df["label"].mean())

            dates = df["date"].to_numpy()
            splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
            y, prob = run_walk_forward_pooled(df, feature_cols, splits, XGB_PARAMS)
            if len(y) == 0:
                logger.info("[%s] no usable folds, skipped", label_name)
                continue
            roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"]

            row = {
                "target_pct": target_pct, "stop_pct": stop_pct, "horizon": horizon,
                "n_panel": len(df), "n_pooled_oos": len(y), "null_win_rate": null_win_rate, "roc_auc": roc_auc,
            }
            log_bits = [f"null={null_win_rate:.3f}", f"roc_auc={roc_auc:.4f}"]
            for top_pct in TOP_PCTS:
                k = max(1, int(len(prob) * top_pct))
                top_idx = np.argsort(prob)[-k:]
                n_top = len(top_idx)
                wins_top = int(y[top_idx].sum())
                wr_top = wins_top / n_top
                lb_top = wilson_lower_bound(wins_top, n_top)
                tag = int(top_pct * 100)
                row[f"top{tag}_n"] = n_top
                row[f"top{tag}_wr"] = wr_top
                row[f"top{tag}_lb"] = lb_top
                row[f"top{tag}_lift"] = lb_top - null_win_rate
                log_bits.append(f"top{tag}_lb={lb_top:.3f}(lift{lb_top - null_win_rate:+.3f})")
            results.append(row)
            logger.info("[%s] %s", label_name, " ".join(log_bits))

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS, sorted by top5_lift (lift over that config's OWN null baseline):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.sort_values("top5_lift", ascending=False).to_string(index=False))
    rdf.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)


if __name__ == "__main__":
    run()
