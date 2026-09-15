"""Continuation of the parameter-optimality trilogy (slope windows, MA
choice, hyperparameters, RSI/MFI/CMF/ATR/BB, regime thresholds, ADX/RVOL/
VWAP/structure) into the two features scripts/test_more_window_params.py
explicitly deferred: relative_strength_20d_pct (vs IHSG) and
sector_relative_strength_20d_pct (vs this ticker's own sector composite).

That deferral's stated reason -- "need IHSG + sector composite
reconstructed, a bigger setup than the others" -- turned out to overstate
the blocker: both already exist as live production infrastructure
(features/build_features.py's `_load_ihsg_close`/`_build_sector_
composites`, run daily), reused as-is here rather than rebuilt. Neither
depends on the window under test, so each is computed ONCE up front and
shared across every candidate -- only the trailing pct_change window
itself varies per candidate, an O(n) recompute per ticker (unlike
pattern similarity's O(n_query * n_bank) cross-ticker search, which is
why that one gets a separate, more expensive treatment).

Same methodology and combined-recheck safeguard as the rest of the
trilogy: one group at a time (holding the other feature + the OTHER
relative-strength column at the shipped baseline), sweep candidates via
pooled walk-forward (target=10%/stop=5%/horizon=5d, threshold=0.60
fixed), combine whichever candidate wins each group and re-test before
drawing any conclusion.

Bug found and fixed mid-run: the first pass produced identical results
for EVERY sector_relative_strength_20d_pct candidate regardless of
window (confirmed via a direct per-ticker check -- the whole column
came back 100% NaN). Root cause: _build_sector_composites aligns its
members via pd.concat(..., axis=1), which matches rows by INDEX, and
this script had built `price_by_code` with reset_index(drop=True)
(plain 0..N integers) instead of a date index like features/
build_features.py's own convention -- the resulting composite's index
was meaningless integers that never matched a real date during the
later merge. Fixed by set_index("date") instead; re-verified the fix
produces genuinely different (non-NaN) values per window before
re-running the full sweep.

RESULT (2026-09-16): current window=20 NOT beaten for either feature --
no change adopted, same decisive pattern as every other trilogy test
except regime thresholds. Baseline (window=20/20): pooled n=804
wilson_lb=71.25%. relative_strength_20d_pct: 4/5 candidates worse
(-1.1pp to -1.8pp); window=40 the only "win" (+0.07pp, 71.32%).
sector_relative_strength_20d_pct (bug-fixed): all 5 candidates worse
(-0.5pp to -1.6pp), no exception. window=40's +0.07pp is smaller than
even the +0.05pp this project's OWN regime-threshold research (scripts/
test_regime_thresholds.py) already dismissed as noise, "an order of
magnitude smaller than even the unconfirmed lead magnitude [hyperparam
research] treated with caution" -- not treated as a real finding here
either, despite technically "beating" baseline in the combined-recheck
(which only re-runs a single already-negligible winner, same caveat as
that other script's one surviving candidate). Both relative-strength
windows (features/technical.py) left unchanged at 20.

Usage:
    python -m scripts.test_relative_strength_window
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from features.build_features import _build_sector_composites, _load_ihsg_close
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import (
    BUY_THRESHOLD,
    FEATURES_PATH,
    HORIZON,
    NUM_BOOST_ROUND,
    PRICES_PATH,
)
from scripts.train_v5 import XGB_PARAMS as SWING_XGB_PARAMS
from scripts.train_v5 import (
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)

logger = get_logger("scripts.test_relative_strength_window")

# Around the shipped 20 -- same "one trading month" family already tested
# for RVOL/VWAP/structure, plus a materially shorter/longer pair since
# relative strength (a momentum-vs-benchmark signal) plausibly behaves
# differently from a pure volatility/structure window.
WINDOW_CANDIDATES = [10, 15, 30, 40, 60]


def _sorted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _relative_strength_variant(prices: pd.DataFrame, ref_by_code: dict, window: int, new_col: str) -> pd.DataFrame:
    """ref_by_code: stock_code -> reference close Series (date-indexed,
    datetime.date index to match prices/features' own date dtype -- same
    normalization features/build_features.py's _load_ihsg_close already
    does). Same formula as features/technical.py's compute_relative_
    strength_series, reimplemented via merge-on-date instead of index
    .reindex() -- sidesteps needing every ticker's price slice to share
    an identical DatetimeIndex object, consistent with how this project's
    other variant-test scripts (e.g. test_regime_thresholds.py) join
    auxiliary series back onto the price panel."""
    out_rows = []
    for code, g in prices.groupby("stock_code"):
        ref = ref_by_code.get(code)
        if ref is None or ref.empty:
            out_rows.append(pd.DataFrame({"stock_code": code, "date": g["date"], new_col: np.nan}))
            continue
        ref_df = pd.DataFrame({"date": ref.index, "ref_close": ref.to_numpy()})
        merged = g[["date", "close"]].merge(ref_df, on="date", how="left")
        stock_return = merged["close"].pct_change(window)
        ref_return = merged["ref_close"].pct_change(window)
        out_rows.append(pd.DataFrame({
            "stock_code": code, "date": g["date"].to_numpy(),
            new_col: ((stock_return - ref_return) * 100).to_numpy(),
        }))
    out = pd.concat(out_rows, ignore_index=True)
    return out


def add_ihsg_relative_strength(features, prices, window, new_col, ihsg_close):
    ref_by_code = {code: ihsg_close for code in prices["stock_code"].unique()}
    out = _relative_strength_variant(_sorted_prices(prices), ref_by_code, window, new_col)
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_sector_relative_strength(features, prices, window, new_col, sector_composites, sector_by_code):
    codes = prices["stock_code"].unique()
    ref_by_code = {code: sector_composites.get(sector_by_code.get(code)) for code in codes}
    out = _relative_strength_variant(_sorted_prices(prices), ref_by_code, window, new_col)
    return features.merge(out, on=["stock_code", "date"], how="left")


def run_walk_forward_pooled(df, feature_cols, splits, xgb_params, label_name):
    pooled_y, pooled_prob = [], []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("[%s] fold %d: skipped (insufficient data)", label_name, fold_i)
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        pooled_y.append(y_test)
        pooled_prob.append(prob)
        logger.info("[%s] fold %d done: n_test=%d", label_name, fold_i, len(X_test))
    if not pooled_y:
        return np.array([]), np.array([])
    return np.concatenate(pooled_y), np.concatenate(pooled_prob)


def evaluate(df, feature_cols, splits, label_name):
    y, prob = run_walk_forward_pooled(df, feature_cols, splits, SWING_XGB_PARAMS, label_name)
    roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"] if len(y) else float("nan")
    buy_mask = prob >= BUY_THRESHOLD
    n = int(buy_mask.sum())
    wins = int(y[buy_mask].sum()) if n else 0
    precision = wins / n if n else float("nan")
    lb = wilson_lower_bound(wins, n) if n else 0.0
    return {"roc_auc": roc_auc, "n": n, "wins": wins, "precision": precision, "wilson_lb": lb}


def eval_variant(features_variant, labels, drop_col, label_name):
    panel_input = features_variant.drop(columns=[drop_col]) if drop_col in features_variant.columns else features_variant
    df, feature_cols = prepare_panel(panel_input, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    return evaluate(df, feature_cols, splits, label_name)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    logger.info("Fetching IHSG + rebuilding sector composites (reusing production helpers, computed once)...")
    ihsg_close = _load_ihsg_close()
    engine = get_engine()
    with engine.connect() as conn:
        sector_by_code = dict(conn.execute(text("SELECT code, sector FROM stocks")).fetchall())
    # _build_sector_composites aligns members via pd.concat(..., axis=1),
    # which matches rows by INDEX -- must be date-indexed (matching
    # features/build_features.py's own _fetch_price_df convention) or
    # different tickers' rows get aligned by row-POSITION instead of by
    # actual date, and the composite's own index ends up meaningless
    # integers that never match a real date during the merge later.
    # Caught live: an early run of this script produced an entirely-NaN
    # sector_relative_strength_20d_pct_custom column for every ticker
    # (confirmed via a direct check on BBCA) because this used
    # reset_index(drop=True) instead.
    price_by_code = {code: g.sort_values("date").set_index("date") for code, g in prices.groupby("stock_code")}
    sector_composites = _build_sector_composites(price_by_code, sector_by_code)
    logger.info("IHSG: %d rows. %d sector composites built.", len(ihsg_close), len(sector_composites))

    results = []
    logger.info("#" * 90)
    logger.info("BASELINE (current shipped feature set, unmodified)")
    logger.info("#" * 90)
    baseline_metrics = eval_variant(features, labels, "", "baseline")
    logger.info("[baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "window": "current", **baseline_metrics})

    GROUPS = {
        "relative_strength_20d_pct": (
            lambda feats, w, col: add_ihsg_relative_strength(feats, prices, w, col, ihsg_close)
        ),
        "sector_relative_strength_20d_pct": (
            lambda feats, w, col: add_sector_relative_strength(feats, prices, w, col, sector_composites, sector_by_code)
        ),
    }

    best_per_group = {}
    variant_by_winner = {}
    for group_col, add_fn in GROUPS.items():
        logger.info("#" * 90)
        logger.info("GROUP: %s", group_col)
        logger.info("#" * 90)
        best = {"wilson_lb": baseline_metrics["wilson_lb"], "window": None}
        for w in WINDOW_CANDIDATES:
            label_name = f"{group_col}_w{w}"
            new_col = f"{group_col}_custom"
            variant = add_fn(features, w, new_col)
            # eval_variant drops the OLD group_col by name; the candidate's
            # values live under the differently-named new_col, which
            # prepare_panel picks up automatically as just another feature
            # (it includes every column not on its exclusion list, by name)
            # -- no rename needed here, same convention test_more_window_
            # params.py's "_custom" columns already rely on.
            m = eval_variant(variant, labels, group_col, label_name)
            logger.info("[%s] window=%d roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                        group_col, w, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                        m["wilson_lb"] - baseline_metrics["wilson_lb"])
            results.append({"group": group_col, "window": w, **m})
            if m["wilson_lb"] > best["wilson_lb"]:
                best = {"wilson_lb": m["wilson_lb"], "window": w}
                variant_by_winner[group_col] = (w, variant, new_col)
        best_per_group[group_col] = best["window"]

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS (single-variable sweeps):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_relative_strength_window_results.csv", index=False)
    logger.info("Saved data/test_relative_strength_window_results.csv")

    winners = {g: w for g, w in best_per_group.items() if w is not None}
    if not winners:
        logger.info("=" * 100)
        logger.info("No group beat baseline individually -- current window=20 for both features already holds up.")
        logger.info("=" * 100)
        return

    logger.info("=" * 100)
    logger.info("Best candidates (to combine and re-check): %s", winners)
    logger.info("=" * 100)
    combined = features.copy()
    for group_col, w in winners.items():
        _, variant, new_col = variant_by_winner[group_col]
        combined = combined.drop(columns=[group_col], errors="ignore")
        combined = combined.merge(variant[["stock_code", "date", new_col]], on=["stock_code", "date"], how="left")
        combined = combined.rename(columns={new_col: group_col})
    combined_metrics = eval_variant(combined, labels, "", "combined_best")
    logger.info("[combined_best] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                combined_metrics["roc_auc"], combined_metrics["n"], combined_metrics["wins"],
                combined_metrics["precision"], combined_metrics["wilson_lb"],
                combined_metrics["wilson_lb"] - baseline_metrics["wilson_lb"])
    if combined_metrics["wilson_lb"] > baseline_metrics["wilson_lb"]:
        logger.info("Combined variant BEATS baseline -- worth considering for adoption.")
    else:
        logger.info("Combined variant does NOT beat baseline -- individual single-variable "
                     "'wins' above were sampling noise, not a real combined improvement. "
                     "Current window=20 for both features kept unchanged.")


if __name__ == "__main__":
    run()
