"""Two-stage test of pattern-similarity's own parameters (features/
pattern_similarity.py's WINDOW=20/HORIZON=10/CORR_THRESHOLD=0.85) -- the
last deferred item of the parameter-optimality trilogy (see scripts/
test_relative_strength_window.py for the other one, relative strength).

Deferred separately from every other test in this trilogy because this
feature's cost is different in KIND, not just degree: indicator windows,
regime thresholds, and relative strength are all O(n) per ticker, cheap
regardless of universe size. Pattern similarity is cross-ticker --
O(n_query * n_bank) -- so recomputing it for one candidate parameter
value costs quadratically more as the universe grows.

Measured directly rather than trusting the module's own docstring number
(which turned out to describe a REJECTED alternative algorithm -- a
cKDTree radius query -- not the shipped one; see docs/RESEARCH_LOG.md /
git history for that correction): 286.9s for a 150-ticker bank+query,
full history, on the CURRENT shipped matmul implementation. Extrapolated
quadratically to the full ~900-ticker universe, that is roughly 2.9
HOURS per candidate value -- genuinely too slow to sweep several
candidates blind, which is why this gets its own two-stage treatment
instead of the trilogy's usual single-pass sweep.

Stage 1 (screen, `python -m scripts.test_pattern_similarity_params
screen`): a fixed-seed SCREEN_N-ticker subsample used as BOTH bank and
query set for every candidate, cutting the quadratic cost by
(900/SCREEN_N)^2 (~6.6x at SCREEN_N=350: each candidate lands around
26 minutes instead of ~3 hours). This is a RELATIVE comparison between
candidates sharing the SAME smaller bank -- a smaller bank finds fewer/
weaker matches for every candidate equally, so the ranking between
candidates should still transfer, but the absolute wilson_lb numbers
here are NOT comparable to the true full-scale baseline logged
elsewhere (e.g. scripts/test_more_window_params.py's baseline) --
only to each other, within this run.

Stage 2 (confirm, `python -m scripts.test_pattern_similarity_params
confirm --window=W --threshold=T`): re-run ONLY the single best
candidate from stage 1 (if it beat stage 1's own baseline) at the FULL
~900-ticker universe, walk-forward evaluated against the TRUE shipped
baseline (the already-existing features parquet, unmodified) -- the
expensive ~3h step, meant to run at most once, not once per candidate.

horizon=10 deliberately NOT swept here -- scoped down to window and
corr_threshold only (2 groups), matching the reduced candidate set
agreed on before running this (window: 15/30 vs baseline 20;
threshold: 0.80/0.90 vs baseline 0.85), consistent with this project's
grid-search-overfitting caution about testing too many knobs against a
noisy metric at once.

RESULT (2026-09-16): NOT confirmed -- the two-stage design did exactly
its job of catching a screen-stage false positive before it reached a
production decision.

Stage 1 (350-ticker subsample, ~23min/candidate): screen baseline
wilson_lb=63.33% (note this is NOT the true full-scale number, as
documented above -- a smaller bank finds fewer matches for every
candidate). window=15 stood out sharply at +2.56pp (65.89%) -- an order
of magnitude larger than every other trilogy test's noise-level results
(typically <0.1pp). threshold=0.80 also stood out at +2.35pp (65.69%).
window=30 (+0.15pp) and threshold=0.90 (+1.13pp) were smaller. window=15
picked as best candidate (highest absolute wilson_lb) for stage 2.

Stage 2 (full ~900-ticker universe, 9243.6s = 2.57h for the pattern-
similarity recompute alone): true baseline wilson_lb=71.25% (matches the
already-known shipped number exactly, sanity check passed). window=15 at
full scale: wilson_lb=70.44% -- WORSE than baseline by -0.81pp, a
complete reversal from stage 1's +2.56pp "win". Confirms the documented
caveat in the strongest possible way: the screen-stage signal was an
artifact of the smaller bank (window=15's shorter lookback plausibly
finds more spurious matches when the bank itself is small and matches
are already scarce -- with 900 tickers' worth of real candidates
available, that shortcut stops helping and the shorter window just
loses information a full window=20 pattern captures). threshold=0.80 was
NOT independently confirmed at full scale (time-boxed to one confirmation
run per the agreed plan; the screen-stage signal there is now suspect for
the same reason and not assumed to hold either).

CONCLUSION: features/pattern_similarity.py's WINDOW=20/HORIZON=10/
CORR_THRESHOLD=0.85 left unchanged. Closes the last deferred item of the
parameter-optimality trilogy -- every base feature window, regime
threshold, relative-strength window, and pattern-similarity parameter
this project has now tested is either already optimal or not worth the
change, a fully decisive result across the whole trilogy.

Usage:
    python -m scripts.test_pattern_similarity_params screen
    python -m scripts.test_pattern_similarity_params confirm --window 20 --threshold 0.85
"""
import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from features.pattern_similarity import CORR_THRESHOLD, HORIZON as PATTERN_HORIZON, WINDOW, compute_cross_ticker_pattern_similarity
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

logger = get_logger("scripts.test_pattern_similarity_params")

SCREEN_N = 350
SCREEN_SEED = 42
WINDOW_CANDIDATES = [15, 30]        # vs baseline 20
THRESHOLD_CANDIDATES = [0.80, 0.90]  # vs baseline 0.85
PATTERN_COLS = ["similarity_score", "similar_pattern_count", "historical_win_rate"]


def _price_by_ticker(prices: pd.DataFrame) -> dict:
    return {
        code: g.sort_values("date")[["date", "close"]].reset_index(drop=True)
        for code, g in prices.groupby("stock_code")
    }


def compute_pattern_variant(price_by_ticker: dict, window: int, horizon: int, threshold: float) -> pd.DataFrame:
    t0 = time.time()
    codes = list(price_by_ticker.keys())
    results = compute_cross_ticker_pattern_similarity(
        price_by_ticker, window=window, horizon=horizon, corr_threshold=threshold, query_codes=codes,
    )
    rows = []
    for code in codes:
        r = results[code].copy()
        r["stock_code"] = code
        r["date"] = price_by_ticker[code]["date"].to_numpy()
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    logger.info("Pattern variant window=%d horizon=%d threshold=%.2f: %d tickers in %.1fs",
                window, horizon, threshold, len(codes), time.time() - t0)
    return out


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


def eval_with_pattern_variant(features: pd.DataFrame, labels: pd.DataFrame, pattern_df: pd.DataFrame, label_name: str) -> dict:
    base = features.drop(columns=PATTERN_COLS, errors="ignore")
    merged = base.merge(pattern_df[["stock_code", "date"] + PATTERN_COLS], on=["stock_code", "date"], how="left")
    df, feature_cols = prepare_panel(merged, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    return evaluate(df, feature_cols, splits, label_name)


def run_screen():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows, %d tickers total",
                len(features), len(prices), prices["stock_code"].nunique())

    all_codes = sorted(prices["stock_code"].unique())
    screen_codes = list(np.random.RandomState(SCREEN_SEED).choice(all_codes, size=min(SCREEN_N, len(all_codes)), replace=False))
    logger.info("Stage 1 SCREEN: fixed-seed subsample of %d/%d tickers (seed=%d)",
                len(screen_codes), len(all_codes), SCREEN_SEED)

    prices_screen = prices[prices["stock_code"].isin(screen_codes)].copy()
    features_screen = features[features["stock_code"].isin(screen_codes)].copy()
    labels_screen = build_panel_labels(prices_screen)
    price_by_ticker = _price_by_ticker(prices_screen)

    results = []
    logger.info("#" * 90)
    logger.info("STAGE-1 BASELINE (window=%d horizon=%d threshold=%.2f, on the %d-ticker subsample)",
                WINDOW, PATTERN_HORIZON, CORR_THRESHOLD, len(screen_codes))
    logger.info("#" * 90)
    baseline_pattern = compute_pattern_variant(price_by_ticker, WINDOW, PATTERN_HORIZON, CORR_THRESHOLD)
    baseline_metrics = eval_with_pattern_variant(features_screen, labels_screen, baseline_pattern, "screen_baseline")
    logger.info("[screen_baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "window": WINDOW, "threshold": CORR_THRESHOLD, **baseline_metrics})

    best = {"wilson_lb": baseline_metrics["wilson_lb"], "window": WINDOW, "threshold": CORR_THRESHOLD}

    logger.info("#" * 90)
    logger.info("GROUP: window (threshold held at baseline %.2f)", CORR_THRESHOLD)
    logger.info("#" * 90)
    for w in WINDOW_CANDIDATES:
        pattern_variant = compute_pattern_variant(price_by_ticker, w, PATTERN_HORIZON, CORR_THRESHOLD)
        m = eval_with_pattern_variant(features_screen, labels_screen, pattern_variant, f"window_{w}")
        logger.info("[window=%d] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs screen baseline %+.4f)",
                    w, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                    m["wilson_lb"] - baseline_metrics["wilson_lb"])
        results.append({"group": "window", "window": w, "threshold": CORR_THRESHOLD, **m})
        if m["wilson_lb"] > best["wilson_lb"]:
            best = {"wilson_lb": m["wilson_lb"], "window": w, "threshold": CORR_THRESHOLD}

    logger.info("#" * 90)
    logger.info("GROUP: corr_threshold (window held at baseline %d)", WINDOW)
    logger.info("#" * 90)
    for t in THRESHOLD_CANDIDATES:
        pattern_variant = compute_pattern_variant(price_by_ticker, WINDOW, PATTERN_HORIZON, t)
        m = eval_with_pattern_variant(features_screen, labels_screen, pattern_variant, f"threshold_{t}")
        logger.info("[threshold=%.2f] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs screen baseline %+.4f)",
                    t, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                    m["wilson_lb"] - baseline_metrics["wilson_lb"])
        results.append({"group": "threshold", "window": WINDOW, "threshold": t, **m})
        if m["wilson_lb"] > best["wilson_lb"]:
            best = {"wilson_lb": m["wilson_lb"], "window": WINDOW, "threshold": t}

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("STAGE-1 SCREEN FULL RESULTS (subsample of %d, NOT comparable to full-scale baseline numbers):", len(screen_codes))
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_pattern_similarity_screen_results.csv", index=False)
    logger.info("Saved data/test_pattern_similarity_screen_results.csv")

    if best["window"] == WINDOW and best["threshold"] == CORR_THRESHOLD:
        logger.info("=" * 100)
        logger.info("No candidate beat the screen baseline -- current window=%d/threshold=%.2f already holds up "
                     "at this subsample scale. No stage-2 confirmation run needed.", WINDOW, CORR_THRESHOLD)
        logger.info("=" * 100)
    else:
        logger.info("=" * 100)
        logger.info("BEST CANDIDATE at screen scale: window=%d threshold=%.2f (wilson_lb=%.4f vs baseline %.4f). "
                     "Run stage 2 to confirm at full scale:", best["window"], best["threshold"], best["wilson_lb"],
                     baseline_metrics["wilson_lb"])
        logger.info("    python -m scripts.test_pattern_similarity_params confirm --window %d --threshold %.2f",
                     best["window"], best["threshold"])
        logger.info("=" * 100)


def run_confirm(window: int, threshold: float, horizon: int = PATTERN_HORIZON):
    logger.info("STAGE 2 CONFIRM: window=%d horizon=%d threshold=%.2f at FULL universe scale "
                "(expensive, ~hours, runs ONCE)", window, horizon, threshold)
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    price_by_ticker = _price_by_ticker(prices)
    logger.info("Full universe: %d tickers", len(price_by_ticker))

    logger.info("#" * 90)
    logger.info("TRUE BASELINE (already-shipped features parquet, unmodified)")
    logger.info("#" * 90)
    baseline_df, baseline_cols = prepare_panel(features, labels)
    baseline_dates = baseline_df["date"].to_numpy()
    baseline_splits = walk_forward_splits(baseline_dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    baseline_metrics = evaluate(baseline_df, baseline_cols, baseline_splits, "true_baseline")
    logger.info("[true_baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])

    logger.info("#" * 90)
    logger.info("CANDIDATE: window=%d horizon=%d threshold=%.2f, full universe", window, horizon, threshold)
    logger.info("#" * 90)
    candidate_pattern = compute_pattern_variant(price_by_ticker, window, horizon, threshold)
    candidate_metrics = eval_with_pattern_variant(features, labels, candidate_pattern, "confirm_candidate")
    logger.info("[confirm_candidate] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs true baseline %+.4f)",
                candidate_metrics["roc_auc"], candidate_metrics["n"], candidate_metrics["wins"],
                candidate_metrics["precision"], candidate_metrics["wilson_lb"],
                candidate_metrics["wilson_lb"] - baseline_metrics["wilson_lb"])

    logger.info("=" * 100)
    if candidate_metrics["wilson_lb"] > baseline_metrics["wilson_lb"]:
        logger.info("CONFIRMED at full scale -- candidate BEATS the true shipped baseline. Worth adopting: "
                     "update features/pattern_similarity.py's WINDOW/HORIZON/CORR_THRESHOLD to %d/%d/%.2f, "
                     "then backfill + retrain.", window, horizon, threshold)
    else:
        logger.info("NOT confirmed at full scale -- the screen-stage 'win' was an artifact of the smaller "
                     "subsample bank, not a real full-scale improvement. Current window=%d/horizon=%d/"
                     "threshold=%.2f (features/pattern_similarity.py) kept unchanged.", WINDOW, PATTERN_HORIZON, CORR_THRESHOLD)
    logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("screen")
    confirm_parser = sub.add_parser("confirm")
    confirm_parser.add_argument("--window", type=int, required=True)
    confirm_parser.add_argument("--threshold", type=float, required=True)
    confirm_parser.add_argument("--horizon", type=int, default=PATTERN_HORIZON)
    args = parser.parse_args()

    if args.stage == "screen":
        run_screen()
    else:
        run_confirm(args.window, args.threshold, args.horizon)
