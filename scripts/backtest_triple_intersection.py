"""Does the INTERSECTION of Momentum Screener's validated_signal with
Swing/Turnaround's own model probability have a better historical win
rate than validated_signal alone?

IMPORTANT METHODOLOGY CAVEAT (unlike backtest_momentum_screener.py and
search_momentum_rules.py, which are leakage-free): Momentum Screener is a
fixed rule that never learned from the data, so replaying it at past
as-of dates carries no lookahead risk. Swing and Turnaround are TRAINED
models, though -- scoring them retroactively at old as-of dates using the
CURRENT, fully-trained-on-all-history booster is NOT the same as what
those models would have said if trained only on data up to that date
(that's what their own walk-forward validation already measures
properly, out-of-sample, and is why THIS script does not re-derive
Swing/Turnaround's own accuracy numbers). The risk here is narrower but
real: the current models were fit on rows that include these exact
historical instances, so there's some optimism baked into how well they
"recognize" a setup that, with hindsight, turned out to work. Treat this
result as suggestive/exploratory, not as rigorous new evidence the way
the rule-only Momentum backtest is.

Usage:
    python -m scripts.backtest_triple_intersection
"""
import numpy as np
import pandas as pd

from engine.model import build_feature_row, load_model_and_metadata, predict_probability
from engine.predict import MODEL_VERSION as SWING_MODEL_VERSION
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION
from features.momentum_screener import is_validated_signal
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import build_dataset, wilson_lower_bound

logger = get_logger("scripts.backtest_triple_intersection")

TURNAROUND_THRESHOLD = 0.85
SWING_BUY_THRESHOLD = 0.60


def load_full_feature_rows(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """pairs: list of (stock_code, date) -- fetches every feature_daily
    column (needed for the models' full feature vectors), for exactly
    those tickers, then filters to the exact dates in pandas."""
    engine = get_engine()
    codes = sorted({p[0] for p in pairs})
    placeholders = ",".join(f"'{c}'" for c in codes)
    df = pd.read_sql(f"SELECT * FROM feature_daily WHERE stock_code IN ({placeholders})", engine)
    wanted = pd.DataFrame(pairs, columns=["stock_code", "date"])
    wanted["date"] = pd.to_datetime(wanted["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df.merge(wanted, on=["stock_code", "date"], how="inner")


def run():
    df = build_dataset()
    df["validated_signal"] = df.apply(
        lambda r: is_validated_signal(r["regime"], r["macd_hist_slope_3d"], r["cmf_20"], r["rvol_20"]), axis=1,
    )
    validated = df[df["validated_signal"]].copy()
    logger.info("validated_signal rows: %d", len(validated))

    pairs = list(zip(validated["stock_code"], validated["as_of"].astype(str)))
    feat_rows = load_full_feature_rows(pairs)
    logger.info("Fetched %d full feature_daily rows for scoring", len(feat_rows))

    swing_booster, swing_meta = load_model_and_metadata(SWING_MODEL_VERSION)
    turnaround_booster, turnaround_meta = load_model_and_metadata(TURNAROUND_MODEL_VERSION)
    swing_watch_threshold = swing_meta["base_rate"]

    scored = []
    for _, row in feat_rows.iterrows():
        rd = row.to_dict()
        X_swing, _ = build_feature_row(rd, swing_meta["feature_cols"])
        X_turn, _ = build_feature_row(rd, turnaround_meta["feature_cols"])
        swing_prob = predict_probability(swing_booster, X_swing)
        turnaround_prob = predict_probability(turnaround_booster, X_turn)
        scored.append({"stock_code": row["stock_code"], "date": row["date"], "swing_prob": swing_prob, "turnaround_prob": turnaround_prob})
    scored_df = pd.DataFrame(scored)

    validated["date"] = pd.to_datetime(validated["as_of"])
    merged = validated.merge(scored_df, on=["stock_code", "date"], how="inner")
    logger.info("Merged (validated + scored): %d rows (some pairs may lack a feature_daily row -> dropped)", len(merged))

    def report(label, mask):
        sub = merged[mask]
        n = len(sub)
        if n == 0:
            logger.info("[%s] n=0 -- no historical instances", label)
            return
        wins = int(sub["outcome"].sum())
        wr = wins / n
        logger.info("[%s] n=%d win_rate=%.4f wilson_lb=%.4f", label, n, wr, wilson_lower_bound(wins, n))

    logger.info("=" * 70)
    logger.info("Momentum validated_signal ALONE (reference, already known): n=%d win_rate=%.4f",
                len(merged), merged["outcome"].mean())
    logger.info("=" * 70)
    report("+ Swing >= WATCH threshold (%.4f)" % swing_watch_threshold, merged["swing_prob"] >= swing_watch_threshold)
    report("+ Swing >= BUY threshold (0.60)", merged["swing_prob"] >= SWING_BUY_THRESHOLD)
    report("+ Turnaround >= POTENSIAL threshold (0.85)", merged["turnaround_prob"] >= TURNAROUND_THRESHOLD)
    report("+ Swing>=WATCH AND Turnaround>=POTENSIAL (triple intersection)",
           (merged["swing_prob"] >= swing_watch_threshold) & (merged["turnaround_prob"] >= TURNAROUND_THRESHOLD))
    report("+ Swing>=BUY AND Turnaround>=POTENSIAL",
           (merged["swing_prob"] >= SWING_BUY_THRESHOLD) & (merged["turnaround_prob"] >= TURNAROUND_THRESHOLD))


if __name__ == "__main__":
    run()
