"""Wider, systematic grid search building on scripts/search_momentum_rules.py's
finding (regime=bottoming + momentum menguat + RVOL>=1.2, 40.2% win rate).
That script hand-picked ~40 candidates; this one sweeps a real grid across
regime x RVOL x momentum-direction x CMF x RSI-band (several thousand
combinations) against the SAME cached dataset, to check whether hand-picked
intuition missed a genuinely stronger combination.

Reuses build_dataset() unmodified (identical no-lookahead replay + +5%/
-2.5%/10-day grading, same 109 as-of dates) so results stay directly
comparable to every number already reported.

Usage:
    python -m scripts.grid_search_momentum_rules
"""
import itertools

import pandas as pd

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import build_dataset, wilson_lower_bound

logger = get_logger("scripts.grid_search_momentum_rules")

MIN_N = 100  # ignore combinations too small to say anything meaningful

REGIME_GROUPS = {
    "any": None,
    "bottoming": ["bottoming"],
    "bearish": ["bearish"],
    "sideways": ["sideways"],
    "accumulation": ["accumulation"],
    "bullish": ["bullish"],
    "early_reversal": ["early_reversal"],
    "bottoming+bearish": ["bottoming", "bearish"],
    "bottoming+sideways": ["bottoming", "sideways"],
    "bottoming+bearish+sideways": ["bottoming", "bearish", "sideways"],
}
RVOL_THRESHOLDS = {"any": None, ">=0.8": 0.8, ">=1.0": 1.0, ">=1.2": 1.2, ">=1.5": 1.5, ">=2.0": 2.0, "<1.0": "low"}
MOMENTUM_MODES = {"any": None, "menguat": "up", "melemah": "down"}
CMF_MODES = {"any": None, ">0": 0.0, ">0.1": 0.1, "<0": "neg"}
RSI_BANDS = {
    "any": None, "<30": (0, 30), "30-50": (30, 50), "40-60": (40, 60),
    "20-60": (20, 60), "0-50": (0, 50), "50-70": (50, 70),
}


def build_mask(df: pd.DataFrame, regime_key, rvol_key, mom_key, cmf_key, rsi_key) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    regimes = REGIME_GROUPS[regime_key]
    if regimes is not None:
        mask &= df["regime"].isin(regimes)

    rvol_val = RVOL_THRESHOLDS[rvol_key]
    if rvol_val == "low":
        mask &= df["rvol_20"] < 1.0
    elif rvol_val is not None:
        mask &= df["rvol_20"] >= rvol_val

    mom_val = MOMENTUM_MODES[mom_key]
    if mom_val == "up":
        mask &= df["macd_hist_slope_3d"] > 0
    elif mom_val == "down":
        mask &= df["macd_hist_slope_3d"] < 0

    cmf_val = CMF_MODES[cmf_key]
    if cmf_val == "neg":
        mask &= df["cmf_20"] < 0
    elif cmf_val is not None:
        mask &= df["cmf_20"] > cmf_val

    rsi_band = RSI_BANDS[rsi_key]
    if rsi_band is not None:
        mask &= df["rsi_14"].between(*rsi_band)

    return mask


def run():
    df = build_dataset()
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(df), null_rate)
    logger.info("=" * 70)

    combos = list(itertools.product(REGIME_GROUPS, RVOL_THRESHOLDS, MOMENTUM_MODES, CMF_MODES, RSI_BANDS))
    logger.info("Evaluating %d grid combinations...", len(combos))

    results = []
    for regime_key, rvol_key, mom_key, cmf_key, rsi_key in combos:
        mask = build_mask(df, regime_key, rvol_key, mom_key, cmf_key, rsi_key)
        n = int(mask.sum())
        if n < MIN_N:
            continue
        wins = int(df.loc[mask, "outcome"].sum())
        wr = wins / n
        results.append({
            "regime": regime_key, "rvol": rvol_key, "momentum": mom_key, "cmf": cmf_key, "rsi": rsi_key,
            "n": n, "win_rate": wr, "lift_vs_null": wr / null_rate, "wilson_lb": wilson_lower_bound(wins, n),
        })

    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 30)

    logger.info("=" * 70)
    logger.info("TOP 25 by Wilson 95%% lower bound (n>=%d), previous best was "
                "bottoming+momentum+RVOL>=1.2 (win_rate=0.4016, wilson_lb=0.3643)", MIN_N)
    logger.info("=" * 70)
    logger.info("\n%s", results_df.head(25).to_string(index=False))

    logger.info("=" * 70)
    logger.info("TOP 15 by raw win_rate (n>=%d) -- less conservative view, higher variance", MIN_N)
    logger.info("=" * 70)
    logger.info("\n%s", results_df.sort_values("win_rate", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    run()
