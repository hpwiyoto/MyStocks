"""How accurate is each screener's TOP 2 daily pick, historically -- and
how does Rekomendasi Emitten's cross-tool consensus compare against
Swing, Turnaround, and Momentum Screener each picked on their own?

This is a different question from every other backtest in this project:
those measure "of everything that fired this rule, what fraction won" (a
recall-agnostic precision number over the WHOLE qualifying set). This one
replays each page's own on-screen RANKING at each historical as-of date
and keeps only the top 2 -- i.e. "if a user only ever acted on the two
names each tool put at the very top, how would that specific, small
selection have done." A tool can have excellent precision on its full
qualifying set yet a mediocre top-2 (or vice versa) -- these are genuinely
different properties, both worth knowing.

Common yardstick: the SAME 10-trading-day, +5%%/-2.5%% triple-barrier
outcome used everywhere else in this project (search_momentum_rules.py's
triple_barrier_outcome), for all four columns -- including Turnaround,
whose own NATIVE target is a 6-month regime hold, not a 10-day price pop.
This is a deliberate, disclosed choice, not an oversight: it is the exact
same common-yardstick convention Rekomendasi Emitten's own validation
(scripts/backtest_triple_intersection.py) already uses when scoring
Turnaround's contribution to the cross-tool consensus, so this stays
consistent with how the app already talks about that page. A low top-2
number for Turnaround under this yardstick does not mean Turnaround
"doesn't work" for its own actual 6-month purpose -- it means a 10-day
price pop was never what a turnaround-regime call was promising in the
first place.

Ranking replicated per page, exactly as the live app sorts:
  Swing        : decision tier (BUY, WATCH, AVOID) asc, probability desc.
                 decision uses engine.decision's real thresholds, including
                 the gocap floor and the IHSG-decline-conditional BUY bar
                 (replayed historically from real ^JKSE closes, not
                 hardcoded False).
  Turnaround   : candidates = regime in BAD_REGIMES that day only (exactly
                 like engine.predict_turnaround -- not scored otherwise);
                 decision tier (POTENSIAL, BELUM) asc, probability desc.
  Momentum     : features.momentum_screener's real sort key (validated_
                 signal desc, divergence_tier asc, regime_priority asc,
                 rsi_pivot_distance asc, divergence_age_days asc,
                 probability desc) -- "probability" here is Swing's, same
                 tiebreaker role it plays on the real page.
  Rekomendasi  : agreement_count desc, combined_score (swing_prob +
                 turnaround_prob) desc -- identical to
                 app/pages/5_Rekomendasi_Emitten.py.

Swing/Turnaround are scored with the CURRENT, fully-trained models at
every past as-of date -- same optimism caveat scripts/
backtest_triple_intersection.py already documents (their own walk-forward
validation is the leakage-free measurement of THEIR OWN accuracy; this
script is about which SELECTION policy on top of them works best, a
different question, but inherits that same caveat for any number that
involves their probabilities).

Usage:
    python -m scripts.backtest_top2_screeners
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from engine.decision import BUY_THRESHOLD, GOCAP_PRICE_FLOOR, IHSG_DECLINE_BUY_THRESHOLD
from engine.model import load_model_and_metadata
from engine.predict import MODEL_VERSION as SWING_MODEL_VERSION
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION
from features.momentum_screener import (
    DEFAULT_REGIME_PRIORITY,
    REGIME_PRIORITY,
    classify_macd_status,
    detect_bullish_divergence,
)
from features.regime import BAD_REGIMES
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from pipeline.yfinance_source import fetch_history
from scripts.search_momentum_rules import (
    AS_OF_STRIDE,
    HORIZON,
    LOOKBACK_DAYS,
    WARMUP_DATES,
    triple_barrier_outcome,
    wilson_lower_bound,
)
from scripts.test_strategy_6_criteria import AVWAP_WINDOW

logger = get_logger("scripts.backtest_top2_screeners")

TURNAROUND_THRESHOLD = 0.85


def load_ihsg_declining_by_date(dates: list) -> dict:
    """Real historical ^JKSE closes, one fetch for the whole backtest --
    replays pipeline.yfinance_source.is_ihsg_declining's exact comparison
    (today's close < close 20 trading days ago) for every as-of date,
    instead of assuming it was always False."""
    logger.info("Fetching full IHSG (^JKSE) history for the historical decline flag...")
    hist = fetch_history("^JKSE", period="max")
    close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index()
    declining = close < close.shift(20)
    out = {}
    for d in dates:
        ts = pd.Timestamp(d)
        idx = close.index.searchsorted(ts, side="right") - 1
        out[d] = bool(declining.iloc[idx]) if idx >= 20 else False
    return out


def load_full_panel() -> pd.DataFrame:
    engine = get_engine()
    logger.info("Loading full price_history + feature_daily (every column, every ticker/date)...")
    df = pd.read_sql(
        """
        SELECT ph.stock_code, ph.date, ph.close, ph.high, ph.low, ph.volume, fd.*
        FROM price_history ph
        JOIN feature_daily fd ON fd.stock_code = ph.stock_code AND fd.date = ph.date
        WHERE ph.source_provider = 'yfinance'
        ORDER BY ph.stock_code, ph.date
        """,
        engine,
    )
    df = df.loc[:, ~df.columns.duplicated()]
    logger.info("Loaded %d rows across %d tickers", len(df), df["stock_code"].nunique())
    return df


def build_feature_matrix(day_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Vectorized cross-sectional equivalent of engine.model.build_feature_row
    -- same preprocessing (has_similar_pattern, historical_win_rate
    imputation, regime one-hot), applied to a whole day's rows at once so
    scoring is one DMatrix/predict() call per as-of date, not one per
    ticker (900-ish tickers x ~110 as-of dates x 2 models otherwise)."""
    df = day_df.copy()
    df["has_similar_pattern"] = (df.get("similar_pattern_count", pd.Series(dtype=float)).fillna(0) > 0).astype(float)
    if "historical_win_rate" in df.columns:
        df["historical_win_rate"] = df["historical_win_rate"].fillna(0.5)
    else:
        df["historical_win_rate"] = 0.5
    for c in feature_cols:
        if c.startswith("regime_"):
            wanted = c[len("regime_"):]
            df[c] = (df["regime"] == wanted).astype(float)
        elif c not in df.columns:
            df[c] = float("nan")
    return df[feature_cols].astype(float)


def swing_decision_tier(probability: float, base_rate: float, close: float, buy_threshold: float) -> int:
    if close <= GOCAP_PRICE_FLOOR:
        return 2  # AVOID
    if probability >= buy_threshold:
        return 0  # BUY
    if probability >= base_rate:
        return 1  # WATCH
    return 2  # AVOID


def momentum_extra_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Same AVWAP/close_above_avwap computation as scripts.test_strategy_
    6_criteria's _add_new_indicators, kept local here to avoid depending on
    that script's full build_dataset() (which reloads its own copy of the
    panel and re-runs the as-of loop for a DIFFERENT purpose)."""
    close = g["close"].to_numpy(dtype=float)
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    volume = g["volume"].to_numpy(dtype=float)
    typical = (high + low + close) / 3
    n = len(close)
    avwap = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - AVWAP_WINDOW + 1)
        seg_close = close[start:i + 1]
        anchor = start + int(np.argmin(seg_close))
        seg_vol = volume[anchor:i + 1]
        vol_sum = seg_vol.sum()
        if vol_sum > 0:
            avwap[i] = (typical[anchor:i + 1] * seg_vol).sum() / vol_sum
    out = g.copy()
    out["avwap_from_low50"] = avwap
    out["close_above_avwap"] = close >= avwap
    return out


def run():
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    ticker_frames = {code: momentum_extra_indicators(g.sort_values("date").reset_index(drop=True)) for code, g in df.groupby("stock_code")}
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates = sorted(df["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates", len(as_of_dates))

    ihsg_declining_by_date = load_ihsg_declining_by_date(as_of_dates)

    swing_booster, swing_meta = load_model_and_metadata(SWING_MODEL_VERSION)
    turnaround_booster, turnaround_meta = load_model_and_metadata(TURNAROUND_MODEL_VERSION)
    swing_base_rate = swing_meta["base_rate"]
    swing_feature_cols = swing_meta["feature_cols"]
    turnaround_feature_cols = turnaround_meta["feature_cols"]

    picks = []  # rows: screener, as_of, rank, stock_code, outcome

    for n_done, as_of in enumerate(as_of_dates):
        day_rows = []
        for code, g in ticker_frames.items():
            idx = as_of_idx_by_ticker[code].get(as_of)
            if idx is None or idx < 30:
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            latest = g.iloc[idx]
            if pd.isna(latest.get("rsi_14")) or pd.isna(latest.get("macd_hist")):
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue
            row = latest.to_dict()
            row["stock_code"] = code
            row["outcome"] = outcome
            # Divergence needs the trailing LOOKBACK_DAYS window (matches
            # every other Momentum backtest's no-lookahead convention).
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window_df = g.iloc[start:idx + 1]
            div = detect_bullish_divergence(window_df)
            row.update(div)
            row["macd_status"] = classify_macd_status(window_df["macd_hist"])
            day_rows.append(row)

        if len(day_rows) < 2:
            continue
        day_df = pd.DataFrame(day_rows)

        # --- Swing: score everyone, tier by the real decision() rules ---
        buy_threshold = IHSG_DECLINE_BUY_THRESHOLD if ihsg_declining_by_date.get(as_of) else BUY_THRESHOLD
        X_swing = build_feature_matrix(day_df, swing_feature_cols)
        swing_prob = swing_booster.predict(xgb.DMatrix(X_swing))
        day_df["swing_prob"] = swing_prob
        day_df["swing_tier"] = [
            swing_decision_tier(p, swing_base_rate, c, buy_threshold)
            for p, c in zip(day_df["swing_prob"], day_df["close"])
        ]

        # --- Turnaround: only bearish/bottoming candidates, same as engine.predict_turnaround ---
        ta_mask = day_df["regime"].isin(BAD_REGIMES)
        day_df["turnaround_prob"] = np.nan
        if ta_mask.any():
            X_turn = build_feature_matrix(day_df.loc[ta_mask], turnaround_feature_cols)
            day_df.loc[ta_mask, "turnaround_prob"] = turnaround_booster.predict(xgb.DMatrix(X_turn))
        day_df["turnaround_tier"] = np.where(day_df["turnaround_prob"] >= TURNAROUND_THRESHOLD, 0, 1)

        # --- Momentum: real sort key, real validated_signal ---
        avwap_ok = day_df["close_above_avwap"] == True  # noqa: E712 -- explicit vs NaN
        day_df["validated_signal"] = (
            (day_df["regime"] == "bottoming")
            & (day_df["macd_hist_slope_3d"] > 0)
            & (day_df["cmf_20"] < 0)
            & (day_df["rvol_20"] >= 0.8)
            & avwap_ok
        )
        day_df["divergence_tier"] = np.select(
            [day_df["divergence_rsi"] & day_df["divergence_macd"], day_df["divergence_rsi"] | day_df["divergence_macd"]],
            [0, 1], default=2,
        )
        day_df["regime_priority"] = day_df["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
        day_df["rsi_pivot_distance"] = (day_df["rsi_14"] - 50).abs()

        # --- Rekomendasi Emitten: same hit/agreement definition as the real page ---
        day_df["swing_hit"] = day_df["swing_tier"] <= 1  # BUY or WATCH
        day_df["turnaround_hit"] = day_df["turnaround_tier"] == 0
        day_df["momentum_hit"] = day_df["validated_signal"].fillna(False)
        day_df["agreement_count"] = day_df[["swing_hit", "turnaround_hit", "momentum_hit"]].sum(axis=1).astype(int)
        day_df["combined_score"] = day_df["swing_prob"].fillna(0) + day_df["turnaround_prob"].fillna(0)

        rankings = {
            "Swing": day_df.sort_values(["swing_tier", "swing_prob"], ascending=[True, False]),
            "Turnaround": day_df.loc[ta_mask].sort_values(["turnaround_tier", "turnaround_prob"], ascending=[True, False]),
            "Momentum": day_df.sort_values(
                ["validated_signal", "divergence_tier", "regime_priority", "rsi_pivot_distance", "divergence_age_days", "swing_prob"],
                ascending=[False, True, True, True, True, False], na_position="last",
            ),
            # >=2, not >=1 -- matches the real page's DEFAULT sidebar filter
            # (st.radio(..., index=1) selects "Minimal 2 dari 3" by default,
            # not the loosest "1 dari 3" option), so this is what a user
            # actually sees at the top of the page without touching any
            # filter, not an artificially broadened candidate pool.
            "Rekomendasi Emitten": day_df[day_df["agreement_count"] >= 2].sort_values(
                ["agreement_count", "combined_score"], ascending=[False, False],
            ),
        }
        for screener, ranked in rankings.items():
            for rank, (_, r) in enumerate(ranked.head(2).iterrows(), start=1):
                picks.append({
                    "screener": screener, "as_of": as_of, "rank": rank, "stock_code": r["stock_code"], "outcome": r["outcome"],
                    # Tracked so the Momentum row can be split into
                    # "actually validated_signal that day" vs "just
                    # whatever ranked highest among the non-validated
                    # majority (validated_signal fires on ~1.25% of all
                    # ticker-days, so most days have 0-1 real candidates
                    # and the 2nd/both slots fall back to non-validated
                    # ones)" -- otherwise a lower top-2 number here reads
                    # as "the validated rule is weak", when it may just be
                    # "most of these picks weren't the validated rule at
                    # all".
                    "was_validated_signal": bool(r.get("validated_signal", False)),
                })

        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d picks recorded so far)", n_done + 1, len(as_of_dates), len(picks))

    picks_df = pd.DataFrame(picks)
    logger.info("Done. %d total top-2 picks recorded across all screeners.", len(picks_df))

    logger.info("=" * 70)
    logger.info("RESULT: top-2-per-day pooled win rate, by screener")
    logger.info("=" * 70)
    rows = []
    for screener, g in picks_df.groupby("screener"):
        n_days = g["as_of"].nunique()
        n = len(g)
        wins = int(g["outcome"].sum())
        wr = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else 0.0
        rows.append({"screener": screener, "n_days_with_2_picks": n_days, "n_picks": n, "win_rate": wr, "wilson_lb": lb})
        for rank, rg in g.groupby("rank"):
            rn, rw = len(rg), int(rg["outcome"].sum())
            logger.info("  [%s] rank=%d n=%d win_rate=%.4f", screener, rank, rn, rw / rn if rn else float("nan"))
        if screener == "Momentum":
            was_v = g["was_validated_signal"]
            n_v, n_nv = int(was_v.sum()), int((~was_v).sum())
            wr_v = g.loc[was_v, "outcome"].mean() if n_v else float("nan")
            wr_nv = g.loc[~was_v, "outcome"].mean() if n_nv else float("nan")
            logger.info("  [Momentum] split: validated_signal picks n=%d win_rate=%.4f | non-validated (just top-ranked) n=%d win_rate=%.4f",
                        n_v, wr_v, n_nv, wr_nv)
    summary = pd.DataFrame(rows).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
