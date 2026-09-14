"""Does the system have enough information to warn BEFORE a held BUY
position hits its full stop-loss -- prompted directly by a user question
("apakah system kita bisa memberikan informasi dini saat prediksi
menyimpang... sebelum menyentuh angka stop loss, system kita sudah tahu
duluan"). Same "test before adopt" posture as every other feature in
this project: this backtests a CANDIDATE early-warning rule against
history before it ships as a real signal on tracked positions, rather
than assuming a plausible-sounding technical rule actually helps.

Candidate rule (uses only already-validated Swing model inputs -- RSI
slope, CMF, EMA9 position -- not new indicators): on any day the
position is ALREADY underwater (close < entry_price), flag a warning if
some number of these three weakness signals also hold that day:
  - close < ema_9            (short-term trend broken)
  - rsi_slope_3d < 0         (momentum weakening)
  - cmf_20 < 0               (money flowing out)
Swept at ALL_3 (strict) vs ANY_2 (looser) required to fire.

Simulated on the historical walk-forward BUY signals for the CURRENTLY
SHIPPED Swing config (target=10%/stop=5%/horizon=5d, threshold=0.60,
same pooled simulation as scripts/check_suspension_risk_v2.py). For each
signal, walks the following HORIZON trading days, resolves win/loss/
timeout with the same first-touch/stop-priority-on-tie convention as
scripts/train_v5.py's triple_barrier_label, and checks whether the
warning rule fires on any day STRICTLY BEFORE that resolution.

Two questions, both required for this to be worth shipping:
1. On LOSING trades (stop hit): does the warning fire early, and how
   much smaller is the loss if you'd exited at the warning day's close
   instead of waiting for the actual stop?
2. On WINNING/TIMEOUT trades: how often does the warning fire ANYWAY
   (false alarm), and what's the opportunity cost of exiting there
   instead of letting the trade play out?

RESULT (2026-09-14, against the shipped 10%/-5%/5-trading-day config):
ADOPTED as ANY_2 (>=2 of 3 signals), NOT as ALL_3 -- ALL_3 never once
fired in 762 historical signals (too strict for a 5-day window). ANY_2:
caught 11/214 losses early (5.1%), avg loss saved when caught +2.99pp
(i.e. exiting there instead of at the full stop averaged ~-2% instead of
the full -5%); false-alarmed on only 1/548 non-losses (0.2%), though
that one alarm was expensive (+11.82pp of upside given up).

IMPORTANT CAVEAT, not a rejection but a real limit on how much this can
ever help under the current SHORT 5-day horizon: 147 of 214 losses
(68.7%) hit the stop on DAY 1 itself -- there is no PRIOR day to warn on
at all for most losses under this config, a single-session gap/crash,
not a gradual slide a multi-day rule could ever catch. The 5.1% overall
catch rate is really "16% of the 31% of losses that unfold over 2+ days"
-- worth stating honestly rather than implying broader coverage than it
has. A longer-horizon config (e.g. the 10-day siblings in
engine.swing_configs) mechanically gives this rule more days to work
with, though that hasn't been separately backtested here.

Shipped in engine/early_warning.py + app/pages/3_Posisi_Saya.py's
position-tracking feature -- computed live per tracked position, not
folded into the Swing model itself (this is a monitoring rule for a
position already taken, not a training feature).

Usage:
    python -m scripts.test_early_warning_rule
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    BUY_THRESHOLD,
    HORIZON,
    NUM_BOOST_ROUND,
    STOP_PCT,
    TARGET_PCT,
    XGB_PARAMS,
    walk_forward_splits,
)
from scripts.tune_v5 import _load_panel as load_panel

logger = get_logger("scripts.test_early_warning_rule")

WARNING_COLS = ["ema_9", "rsi_slope_3d", "cmf_20"]  # "close" comes from the PRICES parquet, not features
RULE_VARIANTS = {"ALL_3": 3, "ANY_2": 2}


def run_walk_forward_pooled(df, feature_cols, splits, xgb_params) -> pd.DataFrame:
    pooled = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test = df.loc[test_mask, feature_cols]
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        fold_df = pd.DataFrame({"fold": fold_i, "prob": prob, "stock_code": df.loc[test_mask, "stock_code"].to_numpy(), "date": df.loc[test_mask, "date"].to_numpy()})
        pooled.append(fold_df)
        logger.info("fold %d done: n_test=%d", fold_i, len(X_test))
    return pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()


def simulate_positions(buy_signals: pd.DataFrame, prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """For each BUY signal, walk the next HORIZON trading days for that
    ticker and record: outcome (win/loss/timeout), the day it resolved,
    and per-day warning-rule inputs -- so the threshold sweep below can
    try both rule variants without re-walking the price history twice."""
    prices = prices.sort_values(["stock_code", "date"]).reset_index(drop=True)
    features = features.sort_values(["stock_code", "date"]).reset_index(drop=True)
    price_idx = {code: g.reset_index(drop=True) for code, g in prices.groupby("stock_code")}
    feat_idx = {code: g.set_index("date")[WARNING_COLS] for code, g in features.groupby("stock_code")}

    rows = []
    for _, sig in buy_signals.iterrows():
        code, entry_date = sig["stock_code"], sig["date"]
        pg = price_idx.get(code)
        if pg is None:
            continue
        entry_pos = pg.index[pg["date"] == entry_date]
        if len(entry_pos) == 0:
            continue
        i0 = entry_pos[0]
        if i0 + HORIZON >= len(pg):
            continue
        entry_price = pg.loc[i0, "close"]
        target_price = entry_price * (1 + TARGET_PCT)
        stop_price = entry_price * (1 - STOP_PCT)
        fg = feat_idx.get(code)

        outcome, resolve_day, resolve_price = "timeout", HORIZON, pg.loc[i0 + HORIZON, "close"]
        day_flags = []  # (day, n_signals_true, close, underwater)
        for t in range(1, HORIZON + 1):
            row = pg.loc[i0 + t]
            stop_hit = row["low"] <= stop_price
            target_hit = row["high"] >= target_price
            if stop_hit:
                outcome, resolve_day, resolve_price = "loss", t, stop_price
            elif target_hit:
                outcome, resolve_day, resolve_price = "win", t, target_price
            if fg is not None and row["date"] in fg.index:
                frow = fg.loc[row["date"]]
                day_close = row["close"]
                underwater = day_close < entry_price
                n_true = int(pd.notna(frow["ema_9"]) and day_close < frow["ema_9"]) + \
                          int(pd.notna(frow["rsi_slope_3d"]) and frow["rsi_slope_3d"] < 0) + \
                          int(pd.notna(frow["cmf_20"]) and frow["cmf_20"] < 0)
                day_flags.append({"day": t, "n_signals": n_true, "close": day_close, "underwater": underwater})
            if stop_hit or target_hit:
                break

        rows.append({
            "stock_code": code, "entry_date": entry_date, "entry_price": entry_price,
            "outcome": outcome, "resolve_day": resolve_day, "resolve_price": resolve_price,
            "day_flags": day_flags,
        })
    return pd.DataFrame(rows)


def evaluate_rule(positions: pd.DataFrame, min_signals: int) -> dict:
    losses, loss_caught, loss_pct_saved = 0, 0, []
    non_losses, false_alarms, opportunity_cost_pct = 0, 0, []

    for _, pos in positions.iterrows():
        warning_day = next(
            (f for f in pos["day_flags"] if f["underwater"] and f["n_signals"] >= min_signals and f["day"] < pos["resolve_day"]),
            None,
        )
        if pos["outcome"] == "loss":
            losses += 1
            if warning_day is not None:
                loss_caught += 1
                actual_loss_pct = (pos["resolve_price"] - pos["entry_price"]) / pos["entry_price"] * 100
                early_exit_loss_pct = (warning_day["close"] - pos["entry_price"]) / pos["entry_price"] * 100
                # early (closer to 0, e.g. -2) minus actual (more negative, e.g. -5) = +3:
                # positive means the early exit lost LESS -- pp of loss avoided.
                loss_pct_saved.append(early_exit_loss_pct - actual_loss_pct)
        else:
            non_losses += 1
            if warning_day is not None:
                false_alarms += 1
                actual_return_pct = (pos["resolve_price"] - pos["entry_price"]) / pos["entry_price"] * 100
                early_exit_return_pct = (warning_day["close"] - pos["entry_price"]) / pos["entry_price"] * 100
                opportunity_cost_pct.append(actual_return_pct - early_exit_return_pct)

    return {
        "losses": losses, "loss_caught": loss_caught,
        "loss_caught_pct": loss_caught / losses * 100 if losses else 0.0,
        "avg_loss_pct_saved": float(np.mean(loss_pct_saved)) if loss_pct_saved else 0.0,
        "non_losses": non_losses, "false_alarms": false_alarms,
        "false_alarm_pct": false_alarms / non_losses * 100 if non_losses else 0.0,
        "avg_opportunity_cost_pct": float(np.mean(opportunity_cost_pct)) if opportunity_cost_pct else 0.0,
    }


def run():
    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    pooled = run_walk_forward_pooled(df, feature_cols, splits, XGB_PARAMS)
    logger.info("Pooled OOS rows: %d", len(pooled))

    buy_signals = pooled[pooled["prob"] >= BUY_THRESHOLD][["stock_code", "date"]].copy()
    logger.info("Historical BUY signals (walk-forward, prob>=%.2f): %d", BUY_THRESHOLD, len(buy_signals))

    prices = pd.read_parquet("data/export_for_colab_prices.parquet")
    features = pd.read_parquet("data/export_for_colab_features.parquet")

    positions = simulate_positions(buy_signals, prices, features)
    logger.info("Simulated %d positions (some dropped: insufficient forward data)", len(positions))
    logger.info("Outcome distribution: %s", positions["outcome"].value_counts().to_dict())

    logger.info("=" * 90)
    for variant, min_signals in RULE_VARIANTS.items():
        r = evaluate_rule(positions, min_signals)
        logger.info(
            "[%s, need>=%d/3 weakness signals while underwater] "
            "Losses: %d, caught early: %d (%.1f%%), avg loss saved if caught: %+.2f pp | "
            "Non-losses: %d, false alarms: %d (%.1f%%), avg opportunity cost if false-alarmed: %+.2f pp",
            variant, min_signals, r["losses"], r["loss_caught"], r["loss_caught_pct"], r["avg_loss_pct_saved"],
            r["non_losses"], r["false_alarms"], r["false_alarm_pct"], r["avg_opportunity_cost_pct"],
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    run()
