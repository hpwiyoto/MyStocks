"""Early-warning check for a TRACKED position (app.db.tracked_positions)
-- direct user request: once a stock is marked "bought", keep watching it
day by day and flag if it looks like it's drifting toward the stop-loss,
so the loss can be cut before the full -X% is actually hit, rather than
only ever finding out at the stop itself.

Empirically backtested first (scripts/test_early_warning_rule.py) against
the shipped Swing config's own historical walk-forward BUY signals, same
"test before shipping" posture as everything else in this project --
NOT assumed to work just because the underlying signals (RSI slope, CMF,
EMA9 position) are individually reasonable-sounding.

Rule (ANY_2 variant -- the one that actually cleared the backtest;
ALL_3 never once fired in 762 historical signals, too strict for a 5-day
window): the position is flagged while UNDERWATER (current close below
entry price) AND at least 2 of these 3 hold on the same day:
  - close < ema_9            (short-term trend broken)
  - rsi_slope_3d < 0         (momentum weakening)
  - cmf_20 < 0               (money flowing out)

Real but limited: catches 11/214 historical losses early (5.1%,
avg loss saved when caught +2.99pp) at a 0.2% false-alarm rate on
non-losses. The ceiling is inherently low under a 5-day horizon --
68.7% of historical losses hit the stop on DAY 1 itself, before any
warning could possibly have had a prior day to fire on. See scripts/
test_early_warning_rule.py's docstring for the full numbers and caveats
-- this is a real, modest edge, not a promise to catch every loss early.

Usage:
    from engine.early_warning import check_position
    warning = check_position(entry_price, feat_row, current_close)
"""
MIN_WEAKNESS_SIGNALS = 2


def check_position(entry_price: float, feat_row: dict, current_close: float) -> dict:
    """`feat_row`: the ticker's latest feature_daily row (dict, e.g. from
    app.data.load_latest_feature_row) -- needs ema_9/rsi_slope_3d/cmf_20.
    Returns {"underwater": bool, "n_signals": int, "warning": bool,
    "signals": {name: bool}} -- the breakdown, not just the final flag, so
    a caller can show WHY a warning fired, not just that it did."""
    underwater = current_close < entry_price

    ema_9 = feat_row.get("ema_9")
    rsi_slope_3d = feat_row.get("rsi_slope_3d")
    cmf_20 = feat_row.get("cmf_20")

    signals = {
        "trend_broken": ema_9 is not None and ema_9 == ema_9 and current_close < ema_9,
        "momentum_weakening": rsi_slope_3d is not None and rsi_slope_3d == rsi_slope_3d and rsi_slope_3d < 0,
        "money_flowing_out": cmf_20 is not None and cmf_20 == cmf_20 and cmf_20 < 0,
    }
    n_signals = sum(signals.values())
    warning = underwater and n_signals >= MIN_WEAKNESS_SIGNALS

    return {"underwater": underwater, "n_signals": n_signals, "warning": warning, "signals": signals}
