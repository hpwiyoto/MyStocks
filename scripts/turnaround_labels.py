"""Turnaround target labeling: for a (ticker, date) row currently in a
"bad" regime (bearish/bottoming), did it genuinely turn around within the
next 6 months -- reach early_reversal/bullish AND hold there for at least
20 consecutive trading days without slipping back into a bad or
over-extended regime?

Design history (two prior calibrations, both checked against the real
dataset before landing here):
- v1 (loosest): hold broken only by falling back to bearish/bottoming.
  Measured 90.3% success rate -- confirmed too lenient (bearish/bottoming
  are only 2 of 7 regimes, so almost every stock drifts through a non-bad
  regime somewhere in a 125-day window by chance).
- v2 (strictest): hold required STRICTLY early_reversal/bullish the whole
  20 days, no sideways/accumulation/overextended at all. Measured 6.2%
  success rate, but walk-forward validation came back with ROC-AUC ~0.49
  (no better than random) -- too few genuinely-independent turnaround
  events for the model to learn from, and/or too narrow a definition to
  correspond to a learnable market pattern.
- v3 (this one, the agreed middle ground): hold is broken by bearish,
  bottoming, OR overextended (overextended = due for a pullback, i.e.
  already stretched again -- not a stable "held" turnaround either) but
  sideways/accumulation during the hold window are tolerated as normal
  wobble in a real uptrend.

Design (agreed with the user before any code was written):
- Starting condition: regime in {bearish, bottoming} on the labeled date.
- Horizon: 6 months =~ 125 trading days (21 trading days/month x 6).
- Success: within the horizon, regime transitions into {early_reversal,
  bullish} at some day T, and for the 20 trading days after T (T+1..T+20),
  regime never falls into HOLD_DISQUALIFYING_REGIMES.
- Unresolved (not enough forward data to confirm or refute within the
  horizon) -> NaN, same convention as the swing model's triple-barrier
  labeling -- excluded from training, not forced to 0.
- A row that ISN'T bearish/bottoming to begin with isn't a turnaround
  candidate at all -- filtered out by the caller before labels ever mean
  anything for it.
"""
import numpy as np
import pandas as pd

from features.regime import BAD_REGIMES, GOOD_REGIMES

HOLD_DISQUALIFYING_REGIMES = BAD_REGIMES | {"overextended"}
HORIZON_TRADING_DAYS = 125  # ~6 months
HOLD_TRADING_DAYS = 20


def label_turnaround_series(regimes: np.ndarray, horizon: int = HORIZON_TRADING_DAYS,
                             hold: int = HOLD_TRADING_DAYS) -> np.ndarray:
    """regimes: 1D array of regime strings for ONE ticker, in ascending date
    order (index i = trading day i). Returns a same-length float array:
    1.0 = turnaround confirmed, 0.0 = horizon exhausted without one, NaN =
    starting regime isn't bad OR not enough forward data to resolve."""
    n = len(regimes)
    labels = np.full(n, np.nan)
    is_bad = np.array([r in BAD_REGIMES for r in regimes])
    is_good = np.array([r in GOOD_REGIMES for r in regimes])

    for i in range(n):
        if not is_bad[i]:
            continue  # not a turnaround candidate row at all
        window_end = i + horizon
        if window_end >= n:
            continue  # not enough forward data -- stays NaN (unresolved), not forced to 0
        resolved = False
        had_unverifiable_attempt = False
        for t in range(i + 1, window_end + 1):
            if not is_good[t]:
                continue
            hold_end = t + hold
            if hold_end >= n:
                # a good-regime transition genuinely happened here, but its
                # 20-day hold window runs past available data -- can't
                # confirm OR refute THIS attempt. Keep scanning in case an
                # EARLIER transition already resolved (it would have, and
                # `break`ed, before we ever got here), but if nothing
                # resolves, this row must stay NaN (ambiguous), not be
                # forced to 0 -- there WAS an attempt here, we just can't
                # see far enough to judge it.
                had_unverifiable_attempt = True
                continue
            held = all(regimes[k] not in HOLD_DISQUALIFYING_REGIMES for k in range(t + 1, hold_end + 1))
            if held:
                labels[i] = 1.0
                resolved = True
                break
        if not resolved and labels[i] != 1.0 and not had_unverifiable_attempt:
            labels[i] = 0.0
    return labels


def build_turnaround_labels(features: pd.DataFrame) -> pd.DataFrame:
    """features: full panel with columns stock_code, date, regime (raw
    string, NOT one-hot yet). Returns stock_code, date, turnaround_label."""
    all_labels = []
    for code, g in features.groupby("stock_code"):
        g = g.sort_values("date")
        regimes = g["regime"].fillna("").to_numpy()
        labels = label_turnaround_series(regimes)
        all_labels.append(pd.DataFrame({
            "stock_code": code, "date": g["date"].to_numpy(), "turnaround_label": labels,
        }))
    return pd.concat(all_labels, ignore_index=True)
