"""Convert a model probability into an actionable decision.

Risk management rules (documented, not arbitrary):

- Entry/SL/TP are derived DIRECTLY from the same target_pct/stop_pct the
  model was trained to predict (from the model's metadata). Using different
  percentages here than what defines the label would make the probability
  meaningless relative to the actual trade proposed.
- risk_reward_ratio = target_pct / stop_pct — fixed by that same definition
  (currently 5% / 2.5% = 2.0), not a separately chosen number.
- Decision tiers are anchored to the model's own historical base rate
  (fraction of all training rows that were a "win"), not an arbitrary
  round number:
    BUY   — probability >= BUY_THRESHOLD (see below).
    WATCH — base_rate <= probability < BUY_THRESHOLD (a real edge over the
            unconditional historical average, but not yet BUY-grade).
    AVOID — probability < base_rate (no edge at all; worse than doing
            nothing / picking at random from history).

BUY_THRESHOLD was 0.5 ("more likely than not, in absolute terms") through
model v3. Raised to 0.60 for v4 after a walk-forward threshold sweep
(threshold_sweep.py, pooled out-of-sample predictions across all 4 folds)
showed a consistent, monotonic improvement in precision/profit-factor/max-
drawdown as the threshold rises from 0.5 to 0.7 -- 0.5->0.60 alone moves
precision 71.4%->76.1% and profit factor 4.99->6.35. 0.60 was picked over
more aggressive values (e.g. 0.70, precision 79.5%/profit factor 7.75) as
the more robust, less overfit-to-the-sweep-itself choice that still keeps a
usable number of live BUY signals rather than making them vanishingly rare.

Raised again to 0.65 for v5 (scripts/tune_v5.py's pooled out-of-sample
sweep, same 4-fold walk-forward methodology). v5's sweep behaves
differently from v4's: it does NOT improve monotonically all the way to
0.70 -- 0.65 is the actual peak (precision 84.5%, profit factor 11.98, max
drawdown -6.7%), with 0.70 slightly worse on profit factor and drawdown
despite fewer trades. Picking 0.70 anyway (as "more conservative must be
better") would have meant choosing a worse point on v5's curve, not a more
robust one -- 0.65 is both the empirical peak AND avoids the extreme-value
overfit concern that ruled out 0.70 for v4.

Brought back down to 0.60 shortly after, at the user's explicit request:
0.65 meant ~2.9 BUY signals/day on average across ~900 tickers (measured
via scripts/random_baseline_check.py over the most recent 250 trading
days), which is an AVERAGE -- the daily count varies enough that several
consecutive days can land at zero, which is exactly what the user hit and
flagged. 0.60 raises that to ~4.2 signals/day at a real but modest
precision cost (78.1% -> 76.5% in that same 250-day check; walk-forward
win_rate 78.1% pooled, not the 84.5% headline number, which was the
stricter 4-fold walk-forward figure) -- a deliberate trade of a few points
of precision for a usable BUY tier that isn't empty on a routine basis.
"""

BUY_THRESHOLD = 0.60

# IDX's practical price floor ("gocap") -- confirmed by checking real Home
# output: PNBS/MDLN/HDIT/CPRO were all parked exactly at Rp50, BTEK at Rp10,
# each showing an inflated BUY probability. At this price, tick size (Rp1)
# alone is 2-3% of price -- most of the model's 5%-target/2.5%-stop window,
# so a "win" there is largely tick-level noise, not a real technical signal.
# Forced to AVOID here regardless of model probability, on top of (not
# instead of) excluding these rows from training entirely -- see
# scripts/export_for_colab.py's matching GOCAP_PRICE_FLOOR use, so a v5+
# model never saw these rows during training either and serving stays
# consistent with what it was actually trained on.
GOCAP_PRICE_FLOOR = 50


def decide(probability: float, base_rate: float, entry_price: float, target_pct: float, stop_pct: float) -> dict:
    if entry_price <= GOCAP_PRICE_FLOOR:
        decision = "AVOID"
    elif probability >= BUY_THRESHOLD:
        decision = "BUY"
    elif probability >= base_rate:
        decision = "WATCH"
    else:
        decision = "AVOID"

    return {
        "decision": decision,
        "entry_price": entry_price,
        "stop_loss_price": entry_price * (1 - stop_pct),
        "take_profit_price": entry_price * (1 + target_pct),
        "risk_reward_ratio": target_pct / stop_pct,
    }
