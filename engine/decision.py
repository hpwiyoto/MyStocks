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
    BUY   — probability >= 0.5 (model believes success is more likely than
            not, in absolute terms).
    WATCH — base_rate <= probability < 0.5 (a real edge over the
            unconditional historical average, but not yet coin-flip-favorable).
    AVOID — probability < base_rate (no edge at all; worse than doing
            nothing / picking at random from history).
"""

BUY_THRESHOLD = 0.5


def decide(probability: float, base_rate: float, entry_price: float, target_pct: float, stop_pct: float) -> dict:
    if probability >= BUY_THRESHOLD:
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
