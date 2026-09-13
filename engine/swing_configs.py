"""Registry of the top-5 Swing target/stop/horizon configs (by top5_lift)
from scripts/search_swing_target.py's 20-config search, each trained as
its own sibling model -- direct user request for a toggle between them in
the app, rather than shipping only the single winner.

scripts/train_v5.py trains config #1 (the winner) in place as
models/direction_xgboost_v5.json -- that's also the app-wide DEFAULT.
scripts/train_v5_variants.py trains the other 4 as separate model files.
All 5 share the same feature set/hyperparameters (see either script's
docstring for why); each has its OWN independently re-tuned BUY_THRESHOLD
(a new target/horizon needs its own threshold sweep, not the winner's
reused -- see scripts/tune_v5_new_target_threshold.py's docstring).

Every entry's precision/Wilson-LB/signals-per-day numbers below are read
live from that config's own models/{model_version}_metadata.json (via
app.data.load_model_metadata) wherever they're displayed -- this registry
only carries the static identity (target/stop/horizon/model_version/rank),
not a frozen snapshot of numbers that would drift out of sync with the
model files.
"""

SWING_CONFIGS = [
    {
        "id": "t10_h5", "model_version": "direction_xgboost_v5", "rank": 1, "top5_lift": 0.228498,
        "label": "10% / -5% / 5 hari (default, paling optimal)",
    },
    {
        "id": "t7_h5", "model_version": "direction_xgboost_v5_t7_h5", "rank": 2, "top5_lift": 0.202931,
        "label": "7% / -3.5% / 5 hari",
    },
    {
        "id": "t10_h10", "model_version": "direction_xgboost_v5_t10_h10", "rank": 3, "top5_lift": 0.189419,
        "label": "10% / -5% / 10 hari",
    },
    {
        "id": "t7_h10", "model_version": "direction_xgboost_v5_t7_h10", "rank": 4, "top5_lift": 0.183642,
        "label": "7% / -3.5% / 10 hari",
    },
    {
        "id": "t5_h5", "model_version": "direction_xgboost_v5_t5_h5", "rank": 5, "top5_lift": 0.176246,
        "label": "5% / -2.5% / 5 hari",
    },
]

DEFAULT_CONFIG_ID = "t10_h5"
DEFAULT_MODEL_VERSION = "direction_xgboost_v5"

CONFIG_BY_ID = {c["id"]: c for c in SWING_CONFIGS}
CONFIG_BY_MODEL_VERSION = {c["model_version"]: c for c in SWING_CONFIGS}


def config_label(config_id: str) -> str:
    c = CONFIG_BY_ID.get(config_id)
    return c["label"] if c else config_id


def model_version_for(config_id: str) -> str:
    c = CONFIG_BY_ID.get(config_id)
    return c["model_version"] if c else DEFAULT_MODEL_VERSION
