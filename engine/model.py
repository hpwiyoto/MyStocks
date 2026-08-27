"""Load a trained model + metadata, and turn one feature_daily row into the
exact feature vector the model expects.

Reproduces training-time preprocessing (features/db.py's FEATURE_VERSION
schema + the notebook's prepare_panel) for a SINGLE row of live data:
historical_win_rate imputation, has_similar_pattern flag, and regime
one-hot encoding. pd.get_dummies on a single row would only ever produce a
column for whatever regime that one row has — the other regime_* columns
the model expects would be silently missing — so those are reconstructed
explicitly from the metadata's feature list instead.
"""
import json
import os

import pandas as pd
import xgboost as xgb

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_model_and_metadata(model_version: str):
    model_path = os.path.join(MODEL_DIR, f"{model_version}.json")
    meta_path = os.path.join(MODEL_DIR, f"{model_version}_metadata.json")

    booster = xgb.Booster()
    booster.load_model(model_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    return booster, metadata


def build_feature_row(feature_row: dict, feature_cols: list[str]):
    """Returns (X, missing_cols). X is a 1-row DataFrame with exactly
    `feature_cols` in order. A column that's genuinely NULL for this row
    (e.g. trailing_pe for a persistently loss-making company -- ~34% of
    rows, confirmed by checking the DB directly) is passed through as NaN
    rather than skipping the ticker entirely: XGBoost handles missing
    values natively (DMatrix treats NaN as "missing" and learns a default
    branch direction for it during training), so there's no reason a single
    absent ratio should blank out an otherwise-complete prediction.
    missing_cols is kept in the return signature for callers that still
    want to log/inspect it, but is no longer used to force a skip here."""
    row = dict(feature_row)

    similar_count = row.get("similar_pattern_count") or 0
    row["has_similar_pattern"] = 1 if similar_count > 0 else 0
    if row.get("historical_win_rate") is None:
        row["historical_win_rate"] = 0.5

    regime_value = row.get("regime")
    for c in feature_cols:
        if c.startswith("regime_"):
            row[c] = 1 if c == f"regime_{regime_value}" else 0

    values = {c: row.get(c) for c in feature_cols}
    missing = [c for c, v in values.items() if v is None]

    X = pd.DataFrame([values], columns=feature_cols).astype(float)  # None -> NaN
    return X, missing


def predict_probability(booster: xgb.Booster, X: pd.DataFrame) -> float:
    return float(booster.predict(xgb.DMatrix(X))[0])
