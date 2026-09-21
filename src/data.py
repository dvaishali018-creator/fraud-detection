"""Data contract validation and feature engineering."""
from __future__ import annotations

import pandas as pd
import numpy as np

from .config import OPTIONAL_INPUT_COLUMNS, REQUIRED_COLUMNS, TARGET


def validate_transactions(frame: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Validate a raw transaction frame and return a safe copy."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if require_target and TARGET not in frame.columns:
        missing.add(TARGET)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")

    checked = frame.copy()
    for column in OPTIONAL_INPUT_COLUMNS:
        if column not in checked:
            checked[column] = np.nan
    # External feeds can legitimately mix ISO-8601 timestamps with and without
    # fractional seconds; parse each valid ISO representation independently.
    checked["timestamp"] = pd.to_datetime(checked["timestamp"], errors="coerce", format="mixed")
    if checked["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")
    checked["amount"] = pd.to_numeric(checked["amount"], errors="coerce")
    if checked["amount"].isna().any() or (checked["amount"] <= 0).any():
        raise ValueError("amount must be numeric and positive")
    if require_target and not set(checked[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 or 1")
    return checked


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create non-leaking features available at transaction decision time."""
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"])
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    result["is_night"] = result["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["log_amount"] = np.log1p(result["amount"])
    account_age = pd.to_numeric(result["account_age_days"], errors="coerce")
    failed_attempts = pd.to_numeric(result["failed_attempts_24h"], errors="coerce")
    result["account_age_missing"] = account_age.isna().astype(int)
    result["failed_attempts_missing"] = failed_attempts.isna().astype(int)
    result["account_age_log"] = np.log1p(account_age.clip(lower=0))
    # Missing means unavailable, not zero.  The accompanying missingness flag
    # preserves that distinction for the model.
    result["failed_attempt_flag"] = (failed_attempts > 0).astype(int)
    # Fixed policy threshold; use a business-approved threshold in a real deployment.
    result["high_amount_flag"] = (result["amount"] >= 500).astype(int)
    result["device_trusted"] = result["device_trusted"].astype(str)
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {"transaction_id", "customer_id", "timestamp", TARGET}]
