"""Small, transparent Population Stability Index (PSI) monitor."""
from __future__ import annotations

import numpy as np
import pandas as pd


DRIFT_COLUMNS = ["amount", "account_age_days", "failed_attempts_24h", "prior_txn_count_1h", "prior_amount_24h"]


def build_reference_profile(frame: pd.DataFrame) -> dict:
    profile = {}
    for column in DRIFT_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
        # Some public datasets do not expose bank-only signals such as failed
        # authentication attempts. They cannot have a meaningful PSI baseline.
        if len(values) == 0:
            continue
        quantiles = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.unique(quantiles)
        if len(bins) < 2:
            bins = np.array([values.min() - .5, values.max() + .5])
        bins[0], bins[-1] = -np.inf, np.inf
        expected = np.histogram(values, bins=bins)[0] / max(1, len(values))
        profile[column] = {"bins": bins.tolist(), "expected": expected.tolist()}
    return profile


def population_stability_index(values: pd.Series, reference: dict) -> float:
    bins = np.asarray(reference["bins"], dtype=float)
    expected = np.asarray(reference["expected"], dtype=float)
    actual_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    actual = np.histogram(actual_values, bins=bins)[0] / max(1, len(actual_values))
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def assess_drift(frame: pd.DataFrame, profile: dict) -> dict:
    scores = {column: population_stability_index(frame[column], ref) for column, ref in profile.items()}
    maximum = max(scores.values(), default=0.0)
    status = "RETRAIN" if maximum >= .25 else "MONITOR" if maximum >= .10 else "STABLE"
    return {"status": status, "max_psi": round(maximum, 4), "feature_psi": {k: round(v, 4) for k, v in scores.items()}}
