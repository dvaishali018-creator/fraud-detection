"""Batch-score transactions with calibrated risk, triage, early warning and drift checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from .config import DRIFT_REPORT_PATH, MODEL_PATH, RAW_DATA, SCORED_DATA
from .cost_engine import CostPolicy, apply_review_capacity
from .data import make_features, validate_transactions
from .drift import assess_drift
from .early_warning import add_historical_features


def calibrated_probabilities(bundle: dict, featured: pd.DataFrame):
    raw = bundle["pipeline"].predict_proba(featured[bundle["features"]])[:, 1]
    return bundle["calibrator"].transform(raw)


def calibrated_probabilities_intent(bundle: dict, featured: pd.DataFrame):
    # Return NaN array if no intent model present
    if not bundle or "intent_pipeline" not in bundle or "intent_calibrator" not in bundle:
        return pd.Series([float("nan")] * len(featured))
    raw = bundle["intent_pipeline"].predict_proba(featured[bundle["features"]])[:, 1]
    return pd.Series(bundle["intent_calibrator"].transform(raw))


def risk_reasons(row: pd.Series) -> str:
    reasons = []
    amount = float(row.get("amount", 0.0) or 0.0)
    if amount >= 500:
        reasons.append("high_amount")
    if int(row.get("device_trusted", 1) or 1) == 0:
        reasons.append("untrusted_device")
    if int(row.get("failed_attempts_24h", 0) or 0) >= 3:
        reasons.append("repeated_failed_attempts")
    if int(row.get("account_age_days", 0) or 0) < 14:
        reasons.append("new_account")
    if int(row.get("prior_txn_count_1h", 0) or 0) >= 3:
        reasons.append("rapid_transaction_burst")
    if float(row.get("minutes_since_prior_txn", 9999) or 9999) <= 5:
        reasons.append("short_time_since_prior_transaction")
    if int(row.get("new_merchant_for_customer", 0) or 0) == 1:
        reasons.append("new_merchant_pattern")
    if int(row.get("new_country_for_customer", 0) or 0) == 1:
        reasons.append("new_country_pattern")
    if float(row.get("amount_to_prior_avg", 0.0) or 0.0) >= 3:
        reasons.append("amount_far_above_recent_average")
    if int(row.get("is_night", 0) or 0) == 1:
        reasons.append("unusual_night_time")
    return ", ".join(reasons) or "model_pattern"


def compute_counterfactual(featured_row: pd.Series, bundle: dict, feature_cols: list[str]) -> str:
    """Compute simple single-change counterfactuals and return a short textual primary driver.

    Tests three independent interventions (one at a time):
    - device_trusted -> trusted
    - new_country_for_customer -> 0 (known country)
    - amount -> 20% of original (recompute derived amount features)

    Returns a human-readable string describing the single best change and the percent risk drop.
    If inputs are missing or baseline probability is zero, returns "N/A".
    """
    if featured_row is None or bundle is None or not feature_cols:
        return "N/A"

    base = featured_row.copy()
    # Build baseline feature vector
    try:
        base_df = pd.DataFrame([base[feature_cols]])
        raw_base = bundle["pipeline"].predict_proba(base_df)[:, 1]
        base_prob = float(bundle["calibrator"].transform(raw_base)[0])
    except Exception:
        return "N/A"

    if base_prob <= 0:
        return "N/A"

    scenarios = []

    # Scenario A: device trusted
    dev = base.copy()
    if "device_trusted" in dev.index:
        # in features device_trusted may be string; set to '1' or 1 accordingly
        dev_val = dev["device_trusted"]
        dev["device_trusted"] = "1" if isinstance(dev_val, str) else 1
        try:
            df = pd.DataFrame([dev[feature_cols]])
            prob = float(bundle["calibrator"].transform(bundle["pipeline"].predict_proba(df)[:, 1])[0])
            scenarios.append(("device_trusted", prob))
        except Exception:
            pass

    # Scenario B: known country (new_country_for_customer -> 0)
    nc = base.copy()
    if "new_country_for_customer" in nc.index:
        nc["new_country_for_customer"] = 0
        try:
            df = pd.DataFrame([nc[feature_cols]])
            prob = float(bundle["calibrator"].transform(bundle["pipeline"].predict_proba(df)[:, 1])[0])
            scenarios.append(("new_country_for_customer", prob))
        except Exception:
            pass

    # Scenario C: amount reduced to 20% (and update dependent features)
    amt = base.copy()
    if "amount" in amt.index:
        try:
            new_amount = float(amt["amount"]) * 0.2
        except Exception:
            new_amount = None
        if new_amount is not None:
            amt["amount"] = new_amount
            # update common derived features if present
            if "log_amount" in amt.index:
                amt["log_amount"] = np.log1p(max(0.0, new_amount))
            if "high_amount_flag" in amt.index:
                amt["high_amount_flag"] = int(new_amount >= 500)
            if "prior_avg_amount_24h" in amt.index and "amount_to_prior_avg" in amt.index:
                prior_avg = float(amt.get("prior_avg_amount_24h", 0.0))
                prior_avg = prior_avg if prior_avg > 0 else 1e-6
                amt["amount_to_prior_avg"] = new_amount / prior_avg
            try:
                df = pd.DataFrame([amt[feature_cols]])
                prob = float(bundle["calibrator"].transform(bundle["pipeline"].predict_proba(df)[:, 1])[0])
                scenarios.append(("amount_20pct", prob))
            except Exception:
                pass

    if not scenarios:
        return "N/A"

    # Compute percent drops
    drops = []
    for name, prob in scenarios:
        drop = max(0.0, (base_prob - prob) / base_prob * 100.0)
        drops.append((name, prob, drop))

    # Pick best drop
    best = max(drops, key=lambda x: x[2])
    name_map = {
        "device_trusted": "device were trusted",
        "new_country_for_customer": "transaction were from a known country",
        "amount_20pct": "amount were 20% of current value",
    }
    if best[2] <= 0:
        return "N/A"
    return f"Risk drops by {best[2]:.0f}% if {name_map.get(best[0], best[0])}"


def score(input_path=RAW_DATA, output_path=SCORED_DATA, review_capacity: int = 100) -> pd.DataFrame:
    input_path, output_path = Path(input_path), Path(output_path)
    bundle = joblib.load(MODEL_PATH)
    data = validate_transactions(pd.read_csv(input_path))
    enriched = add_historical_features(data)
    featured = make_features(enriched)
    probability = calibrated_probabilities(bundle, featured)
    escalation = calibrated_probabilities_intent(bundle, featured)
    result = enriched.copy()
    result["fraud_probability"] = probability.round(4)
    result["fraud_prediction"] = (probability >= bundle["threshold"]).astype(int)
    # intent outputs
    result["escalation_probability"] = escalation.round(4)
    if "intent_threshold" in bundle:
        result["escalation_prediction"] = (escalation >= bundle["intent_threshold"]).astype(int)
    else:
        result["escalation_prediction"] = 0

    policy = CostPolicy(**bundle["cost_policy"])
    result = apply_review_capacity(result, review_capacity, policy)
    result["risk_reasons"] = result.apply(risk_reasons, axis=1)

    # Compute counterfactual primary risk driver only for REVIEW/BLOCK to save compute
    result = result.copy()
    result["primary_risk_driver"] = "N/A"
    actionable_mask = result["recommended_action"].isin(["REVIEW", "BLOCK"]) if "recommended_action" in result.columns else pd.Series(False, index=result.index)
    feature_cols = bundle.get("features", [])
    if actionable_mask.any() and feature_cols:
        for idx in result[actionable_mask].index:
            try:
                featured_row = featured.loc[idx]
            except Exception:
                featured_row = None
            driver = compute_counterfactual(featured_row, bundle, feature_cols)
            result.loc[idx, "primary_risk_driver"] = driver

    drift_report = assess_drift(featured[bundle["features"]], bundle["drift_reference"])
    result["drift_status"] = drift_report["status"]
    result.attrs["drift_report"] = drift_report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    DRIFT_REPORT_PATH.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_DATA))
    parser.add_argument("--output", default=str(SCORED_DATA))
    parser.add_argument("--review-capacity", type=int, default=100)
    args = parser.parse_args()
    scored = score(args.input, args.output, args.review_capacity)
    print(f"Scored {len(scored):,}; review {sum(scored.recommended_action.eq('REVIEW')):,}; "
          f"block {sum(scored.recommended_action.eq('BLOCK')):,}; drift: {scored.attrs['drift_report']['status']}.")
