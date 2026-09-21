"""FastAPI online inference service."""
from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MODEL_PATH
from .cost_engine import CostPolicy, choose_action
from .data import make_features
from .early_warning import EarlyWarningState
from .score import calibrated_probabilities, calibrated_probabilities_intent, risk_reasons

app = FastAPI(title="Transaction Fraud Detection API", version="1.0.0")
_bundle = None
_online_state = EarlyWarningState()


class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    timestamp: str
    amount: float = Field(gt=0)
    merchant_category: str
    payment_method: str
    country: str
    device_trusted: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    failed_attempts_24h: int = Field(ge=0)


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="No model found. Run `python -m src.train` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(transaction: Transaction):
    bundle = get_bundle()
    raw = pd.DataFrame([transaction.model_dump()])
    enriched = _online_state.enrich_frame(raw)
    features = make_features(enriched)

    # Primary fraud probability
    probability = float(calibrated_probabilities(bundle, features)[0])

    # Escalation / intent probability (fraud_within_3)
    escalation_series = calibrated_probabilities_intent(bundle, features)
    escalation_probability = float(escalation_series.iloc[0]) if len(escalation_series) > 0 and not pd.isna(escalation_series.iloc[0]) else None

    action, expected_cost = choose_action(probability, transaction.amount, CostPolicy(**bundle["cost_policy"]))

    response = {
        "transaction_id": transaction.transaction_id,
        "fraud_probability": round(probability, 4),
        "fraud_prediction": int(probability >= bundle["threshold"]),
        "decision": action,
        "threshold": round(float(bundle["threshold"]), 4),
        "risk_reasons": risk_reasons(enriched.iloc[0]).split(", "),
        "early_warning_features": {key: float(enriched.iloc[0][key]) for key in [
            "prior_txn_count_1h", "prior_txn_count_24h", "prior_amount_24h", "prior_avg_amount_24h",
            "amount_to_prior_avg", "minutes_since_prior_txn", "new_merchant_for_customer",
            "new_country_for_customer", "distinct_categories_24h",
        ]},
        "expected_action_cost": {key: round(value, 2) for key, value in expected_cost.items()},
    }

    # Attach escalation probability when available
    if escalation_probability is not None:
        response["escalation_probability"] = round(escalation_probability, 4)
        if "intent_threshold" in bundle and bundle.get("intent_threshold") is not None:
            response["escalation_prediction"] = int(escalation_probability >= bundle["intent_threshold"])

    return response
