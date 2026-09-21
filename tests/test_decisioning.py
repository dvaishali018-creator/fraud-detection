import pandas as pd

from src.cost_engine import apply_review_capacity, choose_action
from src.early_warning import add_historical_features


def test_early_warning_uses_only_prior_customer_events():
    frame = pd.DataFrame([
        {"transaction_id": "T1", "customer_id": "C1", "timestamp": "2026-01-01T10:00:00", "amount": 50, "merchant_category": "grocery"},
        {"transaction_id": "T2", "customer_id": "C1", "timestamp": "2026-01-01T10:05:00", "amount": 75, "merchant_category": "electronics"},
    ])
    featured = add_historical_features(frame)
    assert featured.loc[0, "prior_txn_count_1h"] == 0
    assert featured.loc[1, "prior_txn_count_1h"] == 1
    assert featured.loc[1, "minutes_since_prior_txn"] == 5
    assert featured.loc[1, "new_merchant_for_customer"] == 1


def test_cost_policy_selects_low_and_high_risk_actions():
    assert choose_action(.01, 100)[0] == "APPROVE"
    assert choose_action(.95, 100)[0] == "BLOCK"


def test_review_capacity_limits_queue_size():
    frame = pd.DataFrame({"amount": [100, 100, 100], "fraud_probability": [.2, .25, .3]})
    result = apply_review_capacity(frame, max_reviews=1)
    assert sum(result.recommended_action.eq("REVIEW")) <= 1
