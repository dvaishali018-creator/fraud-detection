import pandas as pd
import pytest

from src.data import make_features, validate_transactions
from src.adapters import adapt_rich_transactions


def sample_frame():
    return pd.DataFrame([{
        "transaction_id": "TX-1", "customer_id": "C-1", "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0, "merchant_category": "grocery", "payment_method": "card", "country": "IN",
        "device_trusted": 1, "account_age_days": 100, "failed_attempts_24h": 0, "is_fraud": 0,
    }])


def test_features_add_time_attributes():
    featured = make_features(validate_transactions(sample_frame(), require_target=True))
    assert featured.loc[0, "hour"] == 10
    assert featured.loc[0, "is_weekend"] == 0


def test_rejects_non_positive_amount():
    frame = sample_frame()
    frame.loc[0, "amount"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_transactions(frame)


def test_adapter_maps_rich_dataset_and_derives_prior_device_trust():
    source = pd.DataFrame([
        {"transaction_id": "A", "customer_id": "C1", "timestamp": "2026-01-01T10:00:00", "amount": 12,
         "merchant_category": "grocery", "channel": "web", "country": "IN", "device_fingerprint": "D1", "is_fraud": 0},
        {"transaction_id": "B", "customer_id": "C1", "timestamp": "2026-01-03T10:00:00", "amount": 25,
         "merchant_category": "grocery", "channel": "web", "country": "IN", "device_fingerprint": "D1", "is_fraud": 1},
    ])
    adapted = adapt_rich_transactions(source)
    assert adapted["device_trusted"].tolist() == [0, 1]
    assert adapted["account_age_days"].tolist() == [0, 2]
    assert adapted["failed_attempts_24h"].isna().all()
