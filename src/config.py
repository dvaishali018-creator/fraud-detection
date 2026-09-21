from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "transactions.csv"
SCORED_DATA = ROOT / "data" / "scored" / "transactions_scored.csv"
FEEDBACK_DATA = ROOT / "data" / "feedback" / "analyst_feedback.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "fraud_pipeline.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
IMPORTANCE_PATH = ARTIFACTS / "feature_importance.csv"
DRIFT_REPORT_PATH = ARTIFACTS / "latest_drift_report.json"
TARGET = "is_fraud"

NUMERIC_COLUMNS = [
    "amount", "account_age_days", "failed_attempts_24h", "hour", "is_weekend",
    "prior_txn_count_1h", "prior_amount_24h", "minutes_since_prior_txn", "new_merchant_for_customer",
    "log_amount", "is_night", "day_of_week", "hour_sin", "hour_cos", "account_age_log",
    "failed_attempt_flag", "high_amount_flag", "prior_txn_count_24h", "prior_avg_amount_24h",
    "amount_to_prior_avg", "new_country_for_customer", "distinct_categories_24h",
    "account_age_missing", "failed_attempts_missing",
]
CATEGORICAL_COLUMNS = ["merchant_category", "payment_method", "country", "device_trusted"]
REQUIRED_COLUMNS = [
    "transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
    "payment_method", "country", "device_trusted",
]
# These signals are useful when a bank supplies them, but public datasets normally
# do not.  The pipeline keeps missing values explicit and lets the imputer handle
# them; adapters must never fabricate failed authentication attempts.
OPTIONAL_INPUT_COLUMNS = ["account_age_days", "failed_attempts_24h"]
