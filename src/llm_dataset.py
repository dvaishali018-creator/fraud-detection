"""Create redacted instruction-tuning examples from reviewed fraud cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import RAW_DATA, TARGET
from .data import make_features, validate_transactions
from .early_warning import add_historical_features

SYSTEM_PROMPT = (
    "You are a fraud-analyst assistant. Explain only the structured evidence supplied. "
    "Never invent evidence, never request personal data, and never make the final block decision. "
    "Return JSON with assessment, evidence, and recommended_review_step."
)


def case_message(row: pd.Series) -> str:
    return json.dumps({
        "amount": round(float(row.amount), 2), "merchant_category": row.merchant_category,
        "payment_method": row.payment_method, "country": row.country, "hour": int(row.hour),
        "device_trusted": int(row.device_trusted), "account_age_days": int(row.account_age_days),
        "failed_attempts_24h": int(row.failed_attempts_24h), "prior_txn_count_1h": int(row.prior_txn_count_1h),
        "prior_txn_count_24h": int(row.prior_txn_count_24h),
        "amount_to_prior_avg": round(float(row.amount_to_prior_avg), 2),
        "new_merchant_for_customer": int(row.new_merchant_for_customer),
        "new_country_for_customer": int(row.new_country_for_customer),
    })


def target_message(row: pd.Series) -> str:
    evidence = []
    if row.failed_attempts_24h >= 3:
        evidence.append("repeated failed attempts")
    if row.prior_txn_count_1h >= 3:
        evidence.append("rapid transaction burst")
    if row.amount_to_prior_avg >= 3:
        evidence.append("amount is unusually high for recent activity")
    if row.new_country_for_customer:
        evidence.append("new transaction country")
    if not evidence:
        evidence.append("no single high-risk signal; compare with account history")
    is_fraud = int(row[TARGET]) == 1
    return json.dumps({
        "assessment": "suspicious" if is_fraud else "low_evidence_of_fraud",
        "evidence": evidence,
        "recommended_review_step": "verify with analyst and supporting account evidence" if is_fraud else "allow standard monitoring",
    })


def build_dataset(input_path: Path, output_path: Path, limit: int | None = None) -> int:
    raw = validate_transactions(pd.read_csv(input_path), require_target=True)
    data = make_features(add_historical_features(raw)).sort_values("timestamp", kind="stable")
    if limit:
        data = data.head(limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for _, row in data.iterrows():
            example = {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a concise analyst case brief from this redacted event: {case_message(row)}"},
                {"role": "assistant", "content": target_message(row)},
            ]}
            file.write(json.dumps(example) + "\n")
    return len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--output", type=Path, default=Path("llm/data/fraud_analyst_sft.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(f"Wrote {build_dataset(args.input, args.output, args.limit):,} redacted SFT examples to {args.output}")
