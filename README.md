# Transaction Fraud Detection — AI/ML Final-Year Project

An end-to-end, reproducible fraud-detection project for financial transactions. It covers data creation/ingestion, data-quality checks, feature engineering, class-imbalance handling, model training, evaluation, explainability, batch scoring, a FastAPI prediction service, and a Streamlit dashboard.

## Workflow

```text
CSV / synthetic data
       ↓
schema + quality validation
       ↓
feature engineering (time, velocity, risk flags)
       ↓
chronological train/calibration/test split → efficient model selection
       ↓
probability calibration → cost-aware APPROVE / REVIEW / BLOCK triage
       ↓
saved model + metrics + drift baseline + feature importance
       ↓
batch scoring / REST API / dashboard / analyst-feedback retraining
```

## Quick start

Requires Python 3.10+.

```powershell
cd C:\Users\CHETAN\Documents\Codex\2026-08-19\he\outputs\publishing_fraud_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data --rows 12000
python -m src.train
python -m src.score --input data\raw\transactions.csv --output data\scored\transactions_scored.csv
uvicorn src.api:app --reload
```

In a second terminal, launch the dashboard:

```powershell
streamlit run dashboard.py
```

## Input data contract

The ingestion CSV needs these columns:

| Field | Meaning |
|---|---|
| `transaction_id` | unique transaction identifier |
| `customer_id` | customer/card identifier (hashed in real deployments) |
| `timestamp` | ISO timestamp |
| `amount` | positive amount in the transaction currency |
| `merchant_category` | merchant category code/name |
| `payment_method` | card, wallet, bank transfer, etc. |
| `country` | transaction country |
| `device_trusted` | 0/1 device trust signal |
| `account_age_days` | age of customer account |
| `failed_attempts_24h` | recent failed payment attempts |
| `is_fraud` | target label: 0 legitimate, 1 fraud (training only) |

For production inference, omit `is_fraud`.

`account_age_days` and `failed_attempts_24h` are optional external-security signals. When a public dataset does not contain them, the pipeline records them as unavailable rather than fabricating values. The rich external-dataset adapter can be run with:

```powershell
python -m src.adapters --input path\to\source.csv --output data\raw\external_transactions.csv
```

It supports the documented transaction/customer/timestamp/amount/category/channel/country/device aliases, derives `device_trusted` from prior device history, and derives a clearly labelled observed-history-age proxy only when true account age is unavailable.

## Research and novelty features

- **Early fraud warnings:** leakage-safe velocity and behavioural-change features identify rapid transaction bursts, rapid repeat payments, high recent spend, and new merchant behaviour.
- **Expanded features:** 22 numeric features include amount transformations, time cyclicality, account maturity, failed-attempt flags, cross-country changes, 1-hour/24-hour velocity, historical average-spend deviation, and category diversity.
- **Efficient model selection:** Logistic Regression is an interpretable baseline; Histogram Gradient Boosting and Extra Trees capture nonlinear fraud patterns. Chronological validation selects by PR-AUC.
- **Calibrated probabilities:** isotonic calibration makes risk probabilities suitable for action-cost calculations.
- **Cost-aware three-way triage:** the system chooses APPROVE, REVIEW or BLOCK based on expected loss, fraud-review effectiveness, false-block friction cost, and analyst review capacity.
- **Adaptive operations:** PSI flags input-data drift; analyst feedback is captured separately and merged only on the next training cycle.
- **Risk-concentration alerts:** the dashboard ranks country, merchant-category, and payment-method cohorts by expected fraud value (transaction amount × calibrated probability), helping analysts spot operational hotspots.
- **Linked-transaction investigation:** analysts can open a high-risk case and view same-customer events plus matching country/category activity within 24 hours. These links are contextual evidence only and never modify a score or action.

Read the [software workflow and implementation plan](docs/SOFTWARE_WORKFLOW.md) for architecture, algorithm rationale, build steps, experiments, and deployment safeguards.

For the optional custom local LoRA analyst-explanation model, data requirements, and operating-system/hardware guidance, read [llm/README.md](llm/README.md). The LLM is advisory only and never replaces the fraud score or action policy.

## Key design choices for the report

- **Leakage-aware split:** chronological 80/20 split approximates deployment: train on history, test on later transactions.
- **Imbalance-aware learning:** fraud examples have greater training weight without fabricating observations.
- **Metric choice:** PR-AUC, recall and precision are primary; Brier score checks probability reliability; raw accuracy is misleading for a highly imbalanced class.
- **Decision policy:** a static F1 threshold is reported for comparison, while production actions are selected from expected approve/review/block cost and a capped review queue.
- **Explainability:** permutation importance is saved after training; individual API predictions include human-readable reason flags.

## API example

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body '{
  "transaction_id":"TX-NEW-01", "customer_id":"C-9001", "timestamp":"2026-08-19T13:30:00",
  "amount":5200, "merchant_category":"electronics", "payment_method":"card",
  "country":"NG", "device_trusted":0, "account_age_days":4, "failed_attempts_24h":4
}'
```

## Project structure

```text
src/                Pipeline modules and API
data/raw/           Source data (generated sample is ignored by Git)
data/scored/        Batch prediction output
artifacts/          Trained model, metrics and importance report
tests/              Unit tests for data validation and feature engineering
dashboard.py        Streamlit monitoring/user-interface page
```

## Responsible deployment

This is an academic prototype. Do not automatically decline a customer solely from its score. Monitor performance/drift by cohort, protect personal data, secure the API, document human review, and retrain only after label quality is verified.
