# Software Workflow and Implementation Plan

## System workflow

```text
Transaction event / CSV
        ↓
Data contract validation
        ↓
Early-warning feature state
  • previous 1-hour transaction count
  • previous 24-hour spend
  • minutes since previous transaction
  • first merchant-category indicator
        ↓
Calibrated fraud probability
        ↓
Cost-aware decision engine
        ├── APPROVE
        ├── REVIEW (priority queue limited by analyst capacity)
        └── BLOCK
        ↓
Dashboard / API / batch output
        ↓
Analyst feedback + drift monitor
        ↓
Label validation → governed retraining
```

## Why these algorithms

| Component | Choice | Reason |
|---|---|---|
| Transparent baseline | Class-weighted Logistic Regression | Fast, interpretable, and essential for a credible baseline comparison. |
| Main classifier | Histogram Gradient Boosting | Efficient on medium-to-large tabular data, captures nonlinear fraud behaviour, and trains far faster than deep learning for this feature set. |
| Model selection | Validation PR-AUC | Better reflects rare-fraud detection than accuracy. |
| Probability reliability | Isotonic calibration | The cost engine requires meaningful probabilities, not only rankings. |
| Early warning | Leakage-safe velocity features | Detects sudden behaviour changes before many failed/chargeback labels arrive. |
| Drift monitoring | Population Stability Index | Lightweight, explainable monthly/weekly alert that works without labels. |
| Decision layer | Expected-cost triage | Turns a probability into operational actions under a real review budget. |

The training pipeline selects the candidate with the strongest chronological validation PR-AUC, then evaluates only once on the final 20% future holdout.

## Build steps

1. **Set up source control**
   - Create a private GitHub repository.
   - Commit only code and small documentation; never commit real cardholder data, trained artifacts, or secrets.

2. **Create an environment and run the baseline**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m src.generate_data --rows 12000
   python -m src.train
   ```

3. **Review the evidence**
   - Read `artifacts/metrics.json` for PR-AUC, Brier score, selected algorithm and business-cost result.
   - Inspect `artifacts/feature_importance.csv`.
   - Keep screenshots/exports for the dissertation.

4. **Demonstrate batch and dashboard operation**
   ```powershell
   python -m src.score --input data\raw\transactions.csv --review-capacity 100
   streamlit run dashboard.py
   ```
   Upload a new CSV with the required inference columns. Record analyst outcomes in the dashboard; feedback is stored separately and applied only when the next training cycle is run.

5. **Demonstrate online scoring**
   ```powershell
   uvicorn src.api:app --reload
   ```
   Open `http://127.0.0.1:8000/docs`, submit multiple transactions for the same customer, and show how the early-warning values change.

6. **Conduct the publication experiment**
   - Use a real approved/anonymised dataset or clearly mark simulated results as simulated.
   - Keep chronological order; never randomly shuffle transactions before the split.
   - Simulate 7-, 14-, and 30-day label delays.
   - Compare: Logistic Regression → Gradient Boosting → calibrated model → cost-aware triage → triage with drift retraining.
   - Report PR-AUC, Brier score, recall@review-capacity, false-block rate, realised cost, and post-drift recovery.

7. **Deploy only as a decision-support prototype**
   - Use HTTPS, authentication, encrypted storage, secrets management, audit logging and a real shared feature store.
   - Do not make an irreversible automatic customer decision from this academic prototype.

## Publication contribution statement

> We propose a leakage-safe, calibrated fraud-triage workflow that combines behavioural early-warning signals, expected-cost decisioning, constrained analyst review, feedback-based retraining, and unlabeled data-drift monitoring. The evaluation studies detection quality, calibration, operational review capacity, financial cost, and resilience under delayed labels.

This is stronger than claiming novelty solely from using a particular classifier.
