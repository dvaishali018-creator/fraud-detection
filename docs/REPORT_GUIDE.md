# Final-Year Report and Viva Guide

## Suggested title

**An Explainable Machine Learning Pipeline for Real-Time Financial Transaction Fraud Detection**

## Problem statement

Digital-payment fraud is rare but costly. Rule-only systems miss new fraud patterns and can create too many false alerts. This project predicts the probability that an incoming transaction is fraudulent and ranks it for manual review.

## Objectives

1. Build a repeatable pipeline to validate and prepare transaction data.
2. Engineer decision-time features without using future information.
3. Train an imbalance-aware fraud classifier.
4. Evaluate with fraud-appropriate metrics and a chronological test set.
5. Deploy the same preprocessing and model artifact through batch, API, and dashboard interfaces.

## Methodology

| Phase | Implementation | Evidence to include |
|---|---|---|
| Data | Synthetic demonstration data or an approved anonymized source | schema, class distribution, data dictionary |
| Validation | missing columns, duplicate IDs, valid timestamp/amount/label checks | validation rules and failed-row examples |
| Features | hour, weekend, account age, failed attempts, device trust, amount and merchant/country signals | feature table and leakage rationale |
| Modelling | HistGradientBoosting, weighted fraud observations | hyperparameters and train/test sizes |
| Evaluation | chronological 80/20 split, PR-AUC, ROC-AUC, precision, recall, F1 | confusion matrix and PR curve |
| Explainability | permutation importance + record-level reason flags | importance chart and case study |
| Deployment | saved joblib pipeline, FastAPI, Streamlit, batch CSV output | architecture and API screenshot |

## Results section template

After running `python -m src.train`, copy values from `artifacts/metrics.json` into this sentence:

> On the temporally held-out test set, the proposed model achieved a PR-AUC of **[x]** and ROC-AUC of **[x]**. At the selected operating threshold of **[x]**, it produced fraud-class precision **[x]** and recall **[x]**. This threshold is a business setting, not a fixed property of the model.

Do not claim results from the included synthetic data generalize to a real bank. If using a public dataset, state its source, scope, preprocessing decisions, and limitations.

## Recommended diagrams for the dissertation

```text
Transaction CSV → Validator → Feature builder → Saved ML pipeline → fraud probability
                                                         ├→ batch review file
                                                         ├→ FastAPI endpoint
                                                         └→ monitoring dashboard
```

## Viva questions to prepare

1. Why is accuracy not your main metric? (Fraud is rare; a model can be highly accurate by predicting every case as legitimate.)
2. Why use chronological splitting? (It mimics learning from past data and testing on future data, reducing leakage.)
3. What is a false positive? (A legitimate transaction flagged for review; it creates customer friction and review cost.)
4. How do you handle imbalance? (Weighted fraud observations; compare with resampling as an experiment.)
5. How will you monitor production performance? (Track alert rate, precision/recall after labels arrive, population drift and subgroup behaviour.)
6. What must change for a real deployment? (Secure data, encryption, access control, compliance review, alert case management, feedback labels and drift monitoring.)
