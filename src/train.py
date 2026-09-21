"""Train calibrated candidates and persist a cost-aware fraud decision pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (ARTIFACTS, CATEGORICAL_COLUMNS, IMPORTANCE_PATH, METRICS_PATH,
                     MODEL_PATH, NUMERIC_COLUMNS, RAW_DATA, TARGET)
from .cost_engine import CostPolicy, apply_review_capacity, realised_cost
from .data import feature_columns, make_features, validate_transactions
from .drift import build_reference_profile
from .early_warning import add_historical_features
from .feedback import apply_confirmed_feedback


def best_f1_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[f1.argmax()])


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL_COLUMNS),
    ], sparse_threshold=0)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def calibrated_scores(pipeline: Pipeline, calibrator: IsotonicRegression, features: pd.DataFrame):
    return calibrator.transform(pipeline.predict_proba(features)[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the calibrated fraud-detection pipeline.")
    parser.add_argument("--input", type=Path, default=RAW_DATA, help="Normalized training CSV; defaults to data/raw/transactions.csv")
    args = parser.parse_args()
    raw = apply_confirmed_feedback(validate_transactions(pd.read_csv(args.input), require_target=True))
    # Each early-warning feature sees only prior events, preventing future-data leakage.
    data = make_features(add_historical_features(raw)).sort_values("timestamp", kind="stable")
    train_end, calibration_end = int(len(data) * .65), int(len(data) * .80)
    train, calibration, test = data.iloc[:train_end], data.iloc[train_end:calibration_end], data.iloc[calibration_end:]

    # Ensure fraud_within_* lookahead columns are not used as features for either model
    features = feature_columns(data)
    features = [f for f in features if not f.startswith("fraud_within_")]

    # Primary target sample weighting
    positive_weight = (train[TARGET] == 0).sum() / max(1, (train[TARGET] == 1).sum())
    weights = train[TARGET].map({0: 1.0, 1: positive_weight})

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2500, C=.7, solver="saga", random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.08, max_leaf_nodes=20, l2_regularization=.5, random_state=42
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=2, max_features=.8,
            n_jobs=-1, random_state=42,
        ),
    }
    trained, comparison = {}, {}
    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(train[features], train[TARGET], model__sample_weight=weights)
        raw_probabilities = pipeline.predict_proba(calibration[features])[:, 1]
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(raw_probabilities, calibration[TARGET])
        probabilities = calibrator.transform(raw_probabilities)
        comparison[name] = {
            "calibration_pr_auc": float(average_precision_score(calibration[TARGET], probabilities)),
            "calibration_brier": float(brier_score_loss(calibration[TARGET], probabilities)),
        }
        trained[name] = (pipeline, calibrator)

    # PR-AUC is the model-selection metric for the rare fraud class.
    selected_name = max(comparison, key=lambda name: comparison[name]["calibration_pr_auc"])
    pipeline, calibrator = trained[selected_name]
    test_probabilities = calibrated_scores(pipeline, calibrator, test[features])
    calibration_probabilities = calibrated_scores(pipeline, calibrator, calibration[features])
    threshold = best_f1_threshold(calibration[TARGET], calibration_probabilities)
    predictions = (test_probabilities >= threshold).astype(int)
    policy = CostPolicy()
    test_decisions = test[["transaction_id", "amount", TARGET]].copy()
    test_decisions["fraud_probability"] = test_probabilities
    review_capacity = max(1, min(100, round(len(test) * .05)))
    test_decisions = apply_review_capacity(test_decisions, review_capacity, policy)

    # --- Train the intent (fraud_within_3) model -------------------------------------------------
    intent_info = None
    intent_target = "fraud_within_3"
    if intent_target in data.columns:
        # Sample weighting for intent model
        positive_weight_intent = (train[intent_target] == 0).sum() / max(1, (train[intent_target] == 1).sum())
        weights_intent = train[intent_target].map({0: 1.0, 1: positive_weight_intent})
        intent_model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.08, max_leaf_nodes=20, l2_regularization=.5, random_state=2026
        )
        intent_pipeline = build_pipeline(intent_model)
        intent_pipeline.fit(train[features], train[intent_target], model__sample_weight=weights_intent)
        raw_prob_intent = intent_pipeline.predict_proba(calibration[features])[:, 1]
        intent_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        intent_calibrator.fit(raw_prob_intent, calibration[intent_target])
        intent_probs = intent_calibrator.transform(raw_prob_intent)
        intent_threshold = best_f1_threshold(calibration[intent_target], intent_probs)
        test_prob_intent = intent_calibrator.transform(intent_pipeline.predict_proba(test[features])[:, 1])
        intent_info = {
            "pipeline": intent_pipeline,
            "calibrator": intent_calibrator,
            "threshold": intent_threshold,
            "test_probabilities": test_prob_intent,
            "calibration_pr_auc": float(average_precision_score(calibration[intent_target], intent_probs)),
            "brier": float(brier_score_loss(calibration[intent_target], intent_probs)),
            "roc_auc": float(roc_auc_score(calibration[intent_target], intent_probs)) if calibration[intent_target].nunique() > 1 else None,
        }
    # ----------------------------------------------------------------------------------------------

    metrics = {
        "algorithm_selected": selected_name, "candidate_comparison": comparison,
        "rows_train": len(train), "rows_calibration": len(calibration), "rows_test": len(test),
        "fraud_rate_test": float(test[TARGET].mean()), "roc_auc": float(roc_auc_score(test[TARGET], test_probabilities)),
        "pr_auc": float(average_precision_score(test[TARGET], test_probabilities)),
        "brier_score": float(brier_score_loss(test[TARGET], test_probabilities)), "threshold": threshold,
        "classification_report": classification_report(test[TARGET], predictions, output_dict=True, zero_division=0),
        "triage": {"review_capacity": review_capacity, "realised_cost": realised_cost(test_decisions, policy),
                   "action_counts": test_decisions["recommended_action"].value_counts().to_dict()},
        "cost_policy": policy.as_dict(),
        "intent_model": {
            "trained": intent_info is not None,
            "metrics": {k: v for k, v in (intent_info or {}).items() if k in {"calibration_pr_auc", "brier", "roc_auc"}} if intent_info else None,
        },
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "threshold": threshold,
        "features": features,
        "model_name": selected_name,
        "cost_policy": policy.as_dict(),
        "drift_reference": build_reference_profile(train[features]),
    }

    if intent_info is not None:
        bundle.update({
            "intent_pipeline": intent_info["pipeline"],
            "intent_calibrator": intent_info["calibrator"],
            "intent_threshold": intent_info["threshold"],
            "intent_target": intent_target,
        })

    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance = permutation_importance(pipeline, test[features], test[TARGET], n_repeats=5,
                                        scoring="average_precision", random_state=42)
    pd.DataFrame({"feature": features, "importance_mean": importance.importances_mean,
                  "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(IMPORTANCE_PATH, index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "candidate_comparison"}}, indent=2))
    print(f"Selected {selected_name}; model saved to {MODEL_PATH}; trained from {args.input}")


if __name__ == "__main__":
    main()
