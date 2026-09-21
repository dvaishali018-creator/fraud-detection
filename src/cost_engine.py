"""Business-cost-aware approve, review, and block decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostPolicy:
    fraud_loss_rate: float = 1.0
    review_cost: float = 3.0
    review_detection_rate: float = 0.92
    false_block_cost: float = 18.0

    def as_dict(self) -> dict:
        return asdict(self)


def expected_costs(probability: float, amount: float, policy: CostPolicy = CostPolicy()) -> dict[str, float]:
    fraud_loss = probability * amount * policy.fraud_loss_rate
    return {
        "APPROVE": fraud_loss,
        "REVIEW": policy.review_cost + fraud_loss * (1 - policy.review_detection_rate),
        "BLOCK": (1 - probability) * policy.false_block_cost,
    }


def choose_action(probability: float, amount: float, policy: CostPolicy = CostPolicy()) -> tuple[str, dict[str, float]]:
    costs = expected_costs(probability, amount, policy)
    return min(costs, key=costs.get), costs


def apply_review_capacity(scored: pd.DataFrame, max_reviews: int, policy: CostPolicy = CostPolicy()) -> pd.DataFrame:
    """Keep only the highest-value review cases when analyst capacity is constrained."""
    result = scored.copy()
    recommendations, review_values = [], []
    for row in result.itertuples():
        action, costs = choose_action(float(row.fraud_probability), float(row.amount), policy)
        recommendations.append(action)
        review_values.append(min(costs["APPROVE"], costs["BLOCK"]) - costs["REVIEW"])
    result["recommended_action"] = recommendations
    result["review_value"] = review_values
    review_candidates = result.index[result.recommended_action.eq("REVIEW")]
    allowed = set(result.loc[review_candidates].nlargest(max_reviews, "review_value").index)
    for index in review_candidates:
        if index not in allowed:
            probability, amount = float(result.at[index, "fraud_probability"]), float(result.at[index, "amount"])
            costs = expected_costs(probability, amount, policy)
            result.at[index, "recommended_action"] = min(("APPROVE", "BLOCK"), key=costs.get)
    return result


def realised_cost(scored: pd.DataFrame, policy: CostPolicy = CostPolicy()) -> float:
    """Evaluate a strategy after labels arrive; useful for research comparison."""
    if "is_fraud" not in scored:
        raise ValueError("is_fraud is required to calculate realised cost")
    action = scored["recommended_action"]
    fraud_loss = scored["amount"] * policy.fraud_loss_rate
    return float(
        ((action == "APPROVE") * scored["is_fraud"] * fraud_loss).sum()
        + ((action == "REVIEW") * (policy.review_cost + scored["is_fraud"] * fraud_loss * (1 - policy.review_detection_rate))).sum()
        + ((action == "BLOCK") * (1 - scored["is_fraud"]) * policy.false_block_cost).sum()
    )
