"""Analyst feedback capture and label merge for the next governed retraining cycle."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import FEEDBACK_DATA, TARGET

VALID_LABELS = {"fraud": 1, "legitimate": 0}


def record_feedback(transaction_id: str, analyst_label: str, reviewer: str, comment: str = "") -> None:
    if analyst_label not in VALID_LABELS:
        raise ValueError(f"analyst_label must be one of {sorted(VALID_LABELS)}")
    row = pd.DataFrame([{
        "transaction_id": transaction_id, "analyst_label": analyst_label, "reviewer": reviewer,
        "comment": comment, "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }])
    FEEDBACK_DATA.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(FEEDBACK_DATA, mode="a", header=not FEEDBACK_DATA.exists(), index=False)


def apply_confirmed_feedback(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge latest review labels into training data; retain the raw source unchanged."""
    if not FEEDBACK_DATA.exists():
        return frame
    feedback = pd.read_csv(FEEDBACK_DATA)
    valid = feedback[feedback["analyst_label"].isin(VALID_LABELS)].drop_duplicates("transaction_id", keep="last")
    labels = valid.set_index("transaction_id")["analyst_label"].map(VALID_LABELS)
    result = frame.copy()
    result[TARGET] = result["transaction_id"].map(labels).fillna(result[TARGET]).astype(int)
    return result
