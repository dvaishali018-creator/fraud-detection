"""Leakage-safe behavioural features for detecting a fraud burst early."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

EARLY_WARNING_COLUMNS = [
    "prior_txn_count_1h", "prior_amount_24h", "minutes_since_prior_txn", "new_merchant_for_customer",
    "prior_txn_count_24h", "prior_avg_amount_24h", "amount_to_prior_avg", "new_country_for_customer",
    "distinct_categories_24h",
]


@dataclass
class EarlyWarningState:
    """In-memory online feature state. Replace with Redis/feature store in production."""
    history: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    seen_categories: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    seen_countries: dict[str, set] = field(default_factory=lambda: defaultdict(set))

    def enrich_record(self, record: dict) -> dict:
        result = dict(record)
        customer = str(record["customer_id"])
        timestamp = pd.Timestamp(record["timestamp"])
        amount = float(record["amount"])
        category = str(record["merchant_category"])
        country = str(record["country"])
        events = self.history[customer]
        while events and timestamp - events[0][0] > pd.Timedelta(hours=24):
            events.popleft()
        prior_1h = sum(timestamp - event_time <= pd.Timedelta(hours=1) for event_time, _, _, _ in events)
        prior_amount = sum(event_amount for _, event_amount, _, _ in events)
        prior_count = len(events)
        prior_average = prior_amount / prior_count if prior_count else amount
        minutes_since = 24 * 60 if not events else max(0.0, (timestamp - events[-1][0]).total_seconds() / 60)
        result.update({
            "prior_txn_count_1h": prior_1h,
            "prior_amount_24h": prior_amount,
            "minutes_since_prior_txn": minutes_since,
            "new_merchant_for_customer": int(category not in self.seen_categories[customer]),
            "prior_txn_count_24h": prior_count,
            "prior_avg_amount_24h": prior_average,
            "amount_to_prior_avg": amount / max(prior_average, 1.0),
            "new_country_for_customer": int(country not in self.seen_countries[customer]),
            "distinct_categories_24h": len({event_category for _, _, event_category, _ in events}),
        })
        # Commit only after the transaction was represented by its past behaviour.
        events.append((timestamp, amount, category, country))
        self.seen_categories[customer].add(category)
        self.seen_countries[customer].add(country)
        return result

    def enrich_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for index, row in output.sort_values("timestamp", kind="stable").iterrows():
            enriched = self.enrich_record(row.to_dict())
            for column in EARLY_WARNING_COLUMNS:
                output.loc[index, column] = enriched[column]
        return output


def add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive historical features using only transactions before each event timestamp."""
    return EarlyWarningState().enrich_frame(frame)
