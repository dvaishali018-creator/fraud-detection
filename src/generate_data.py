"""Generate realistic synthetic data so the project can be demonstrated immediately."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .config import RAW_DATA
from .early_warning import add_historical_features


REQUIRED_COLUMNS = [
    "transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
    "payment_method", "country", "device_trusted", "account_age_days", "failed_attempts_24h",
    "is_fraud",
]


def inject_fraud_trajectories(df: pd.DataFrame, num_patterns: int = 50) -> pd.DataFrame:
    """Inject sequential fraud campaigns into random customer timelines.

    The function adds three canonical attack motifs: probing attacks, velocity bursts,
    and account-takeover chains. Every row inside a motif is flagged as fraud so the
    downstream multi-horizon objectives can learn the lead-up behavior.
    """
    if df.empty:
        return df.copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns for a fraud trajectory injection: {missing}")

    result = df.copy()
    result["customer_id"] = result["customer_id"].astype(str)
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").fillna(0.0)
    result["device_trusted"] = pd.to_numeric(result["device_trusted"], errors="coerce").fillna(1).astype(int)
    result["account_age_days"] = pd.to_numeric(result["account_age_days"], errors="coerce").fillna(0).astype(int)
    result["failed_attempts_24h"] = pd.to_numeric(result["failed_attempts_24h"], errors="coerce").fillna(0).astype(int)
    result["is_fraud"] = pd.to_numeric(result["is_fraud"], errors="coerce").fillna(0).astype(int)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["timestamp"] = result["timestamp"].fillna(pd.Timestamp("2025-01-01 00:00:00"))

    rng = np.random.default_rng(2025 + max(1, num_patterns))
    customers = result["customer_id"].drop_duplicates().tolist()
    customer_anchor = result.groupby("customer_id")["timestamp"].max().to_dict()
    tx_counter = int(
        pd.to_numeric(
            result["transaction_id"].str.extract(r"(\d+)$", expand=False),
            errors="coerce",
        ).dropna().max() if not result["transaction_id"].empty else 0
    )

    categories = ["grocery", "fuel", "travel", "electronics", "fashion", "gaming"]
    countries = ["IN", "US", "GB", "AE", "NG"]
    methods = ["card", "wallet", "bank_transfer"]

    def _new_row(customer_id: str, ts: pd.Timestamp, amount: float, category: str,
                 method: str, country: str, trusted: int, age_days: int,
                 failed: int) -> dict:
        nonlocal tx_counter
        tx_counter += 1
        return {
            "transaction_id": f"TX-{tx_counter:07d}",
            "customer_id": customer_id,
            "timestamp": ts,
            "amount": float(amount),
            "merchant_category": category,
            "payment_method": method,
            "country": country,
            "device_trusted": int(trusted),
            "account_age_days": int(age_days),
            "failed_attempts_24h": int(failed),
            "is_fraud": 1,
        }

    def _customer_base_age(customer_id: str) -> int:
        customer_age = result.loc[result["customer_id"] == customer_id, "account_age_days"]
        if customer_age.empty:
            return 365
        return int(customer_age.median())

    def _new_country(customer_id: str, current_country: str) -> str:
        recent = result.loc[result["customer_id"] == customer_id, "country"].dropna()
        valid = [c for c in countries if c != current_country and c not in recent.unique()]
        if not valid:
            valid = [c for c in countries if c != current_country]
        return rng.choice(valid)

    def _new_merchant(customer_id: str, preferred=None) -> str:
        recent = result.loc[result["customer_id"] == customer_id, "merchant_category"].dropna()
        valid = [c for c in categories if c not in recent.unique()]
        if preferred is not None and preferred in valid:
            return preferred
        if not valid:
            valid = categories
        return rng.choice(valid)

    injected_rows = []
    pattern_cycle = ["probing", "velocity", "ato"]

    for pattern_idx in range(max(0, num_patterns)):
        pattern = pattern_cycle[pattern_idx % len(pattern_cycle)]
        customer_id = rng.choice(customers)
        anchor_ts = customer_anchor.get(customer_id, result["timestamp"].max())
        start_ts = anchor_ts + pd.Timedelta(hours=rng.integers(6, 72))

        if pattern == "probing":
            n_small = int(rng.integers(2, 4))
            probe_offsets = [pd.Timedelta(hours=int(h)) for h in rng.integers(3, 10, size=n_small)]
            small_times = [start_ts + off for off in probe_offsets]
            last_small = small_times[-1]
            big_time = last_small + pd.Timedelta(minutes=rng.integers(15, 90))
            small_amounts = np.round(rng.uniform(10, 49, size=n_small), 2)
            big_amount = float(rng.uniform(5000, 18000))
            category = _new_merchant(customer_id, preferred="electronics")
            for ts, amount in zip(small_times, small_amounts):
                age_days = max(1, int(_customer_base_age(customer_id) + rng.integers(-30, 30)))
                failed = int(rng.integers(2, 8))
                method = rng.choice(methods)
                injected_rows.append(_new_row(customer_id, ts, amount, category, method,
                                             result.loc[result["customer_id"] == customer_id, "country"].mode().iloc[0],
                                             1, age_days, failed))
            method = rng.choice(methods)
            country = result.loc[result["customer_id"] == customer_id, "country"].mode().iloc[0]
            age_days = max(1, int(_customer_base_age(customer_id) + rng.integers(-20, 20)))
            injected_rows.append(_new_row(customer_id, big_time, big_amount, category, method, country, 1, age_days, int(rng.integers(4, 10))))
            customer_anchor[customer_id] = big_time

        elif pattern == "velocity":
            n_tx = int(rng.integers(5, 9))
            cluster_start = start_ts
            offsets = np.sort(rng.uniform(0, 20, size=n_tx))
            velocities = np.round(rng.uniform(200, 800, size=n_tx), 2)
            category = _new_merchant(customer_id, preferred="travel")
            customer_country = result.loc[result["customer_id"] == customer_id, "country"].mode().iloc[0]
            for idx in range(n_tx):
                ts = cluster_start + pd.Timedelta(minutes=float(offsets[idx]))
                age_days = max(1, int(_customer_base_age(customer_id) + rng.integers(-25, 25)))
                method = rng.choice(methods)
                failed = int(rng.integers(1, 6))
                injected_rows.append(_new_row(customer_id, ts, float(velocities[idx]), category, method,
                                             customer_country, 1, age_days, failed))
            customer_anchor[customer_id] = cluster_start + pd.Timedelta(minutes=float(offsets[-1]))

        else:
            medium_amount = float(rng.uniform(250, 1200))
            scale = float(rng.uniform(1.7, 2.5))
            current_country = result.loc[result["customer_id"] == customer_id, "country"].mode().iloc[0]
            new_country = _new_country(customer_id, current_country)
            category = _new_merchant(customer_id, preferred="electronics")
            times = [start_ts + pd.Timedelta(minutes=offset) for offset in [0, 12, 30, 60, 95]]
            amounts = [medium_amount] + [medium_amount * (scale ** idx) for idx in range(1, 5)]
            for ts, amount in zip(times, amounts):
                age_days = max(1, int(_customer_base_age(customer_id) + rng.integers(-10, 20)))
                method = rng.choice(methods)
                failed = int(rng.integers(3, 12))
                injected_rows.append(_new_row(customer_id, ts, float(amount), category, method,
                                             new_country, 0, age_days, failed))
            customer_anchor[customer_id] = times[-1]

    if injected_rows:
        injected = pd.DataFrame(injected_rows)
        result = pd.concat([result, injected], ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
        result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Restore the exact column ordering and dtypes expected by the downstream pipeline.
    result["transaction_id"] = result["transaction_id"].astype(str)
    result["customer_id"] = result["customer_id"].astype(str)
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").round(2)
    result["merchant_category"] = result["merchant_category"].astype(str)
    result["payment_method"] = result["payment_method"].astype(str)
    result["country"] = result["country"].astype(str)
    result["device_trusted"] = pd.to_numeric(result["device_trusted"], errors="coerce").fillna(1).astype(int)
    result["account_age_days"] = pd.to_numeric(result["account_age_days"], errors="coerce").fillna(0).astype(int)
    result["failed_attempts_24h"] = pd.to_numeric(result["failed_attempts_24h"], errors="coerce").fillna(0).astype(int)
    result["is_fraud"] = pd.to_numeric(result["is_fraud"], errors="coerce").fillna(0).astype(int)
    result = result[REQUIRED_COLUMNS]
    return result


def create_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=rows, freq="min")
    categories = rng.choice(["grocery", "fuel", "travel", "electronics", "fashion", "gaming"], rows,
                           p=[.28, .14, .09, .16, .19, .14])
    countries = rng.choice(["IN", "US", "GB", "AE", "NG"], rows, p=[.58, .16, .10, .09, .07])
    methods = rng.choice(["card", "wallet", "bank_transfer"], rows, p=[.67, .24, .09])
    amounts = np.round(rng.lognormal(mean=4.1, sigma=.9, size=rows), 2)
    trusted = rng.choice([0, 1], rows, p=[.18, .82])
    age = rng.integers(1, 2500, rows)
    failed = rng.poisson(.25, rows)
    high_risk = ((amounts > 500) | np.isin(categories, ["electronics", "gaming"])) & (trusted == 0)
    frame = pd.DataFrame({
        "transaction_id": [f"TX-{i:07d}" for i in range(rows)],
        "customer_id": [f"C-{x:05d}" for x in rng.integers(1, max(80, rows // 40), rows)],
        "timestamp": times.astype(str), "amount": amounts, "merchant_category": categories,
        "payment_method": methods, "country": countries, "device_trusted": trusted,
        "account_age_days": age, "failed_attempts_24h": failed,
    })
    early = add_historical_features(frame)
    # Simulated account-takeover/burst signal makes the early-warning experiment demonstrable.
    fraud_probability = (0.012 + .19 * high_risk + .10 * (failed >= 3) + .06 * (age < 14)
                         + .05 * (countries == "NG") + .14 * (early.prior_txn_count_1h >= 2)
                         + .06 * (early.minutes_since_prior_txn <= 5))
    frame["is_fraud"] = (rng.random(rows) < np.clip(fraud_probability, 0, .92)).astype(int)
    frame = inject_fraud_trajectories(frame, num_patterns=50)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12000)
    args = parser.parse_args()
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    data = create_dataset(args.rows)
    data.to_csv(RAW_DATA, index=False)
    print(f"Saved {len(data):,} rows to {RAW_DATA}; fraud rate: {data.is_fraud.mean():.2%}")
