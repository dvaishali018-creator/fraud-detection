"""Adapters that convert external fraud datasets to the project's safe schema.

The rich synthetic transaction dataset described in the project documentation
contains customer histories, device identifiers, countries, merchant categories,
payment channels, and labels.  Its exact column names vary between releases, so
this adapter resolves documented aliases and fails clearly when a required signal
is absent.  It never invents authentication-failure data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REQUIRED_COLUMNS, TARGET


ALIASES = {
    "transaction_id": ("transaction_id", "TransactionID", "id"),
    "customer_id": ("customer_id", "user_id", "customer", "account_id"),
    "timestamp": ("timestamp", "transaction_timestamp", "date", "datetime"),
    "amount": ("amount", "transaction_amount", "TransactionAmt"),
    "merchant_category": ("merchant_category", "category", "merchant_type", "merchant"),
    "payment_method": ("payment_method", "channel", "transaction_type", "type"),
    "country": ("country", "location", "country_code"),
    "device": ("device_fingerprint", "device_id", "device", "device_used", "DeviceInfo"),
    "account_age_days": ("account_age_days",),
    "failed_attempts_24h": ("failed_attempts_24h", "failed_attempts"),
    TARGET: (TARGET, "isFraud", "fraud_label", "label"),
}


def _source_column(columns: pd.Index, field: str, required: bool = True) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in ALIASES[field]:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"Dataset is missing a source column for '{field}'. Expected one of {ALIASES[field]}.")
    return None


def adapt_rich_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Map one ordered source frame to the project schema.

    `device_trusted` is a leakage-safe proxy: a device is trusted only after it
    has appeared in a *prior* transaction for the same customer.  If account age
    is absent, `account_age_days` is the observed-history age, clearly a proxy
    rather than a claim about the real account opening date.
    """
    sources = {field: _source_column(frame.columns, field, field not in {"account_age_days", "failed_attempts_24h", TARGET})
               for field in ALIASES}
    result = pd.DataFrame({field: frame[column] for field, column in sources.items()
                           if column is not None and field not in {"device", "account_age_days", "failed_attempts_24h"}})
    # Large generated files can mix ISO strings with and without fractional
    # seconds. `mixed` parses each ISO representation instead of applying the
    # first row's exact format to every later row.
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", format="mixed", utc=True)
    if result["timestamp"].isna().any():
        raise ValueError("The source timestamp contains invalid values.")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    if result["amount"].isna().any() or (result["amount"] <= 0).any():
        raise ValueError("The source amount must be numeric and positive.")

    # Sorting before stateful derivation ensures every trust/age feature only sees
    # events that happened earlier. Stable sort makes equal timestamps deterministic.
    result["_device"] = frame[sources["device"]].astype("string") if sources["device"] else pd.Series(pd.NA, index=frame.index, dtype="string")
    result["_source_account_age"] = (pd.to_numeric(frame[sources["account_age_days"]], errors="coerce")
                                     if sources["account_age_days"] else np.nan)
    result["_source_failed_attempts"] = (pd.to_numeric(frame[sources["failed_attempts_24h"]], errors="coerce")
                                           if sources["failed_attempts_24h"] else np.nan)
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    seen_devices: dict[str, set[str]] = {}
    first_seen: dict[str, pd.Timestamp] = {}
    trusted, observed_age = [], []
    for customer, timestamp, device in result[["customer_id", "timestamp", "_device"]].itertuples(index=False, name=None):
        customer = str(customer)
        known = seen_devices.setdefault(customer, set())
        device_known = pd.notna(device) and str(device) in known
        trusted.append(int(device_known))
        first_timestamp = first_seen.setdefault(customer, timestamp)
        observed_age.append(max(0, int((timestamp - first_timestamp).total_seconds() // 86400)))
        if pd.notna(device):
            known.add(str(device))
    result["device_trusted"] = trusted
    result["account_age_days"] = result["_source_account_age"] if sources["account_age_days"] else observed_age
    result["failed_attempts_24h"] = result["_source_failed_attempts"] if sources["failed_attempts_24h"] else np.nan
    result = result.drop(columns=["_device", "_source_account_age", "_source_failed_attempts"])
    if TARGET in result:
        result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    return result[[*REQUIRED_COLUMNS, "account_age_days", "failed_attempts_24h", *([TARGET] if TARGET in result else [])]]


def adapt_rich_transactions_sample(source: Path, sample_size: int) -> pd.DataFrame:
    """Read evenly distributed windows from an ordered rich CSV into a safe sample.

    The source must be ordered by timestamp and use conventional one-line CSV
    records. Sampling windows are spread across the file rather than taken only
    from its beginning. Trust and observed-history age are derived only from
    previously retained events, so the sample stays leakage-safe without a
    multi-gigabyte sequential scan.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    records_per_window = min(1_000, sample_size)
    window_count = (sample_size + records_per_window - 1) // records_per_window
    records: list[dict] = []
    seen_devices: dict[str, set[str]] = {}
    first_seen: dict[str, pd.Timestamp] = {}
    last_timestamp: pd.Timestamp | None = None

    with source.open("rb") as stream:
        header_end = stream.tell()
        header = next(csv.reader([stream.readline().decode("utf-8")]))
        header_end = stream.tell()
        required_sources = ("transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "channel", "country", "device_fingerprint", TARGET)
        missing = set(required_sources) - set(header)
        if missing:
            raise ValueError(f"Dataset is missing columns required for streaming adaptation: {sorted(missing)}")
        file_size = source.stat().st_size
        for window in range(window_count):
            remaining = sample_size - len(records)
            rows_to_read = min(records_per_window, remaining)
            start = header_end + (file_size - header_end) * window // window_count
            stream.seek(start)
            if window:
                stream.readline()  # Drop the partial row introduced by seeking into a CSV record.
            for _ in range(rows_to_read):
                raw_row = stream.readline()
                if not raw_row:
                    break
                values = next(csv.reader([raw_row.decode("utf-8")]))
                if len(values) != len(header):
                    raise ValueError("Encountered a multi-line or malformed CSV record while sampling.")
                row = dict(zip(header, values))
                timestamp = pd.Timestamp(row["timestamp"])
                if last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError("Streaming sample requires the source CSV to be sorted by timestamp.")
                last_timestamp = timestamp
                amount = float(row["amount"])
                if amount <= 0:
                    raise ValueError("The source amount must be positive.")
                customer = row["customer_id"]
                device = row["device_fingerprint"]
                known_devices = seen_devices.setdefault(customer, set())
                trusted = int(device in known_devices)
                first_timestamp = first_seen.setdefault(customer, timestamp)
                observed_age = max(0, int((timestamp - first_timestamp).total_seconds() // 86400))
                known_devices.add(device)
                label = str(row[TARGET]).strip().lower()
                if label not in {"0", "1", "false", "true"}:
                    raise ValueError(f"Unexpected is_fraud label: {row[TARGET]!r}")
                records.append({
                    "transaction_id": row["transaction_id"], "customer_id": customer,
                    "timestamp": timestamp.isoformat(), "amount": amount,
                    "merchant_category": row["merchant_category"], "payment_method": row["channel"],
                    "country": row["country"], "device_trusted": trusted,
                    "account_age_days": observed_age, "failed_attempts_24h": np.nan,
                    TARGET: int(label in {"1", "true"}),
                })
    return pd.DataFrame.from_records(records, columns=[*REQUIRED_COLUMNS, "account_age_days", "failed_attempts_24h", TARGET])


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt a rich external fraud CSV to the project schema.")
    parser.add_argument("--input", required=True, help="Source CSV path")
    parser.add_argument("--output", required=True, help="Adapted CSV path")
    parser.add_argument("--sample-size", type=int, help="Read an evenly distributed, timestamp-ordered sample of this many rows")
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    adapted = (adapt_rich_transactions_sample(source, args.sample_size)
               if args.sample_size else adapt_rich_transactions(pd.read_csv(source)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(destination, index=False)
    print(f"Adapted {len(adapted):,} rows to {destination}")


if __name__ == "__main__":
    main()
