import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DRIFT_REPORT_PATH, IMPORTANCE_PATH, METRICS_PATH, SCORED_DATA
from src.feedback import record_feedback
from src.score import score
from src.local_llm_infer import generate_case_brief


def risk_band(probability: float) -> str:
    """Convert a calibrated probability into a compact analyst-facing band."""
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.20:
        return "Elevated"
    return "Routine"


def concentration_summary(scored: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Surface unusually risky cohorts without using protected characteristics."""
    columns = [group_by, "transactions", "avg_risk", "risk_exposure", "review_or_block"]
    required = {group_by, "fraud_probability", "amount", "recommended_action"}
    if not required.issubset(scored.columns):
        return pd.DataFrame(columns=columns)

    summary = (
        scored.groupby(group_by, dropna=False)
        .agg(
            transactions=("transaction_id", "size"),
            avg_risk=("fraud_probability", "mean"),
            risk_exposure=("amount", lambda amount: float(amount.sum())),
            review_or_block=("recommended_action", lambda action: int(action.isin(["REVIEW", "BLOCK"]).sum())),
        )
        .reset_index()
    )
    # Expected fraud value focuses attention on material, not merely frequent, cohorts.
    expected_value = (
        scored.assign(expected_fraud_value=scored["amount"] * scored["fraud_probability"])
        .groupby(group_by, dropna=False)["expected_fraud_value"]
        .sum()
        .reset_index()
    )
    summary = summary.merge(expected_value, on=group_by, how="left")
    summary["avg_risk"] = summary["avg_risk"].round(3)
    summary["risk_exposure"] = summary["risk_exposure"].round(2)
    summary["expected_fraud_value"] = summary["expected_fraud_value"].round(2)
    return summary.sort_values(["expected_fraud_value", "avg_risk"], ascending=False)


def linked_transactions(scored: pd.DataFrame, transaction_id: str) -> pd.DataFrame:
    """Find nearby events that help an analyst investigate a possible fraud burst.

    Links are evidence for review only: shared customer, same device/context, or a
    short time window. They do not alter the model score or recommended action.
    """
    if transaction_id not in set(scored.get("transaction_id", [])):
        return pd.DataFrame()
    selected = scored.loc[scored["transaction_id"] == transaction_id].iloc[0]
    working = scored.copy()
    working["timestamp_parsed"] = pd.to_datetime(working["timestamp"], errors="coerce")
    selected_time = pd.to_datetime(selected.get("timestamp"), errors="coerce")
    time_nearby = (working["timestamp_parsed"] - selected_time).abs().le(pd.Timedelta(hours=24))
    same_customer = working["customer_id"].eq(selected.get("customer_id"))
    same_context = (
        working["country"].eq(selected.get("country"))
        & working["merchant_category"].eq(selected.get("merchant_category"))
        & time_nearby
    )
    links = working.loc[(same_customer | same_context) & working["transaction_id"].ne(transaction_id)].copy()
    if links.empty:
        return links
    links["link_reason"] = "same customer"
    links.loc[~same_customer & same_context, "link_reason"] = "same country/category within 24h"
    links["hours_from_case"] = ((links["timestamp_parsed"] - selected_time).abs().dt.total_seconds() / 3600).round(1)
    return links.sort_values(["fraud_probability", "hours_from_case"], ascending=[False, True])

st.set_page_config(page_title="Fraud Detection Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ Transaction Fraud Detection Monitor")
st.caption("Academic prototype — use scores for review prioritisation, not automatic adverse action.")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cols = st.columns(4)
    cols[0].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    cols[1].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    cols[2].metric("Decision threshold", f"{metrics['threshold']:.3f}")
    cols[3].metric("Test fraud rate", f"{metrics['fraud_rate_test']:.1%}")
    st.caption(f"Selected algorithm: `{metrics.get('algorithm_selected', 'unknown')}` · "
               "22 numeric risk/behaviour features plus categorical transaction context.")
else:
    st.info("Train a model first: `python -m src.generate_data` then `python -m src.train`.")

uploaded = st.file_uploader("Upload a transaction CSV to score", type="csv")
review_capacity = st.slider("Maximum transactions an analyst team can review", 1, 500, 100)
if uploaded is not None:
    temp_input = Path("data/raw/dashboard_upload.csv")
    temp_input.parent.mkdir(parents=True, exist_ok=True)
    temp_input.write_bytes(uploaded.getvalue())
    try:
        output = score(temp_input, SCORED_DATA, review_capacity)
        st.success(f"Scored {len(output):,} transactions. {sum(output.recommended_action.eq('REVIEW')):,} sent for review.")
    except Exception as exc:
        st.error(str(exc))

if SCORED_DATA.exists():
    scored = pd.read_csv(SCORED_DATA)
    st.subheader("Latest batch")
    c1, c2 = st.columns(2)
    c1.metric("Review queue", int(sum(scored.recommended_action.eq("REVIEW"))))
    c2.metric("Recommended blocks", int(sum(scored.recommended_action.eq("BLOCK"))))
    # Expose primary_risk_driver in the table for analysts
    display_cols = [c for c in scored.columns if c in [
        "transaction_id", "amount", "merchant_category", "country", "fraud_probability",
        "escalation_probability", "recommended_action", "primary_risk_driver",
    ]]
    # Fallback to showing all if the selected subset is not available
    if len(display_cols) >= 1:
        st.dataframe(scored.sort_values("fraud_probability", ascending=False)[display_cols].head(100), width="stretch")
    else:
        st.dataframe(scored.sort_values("fraud_probability", ascending=False).head(100), width="stretch")
    st.download_button("Download scored CSV", SCORED_DATA.read_bytes(), "transactions_scored.csv", "text/csv")

    st.subheader("Novelty operations lab")
    st.caption(
        "These investigation aids prioritise human review. They never change the model score or take an automatic customer action."
    )
    cohort_tab, link_tab = st.tabs(["Risk concentration alerts", "Linked-transaction investigator"])
    with cohort_tab:
        cohort_dimension = st.selectbox(
            "Find concentrated risk by", ["country", "merchant_category", "payment_method"], key="cohort_dimension"
        )
        cohorts = concentration_summary(scored, cohort_dimension)
        if cohorts.empty:
            st.info("The latest batch does not include the fields needed for cohort analysis.")
        else:
            total_expected = cohorts["expected_fraud_value"].sum()
            leading_share = 0.0 if total_expected == 0 else cohorts.iloc[0]["expected_fraud_value"] / total_expected
            a, b, c = st.columns(3)
            a.metric("Cohorts monitored", len(cohorts))
            b.metric("Largest expected-loss concentration", f"{leading_share:.0%}")
            c.metric("Top cohort", str(cohorts.iloc[0][cohort_dimension]))
            st.dataframe(
                cohorts.head(20),
                width="stretch",
                column_config={
                    "avg_risk": st.column_config.ProgressColumn("Average risk", min_value=0.0, max_value=1.0, format="%.3f"),
                    "risk_exposure": st.column_config.NumberColumn("Transaction value", format="$%.2f"),
                    "expected_fraud_value": st.column_config.NumberColumn("Expected fraud value", format="$%.2f"),
                },
                hide_index=True,
            )
            st.caption("Expected fraud value is the sum of amount × calibrated fraud probability; it is a triage signal, not an estimate of realised loss.")

    with link_tab:
        case_id = st.selectbox(
            "Case to investigate", scored.sort_values("fraud_probability", ascending=False)["transaction_id"].head(200), key="linked_case"
        )
        case = scored.loc[scored["transaction_id"] == case_id].iloc[0]
        linked = linked_transactions(scored, case_id)
        l1, l2, l3 = st.columns(3)
        l1.metric("Case risk band", risk_band(float(case["fraud_probability"])))
        l2.metric("Potentially linked events", len(linked))
        l3.metric("Linked high-risk events", int((linked.get("fraud_probability", pd.Series(dtype=float)) >= 0.50).sum()))
        if linked.empty:
            st.info("No same-customer or same country/category events were found within 24 hours in this batch.")
        else:
            link_columns = [
                column for column in [
                    "transaction_id", "timestamp", "customer_id", "amount", "merchant_category", "country",
                    "fraud_probability", "recommended_action", "link_reason", "hours_from_case",
                ] if column in linked.columns
            ]
            st.dataframe(linked[link_columns], width="stretch", hide_index=True)
            st.caption("A link is contextual evidence only. Confirm relationships from trusted case records before escalating a case.")

    with st.form("feedback"):
        st.subheader("Analyst feedback")
        transaction_id = st.selectbox("Transaction", scored.sort_values("fraud_probability", ascending=False)["transaction_id"].head(100))
        label = st.radio("Review outcome", ["fraud", "legitimate"], horizontal=True)
        reviewer = st.text_input("Reviewer name", value="analyst-1")
        comment = st.text_input("Optional review note")
        if st.form_submit_button("Save feedback"):
            record_feedback(transaction_id, label, reviewer, comment)
            st.success("Feedback saved. It will be merged during the next governed training run.")


if SCORED_DATA.exists():
    st.subheader("🤖 AI Analyst Case Brief")

    scored_for_llm = pd.read_csv(SCORED_DATA)

    transaction_options = (
        scored_for_llm
        .sort_values("fraud_probability", ascending=False)
        ["transaction_id"]
        .head(100)
        .tolist()
    )

    selected_llm_transaction = st.selectbox(
        "Select a transaction for an AI-generated case brief",
        transaction_options,
        key="llm_transaction",
    )

    if st.button("Generate AI Case Brief", type="primary"):
        row = scored_for_llm[
            scored_for_llm["transaction_id"] == selected_llm_transaction
        ].iloc[0]

        evidence = {
            "transaction_id": row["transaction_id"],
            "amount": row["amount"],
            "merchant_category": row["merchant_category"],
            "payment_method": row["payment_method"],
            "country": row["country"],
            "device_trusted": row["device_trusted"],
            "account_age_days": row["account_age_days"],
            "failed_attempts_24h": row["failed_attempts_24h"],
            "prior_txn_count_1h": row["prior_txn_count_1h"],
            "prior_amount_24h": row["prior_amount_24h"],
            "minutes_since_prior_txn": row["minutes_since_prior_txn"],
            "new_merchant_for_customer": row["new_merchant_for_customer"],
            "fraud_probability": row["fraud_probability"],
            "fraud_prediction": row["fraud_prediction"],
            "recommended_action": row["recommended_action"],
            "risk_reasons": row["risk_reasons"],
            "drift_status": row["drift_status"],
            "escalation_probability": row.get("escalation_probability", 0.0),
            "primary_risk_driver": row.get("primary_risk_driver", "N/A"),
        }

        with st.spinner("Generating analyst case brief..."):
            try:
                brief = generate_case_brief(
                    evidence=evidence,
                    adapter=None,
                    base_model="Qwen/Qwen2.5-1.5B-Instruct",
                    max_new_tokens=200,
                )

                st.success("Case brief generated")
                st.code(brief, language="json")

            except Exception as exc:
                st.error(f"LLM inference failed: {exc}")

if DRIFT_REPORT_PATH.exists():
    drift = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))
    st.subheader("Data drift monitor")
    st.metric("Status", drift["status"], help="PSI ≥ 0.10: monitor; PSI ≥ 0.25: validate recent labels and retrain.")
    st.dataframe(pd.DataFrame(drift["feature_psi"].items(), columns=["feature", "PSI"]), width="stretch")

if IMPORTANCE_PATH.exists():
    st.subheader("Feature importance (permutation PR-AUC impact)")
    importance = pd.read_csv(IMPORTANCE_PATH).head(10).set_index("feature")
    st.bar_chart(importance["importance_mean"])
