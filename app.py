import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from pathlib import Path
from typing import List, Dict, Optional

# --- Config ---
st.set_page_config("🎯 Meta Ads Funnel Dashboard", layout="wide")
st.title("📊 Meta Ads Funnel Analysis")

# --- Constants ---
MAPPING_FILE = Path("campaign_mapping_fixed.csv")
CLASSIFICATION_FIELDS = [
    "ad_set_name", "campaign_name", "ad_name", "result_type", "result_indicator",
    "objective", "optimization_goal", "optimization_event",
    "campaign_objective", "adset_objective"
]

NUMERIC_FIELDS = [
    "amount_spent_(usd)", "impressions", "results", "cost_per_results",
    "clicks", "cpm", "ctr_(link_click-through_rate)"
]

# --- Load Mapping ---
if not MAPPING_FILE.exists():
    st.error("Mapping file not found.")
    st.stop()

mapping_df = pd.read_csv(MAPPING_FILE).fillna("")


class MappingRule:
    def __init__(self, pattern: str, mapping_type: str, mapping_value: str):
        self.compiled = re.compile(pattern, re.IGNORECASE)
        self.mapping_type = mapping_type.lower()
        self.mapping_value = mapping_value.strip()

    def matches(self, text: str) -> bool:
        return bool(self.compiled.search(text))


def build_mapping_rules(table: pd.DataFrame) -> List[MappingRule]:
    rules = []
    for _, row in table.iterrows():
        try:
            rules.append(
                MappingRule(
                    row["regex_pattern"], row["mapping_type"], row["mapping_value"]
                )
            )
        except re.error as e:
            st.warning(f"⚠️ Invalid regex: {row['regex_pattern']} ({e})")
    return rules


MAPPING_RULES = build_mapping_rules(mapping_df)


def normalize_text(row: pd.Series) -> str:
    parts = [str(row.get(f, "")).lower() for f in CLASSIFICATION_FIELDS if pd.notna(row.get(f))]
    return " ".join(parts)


def apply_classification(row: pd.Series) -> pd.Series:
    text = normalize_text(row)
    show = funnel = None
    legacy_labels = []

    for rule in MAPPING_RULES:
        if rule.matches(text):
            if rule.mapping_type == "show" and not show:
                show = rule.mapping_value
            elif rule.mapping_type == "funnel" and not funnel:
                funnel = rule.mapping_value if rule.mapping_value.lower().startswith("funnel") else f"Funnel {rule.mapping_value}"
            elif rule.mapping_type == "legacy":
                legacy_labels.append(rule.mapping_value)

    return pd.Series({
        "show": show or "Unknown",
        "funnel": funnel or "Unclassified",
        "legacy_label": ", ".join(sorted(set(legacy_labels))) if legacy_labels else None,
        "classification": " | ".join(filter(None, [show, funnel] + legacy_labels)) or "Unclassified"
    })


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_per_") for c in df.columns]
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "reporting_starts" in df.columns:
        df["reporting_starts"] = pd.to_datetime(df["reporting_starts"], errors="coerce")
    return df


def parse_hour(value) -> Optional[int]:
    if pd.isna(value): return None
    match = re.search(r"\b(\d{1,2})", str(value))
    return int(match.group(1)) % 24 if match else None


# --- File Upload ---
st.sidebar.header("📁 Upload Meta Ad Exports")
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)

def detect_file_type(df: pd.DataFrame) -> str:
    cols = df.columns.str.lower().tolist()
    if any("time_of_day" in col for col in cols):
        return "days_time"
    if any("placement" in col or "device" in col for col in cols):
        return "days_pd"
    return "days"


def process_file(file) -> Optional[tuple[str, pd.DataFrame]]:
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="latin1")
    df = sanitize_columns(df)
    classification = df.apply(apply_classification, axis=1)
    for col in classification.columns:
        df[col] = classification[col]
    return detect_file_type(df), df


file_map: Dict[str, pd.DataFrame] = {}

if uploaded_files:
    for file in uploaded_files:
        result = process_file(file)
        if result:
            key, df = result
            file_map[key] = pd.concat([file_map.get(key, pd.DataFrame()), df], ignore_index=True)


# --- UI Logic ---
if not file_map:
    st.info("⬆️ Upload at least one Meta export file to continue.")
    st.stop()

# --- KPI Section ---
if "days" in file_map:
    df = file_map["days"]
    st.subheader("🚦 Show Health Indicators")

    shows = sorted(df["show"].dropna().unique())
    selected_show = st.selectbox("Select Show", shows)
    show_df = df[df["show"] == selected_show]

    kpi1 = show_df["amount_spent_(usd)"].sum()
    kpi2 = show_df["results"].sum()
    kpi3 = kpi1 / kpi2 if kpi2 else None
    kpi4 = show_df["clicks"].sum() / kpi2 if kpi2 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${kpi1:,.0f}")
    col2.metric("Tickets Sold", int(kpi2))
    col3.metric("CPA", f"${kpi3:,.2f}" if kpi3 else "N/A")
    col4.metric("Clicks per Ticket", f"{kpi4:.2f}" if kpi4 else "N/A")

    fig = px.line(show_df.sort_values("reporting_starts"), x="reporting_starts", y="cost_per_results",
                  title=f"Daily CPA – {selected_show}")
    st.plotly_chart(fig, use_container_width=True)

# --- Time-of-day ---
if "days_time" in file_map:
    st.subheader("🕒 Time-of-Day Performance")
    df = file_map["days_time"]
    df["hour"] = df["time_of_day_(viewer's_time_zone)"].apply(parse_hour)

    funnel = st.selectbox("Select Funnel", sorted(df["funnel"].dropna().unique()))
    filt = df[df["funnel"] == funnel]

    hourly = filt.groupby("hour").apply(
        lambda x: np.average(x["cost_per_results"], weights=x["results"])
        if x["results"].sum() else x["cost_per_results"].mean()
    ).reset_index(name="avg_cpr")

    fig = px.line(hourly, x="hour", y="avg_cpr", title=f"Avg CPA by Hour – {funnel}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- Placement Performance ---
if "days_pd" in file_map:
    st.subheader("📱 Placement & Device Performance")
    df = file_map["days_pd"]
    funnel = st.selectbox("Select Funnel (Placement)", sorted(df["funnel"].dropna().unique()))

    placement_perf = df[df["funnel"] == funnel].groupby("placement").agg(
        avg_cpr=("cost_per_results", "mean"),
        results=("results", "sum"),
        spend=("amount_spent_(usd)", "sum")
    ).reset_index()

    fig1 = px.bar(placement_perf, x="placement", y="avg_cpr", color="spend", text="results", title="By Placement")
    st.plotly_chart(fig1, use_container_width=True)

    if "impression_device" in df.columns:
        device_perf = df[df["funnel"] == funnel].groupby("impression_device").agg(
            avg_cpr=("cost_per_results", "mean"),
            results=("results", "sum"),
            spend=("amount_spent_(usd)", "sum")
        ).reset_index()

        fig2 = px.bar(device_perf, x="impression_device", y="avg_cpr", color="spend", text="results", title="By Device")
        st.plotly_chart(fig2, use_container_width=True)

st.success("✅ Done! Explore the dashboard using the controls above.")
