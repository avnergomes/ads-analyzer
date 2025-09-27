import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="Meta Ads Funnel Analysis", layout="wide")
st.title("📊 Meta Ads Funnel Analysis Dashboard")

# --------------------------------
# Load mapping file
# --------------------------------
MAPPING_FILE = Path("campaign_mapping_refined.csv")
if not MAPPING_FILE.exists():
    st.error("❌ Mapping file 'campaign_mapping_refined.csv' not found in repo.")
    st.stop()

mapping_df = pd.read_csv(MAPPING_FILE)

def classify_name(name):
    """Classify campaign/adset/ad name using regex mapping."""
    s = str(name)
    for _, row in mapping_df.iterrows():
        if re.search(row["regex_pattern"], s, re.IGNORECASE):
            if row["mapping_type"] == "funnel":
                return f"Funnel {row['mapping_value']}"
            return row["mapping_value"]
    return "Unclassified"

def parse_hour(s):
    try:
        return int(str(s).split(":")[0])
    except:
        return None

def compute_decay(df):
    """Compute funnel decay (number of good days before drop) per adset."""
    out = []
    if "ad_set_name" not in df.columns:
        return pd.DataFrame()
    for adset, g in df.groupby("ad_set_name"):
        if "reporting_starts" not in g.columns or "cost_per_results" not in g.columns:
            continue
        g = g.sort_values("reporting_starts")
        base = g["cost_per_results"].replace([np.inf, -np.inf], np.nan).dropna().head(3).median()
        if pd.isna(base) or base <= 0:
            continue
        g["is_good"] = g["cost_per_results"] <= 1.3 * base
        good_days, bad_run = 0, 0
        for ok in g["is_good"]:
            if ok:
                good_days += 1
                bad_run = 0
            else:
                bad_run += 1
                if bad_run >= 3:
                    break
        out.append({
            "ad_set_name": adset,
            "funnel": g["funnel"].iloc[0] if "funnel" in g else "Unclassified",
            "show": g["show"].iloc[0] if "show" in g else None,
            "good_days_before_drop": good_days
        })
    return pd.DataFrame(out)

# --------------------------------
# File uploaders
# --------------------------------
st.sidebar.header("Upload your Meta CSV exports")
days_file = st.sidebar.file_uploader("Days.csv", type="csv")
days_time_file = st.sidebar.file_uploader("Days + Time.csv", type="csv")
days_pd_file = st.sidebar.file_uploader("Days + Placement + Device.csv", type="csv")

uploaded_dfs = {}

def load_csv(uploaded_file, name):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_per_") for c in df.columns]
        if "reporting_starts" in df.columns:
            df["reporting_starts"] = pd.to_datetime(df["reporting_starts"], errors="coerce")
        # Apply classification
        if "ad_set_name" in df.columns:
            df["classification"] = df["ad_set_name"].apply(classify_name)
            df["funnel"] = df["classification"].apply(lambda x: x if "Funnel" in x else "Unclassified")
            df["show"] = df["classification"].apply(
                lambda x: x if "Funnel" not in x and x not in ["Interest", "Target", "Unclassified"] else None
            )
        uploaded_dfs[name] = df

# Load files
load_csv(days_file, "days")
load_csv(days_time_file, "days_time")
load_csv(days_pd_file, "days_pd")

if not uploaded_dfs:
    st.info("⬆️ Please upload at least one Meta Ads export file to start.")
    st.stop()

# --------------------------------
# Analyses
# --------------------------------

# 1. Funnel Decay Analysis
if "days" in uploaded_dfs:
    st.subheader("⏳ Funnel Decay Analysis")
    decay_df = compute_decay(uploaded_dfs["days"])
    if not decay_df.empty:
        fig = px.box(decay_df, x="funnel", y="good_days_before_drop",
                     title="Distribution of Good Days Before Drop by Funnel",
                     points="all", color="funnel")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(decay_df.groupby("funnel")["good_days_before_drop"].describe())
    else:
        st.info("Not enough data to compute funnel decay.")

# 2. Country/Show × Funnel Overview
if "days" in uploaded_dfs:
    st.subheader("🌍 Show × Funnel Overview")
    df = uploaded_dfs["days"]
    if {"funnel", "show", "amount_spent_(usd)", "impressions", "results", "cost_per_results"}.issubset(df.columns):
        overview = df.groupby(["show", "funnel"]).agg(
            spend=("amount_spent_(usd)", "sum"),
            impressions=("impressions", "sum"),
            results=("results", "sum"),
            avg_cpr=("cost_per_results", "mean")
        ).reset_index()
        fig = px.scatter(overview, x="spend", y="results", size="impressions",
                         color="funnel", facet_col="show",
                         hover_data=["avg_cpr"],
                         title="Spend vs Results by Show & Funnel")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(overview)
    else:
        st.warning("Some required columns are missing from Days.csv export.")

# 3. Time-of-Day Performance
if "days_time" in uploaded_dfs:
    st.subheader("🕒 Time-of-Day Performance")
    df = uploaded_dfs["days_time"]
    if "time_of_day_(viewer's_time_zone)" in df.columns and "cost_per_results" in df.columns:
        df["hour"] = df["time_of_day_(viewer's_time_zone)"].apply(parse_hour)
        hour_perf = df.groupby(["funnel", "hour"]).apply(
            lambda x: np.average(x["cost_per_results"], weights=x.get("results", None))
            if "results" in x and x["results"].sum() > 0 else x["cost_per_results"].mean()
        ).reset_index(name="avg_cpr")

        funnel_sel = st.selectbox("Funnel (Time)", hour_perf["funnel"].unique())
        filt = hour_perf[hour_perf["funnel"] == funnel_sel]
        fig_t = px.line(filt, x="hour", y="avg_cpr", markers=True,
                        title=f"Avg Cost per Result by Hour ({funnel_sel})")
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(filt)
    else:
        st.info("No valid time-of-day data available.")

# 4. Placement Performance
if "days_pd" in uploaded_dfs:
    st.subheader("📱 Placement Performance")
    df = uploaded_dfs["days_pd"]
    if {"placement", "cost_per_results", "results", "amount_spent_(usd)", "funnel"}.issubset(df.columns):
        placement_perf = df.groupby(["funnel", "placement"]).agg(
            avg_cpr=("cost_per_results", "mean"),
            results=("results", "sum"),
            spend=("amount_spent_(usd)", "sum")
        ).reset_index()
        funnel_sel_p = st.selectbox("Funnel (Placement)", placement_perf["funnel"].unique())
        filt_p = placement_perf[placement_perf["funnel"] == funnel_sel_p]
        fig_p = px.bar(filt_p.sort_values("avg_cpr"), x="placement", y="avg_cpr",
                       color="spend", text="results",
                       title=f"Placement Efficiency ({funnel_sel_p})")
        st.plotly_chart(fig_p, use_container_width=True)
        st.dataframe(filt_p)
    else:
        st.warning("Placement columns not found in Days + Placement + Device export.")

# 5. Device Performance
if "days_pd" in uploaded_dfs:
    st.subheader("💻 Device Performance")
    df = uploaded_dfs["days_pd"]
    if {"impression_device", "cost_per_results", "results", "amount_spent_(usd)", "funnel"}.issubset(df.columns):
        device_perf = df.groupby(["funnel", "impression_device"]).agg(
            avg_cpr=("cost_per_results", "mean"),
            results=("results", "sum"),
            spend=("amount_spent_(usd)", "sum")
        ).reset_index()
        funnel_sel_d = st.selectbox("Funnel (Device)", device_perf["funnel"].unique())
        filt_d = device_perf[device_perf["funnel"] == funnel_sel_d]
        fig_d = px.bar(filt_d.sort_values("avg_cpr"), x="impression_device", y="avg_cpr",
                       color="spend", text="results",
                       title=f"Device Efficiency ({funnel_sel_d})")
        st.plotly_chart(fig_d, use_container_width=True)
        st.dataframe(filt_d)
    else:
        st.warning("Device columns not found in Days + Placement + Device export.")

st.success("✅ Analyses complete. Explore interactively!")
