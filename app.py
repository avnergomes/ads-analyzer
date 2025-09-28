"""
Ads Analyzer - Streamlit App
----------------------------
Main entry point for the Ads Analyzer dashboard.
"""

import streamlit as st
import pandas as pd
from data_loader import load_sales_from_sheets, load_ads_file
from parser import normalize_columns
from metrics import compute_kpis
from visualizer import show_kpi_cards, plot_timeseries, plot_funnel

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Ads Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Ads Analyzer")
st.markdown("Interactive dashboard combining Google Sheets sales data with Meta Ads performance.")

# --------------------------
# LOAD SALES DATA (Google Sheets)
# --------------------------
sales_url = "https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/export?format=csv&gid=0"
sales_df = load_sales_from_sheets(sales_url)

st.subheader("🛒 Sales Data (Google Sheets)")
st.dataframe(sales_df.head())

# --------------------------
# LOAD ADS DATA (Upload)
# --------------------------
uploaded_file = st.file_uploader("Upload your Meta Ads file (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    ads_df = load_ads_file(uploaded_file)
    ads_df = normalize_columns(ads_df)

    st.subheader("📄 Ads Data Preview")
    st.dataframe(ads_df.head())

    # --------------------------
    # COMPUTE KPIs
    # --------------------------
    st.subheader("📈 KPIs")
    kpis = compute_kpis(sales_df, ads_df)
    show_kpi_cards(kpis)

    # --------------------------
    # VISUALIZATIONS
    # --------------------------
    st.subheader("📊 Visualizations")
    plot_timeseries(sales_df, ads_df)
    plot_funnel(ads_df)
else:
    st.info("Please upload your Meta Ads file to begin analysis.")
