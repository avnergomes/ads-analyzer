"""Streamlit entry-point for the Ads Performance Analyzer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import (
    clean_ads_dataframe,
    clean_sales_dataframe,
    load_ads_data_from_files,
    load_ads_data_from_folder,
    load_sales_data_from_sheet,
)
from mapper import get_unmapped_campaigns, map_campaigns
from metrics import compute_funnel_efficiency, summarize_show_metrics
from visualizer import plot_campaign_spend, plot_daily_sales, plot_funnel_efficiency, show_kpis

st.set_page_config(page_title="🎯 Ads Performance Analyzer", layout="wide")

st.title("📈 Ads Performance Analyzer")
st.markdown(
    """
This dashboard combines Meta Ads results with sales metrics so the team can identify
which campaigns drive each show.
"""
)

with st.sidebar:
    st.header("1. Campaign data")
    data_source = st.radio(
        "Choose the ads CSV source",
        ("Repository samples", "Upload files"),
    )

    uploaded_files = None
    samples_folder = Path("samples")
    if data_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload one or more CSVs exported from Meta Ads",
            type="csv",
            accept_multiple_files=True,
        )
    else:
        default_folder = Path("samples")
        samples_folder = Path(st.text_input("Folder with CSVs", value=str(default_folder)))

    st.header("2. Sales spreadsheet (optional)")
    sheet_url = st.text_input("Google Sheet URL")
    credentials_path = st.text_input("Credentials JSON path", value="credentials.json")
    uploaded_credentials = st.file_uploader("Or upload the credentials JSON", type="json")

    if uploaded_credentials is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded_credentials.getbuffer())
            credentials_path = tmp.name
            st.session_state["google_credentials_path"] = credentials_path
    elif "google_credentials_path" in st.session_state:
        credentials_path = st.session_state["google_credentials_path"]

    st.header("3. Manual campaign mapping (optional)")
    manual_mapping_path = st.text_input(
        "Manual mapping CSV path",
        value="campaign_mapping_fixed.csv",
        help="Provide a CSV with two columns: campaign name (or regex) and the target show ID.",
    )
    uploaded_mapping = st.file_uploader(
        "Or upload a mapping CSV", type="csv", key="mapping_uploader"
    )

    manual_mapping_path = manual_mapping_path.strip()
    if not manual_mapping_path:
        manual_mapping_path = "campaign_mapping_fixed.csv"

    if uploaded_mapping is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_mapping.getbuffer())
            manual_mapping_path = tmp.name
            st.session_state["manual_mapping_path"] = manual_mapping_path
    elif "manual_mapping_path" in st.session_state:
        manual_mapping_path = st.session_state["manual_mapping_path"]

    mapping_file_exists = Path(manual_mapping_path).expanduser().exists()
    if manual_mapping_path:
        if mapping_file_exists:
            st.caption(f"Using mapping file: {manual_mapping_path}")
        else:
            st.info("Mapping file not found. Default rules will be used unless a CSV is uploaded.")

st.divider()

with st.spinner("Loading campaign data..."):
    if data_source == "Upload files" and uploaded_files:
        ads_df = load_ads_data_from_files(uploaded_files)
    else:
        ads_df = load_ads_data_from_folder(samples_folder)

ads_df = clean_ads_dataframe(ads_df)

if ads_df.empty:
    st.warning("No valid campaign data found.")
    st.stop()

st.success(f"{len(ads_df):,} campaign rows loaded.")
with st.expander("Campaign data preview"):
    st.dataframe(ads_df.head(100))

sales_df = pd.DataFrame()
if sheet_url and credentials_path:
    try:
        with st.spinner("Fetching Google Sheet..."):
            sales_df = load_sales_data_from_sheet(sheet_url, credentials_path)
        sales_df = clean_sales_dataframe(sales_df)
        if sales_df.empty:
            st.warning("Sheet loaded but no valid show rows were found.")
        else:
            st.success(f"{len(sales_df):,} sales records loaded.")
            with st.expander("Spreadsheet preview"):
                st.dataframe(sales_df.head(50))
    except Exception as exc:  # pragma: no cover - requires real credentials
        st.error(f"Unable to access the spreadsheet: {exc}")
        sales_df = pd.DataFrame()
else:
    st.info("Provide the URL and credentials JSON to enable sales data.")

if sales_df.empty:
    st.stop()

with st.spinner("Mapping campaigns to shows..."):
    ads_mapped = map_campaigns(
        ads_df,
        sales_df,
        manual_mapping_path=manual_mapping_path,
    )

if ads_mapped["mapped_show_id"].notnull().sum() == 0:
    st.warning("No campaigns were mapped to a show. Adjust the rules or inspect the data.")

with st.expander("Unmapped campaigns"):
    unmapped = get_unmapped_campaigns(ads_mapped)
    if unmapped.empty:
        st.success("All campaigns were mapped!")
    else:
        st.warning(f"{len(unmapped)} campaigns without an associated show.")
        st.dataframe(unmapped[["campaign_name", "source_file"]].drop_duplicates())

available_shows = (
    ads_mapped["mapped_show_id"].dropna().astype(str).str.strip().sort_values().unique().tolist()
)
if not available_shows:
    st.stop()

selected_show = st.selectbox("Select a show", options=available_shows)
show_ads = ads_mapped[ads_mapped["mapped_show_id"].astype(str).str.strip() == selected_show]
show_sales = sales_df[sales_df["show_id"].astype(str).str.strip() == selected_show]

if show_sales.empty:
    st.error("The selected show was not found in the spreadsheet. Check the ID.")
    st.stop()

show_sales_row = show_sales.iloc[0]
spend_total = float(show_ads["amount_spent"].sum())
metrics = summarize_show_metrics(show_sales_row, spend_total)
show_kpis(metrics)

funnel_metrics = compute_funnel_efficiency(show_ads, show_sales_row)

col1, col2 = st.columns(2)
with col1:
    plot_funnel_efficiency(funnel_metrics)
with col2:
    plot_campaign_spend(ads_mapped, selected_show)

plot_daily_sales(sales_df, selected_show)
