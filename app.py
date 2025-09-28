import streamlit as st
import pandas as pd
from data_loader import load_ads_data
from visualizer import show_kpis, plot_roas, plot_cpa, plot_conversion_funnel

st.set_page_config(page_title="🎯 Ads Performance Analyzer", layout="wide")

st.title("📈 Ads Performance Analyzer")
st.markdown(\"""
Upload your folder of CSV campaign files.  
The app will auto-parse shows, normalize funnel steps, and show ROAS, CPA, and health indicators.
\""")

# File upload
folder = st.text_input("📂 Enter path to ads data folder:", value="/mnt/data/samples/samples")

if folder:
    with st.spinner("Loading and cleaning data..."):
        df = load_ads_data(folder)
    
    if df.empty:
        st.error("No valid data found in the folder.")
    else:
        st.success(f"✅ Loaded {len(df):,} rows.")

        show_kpis(df)
        plot_roas(df)
        plot_cpa(df)

        st.subheader("📍 Funnel Breakdown")
        selected_show = st.selectbox("Select Show ID", df["show_id"].dropna().unique())
        plot_conversion_funnel(df, selected_show)