import streamlit as st
import pandas as pd
import tempfile
import os
import sys

sys.path.append(os.path.dirname(__file__))

from data_loader import load_ads_data
from visualizer import show_kpis, plot_roas, plot_cpa, plot_conversion_funnel

st.set_page_config(page_title="🎯 Ads Performance Analyzer", layout="wide")

st.title("📈 Ads Performance Analyzer")
st.markdown("""
Upload your campaign CSVs exported from Meta Ads Manager.  
The app will auto-parse show IDs, normalize funnel stages, and show ROAS, CPA, and funnel performance.
""")

uploaded_files = st.file_uploader("📂 Upload multiple ad CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                file_path = os.path.join(tmpdir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            df = load_ads_data(tmpdir)

    if df.empty:
        st.error("❌ No valid ad data could be parsed.")
    else:
        st.success(f"✅ Loaded {len(df):,} records from {len(uploaded_files)} files.")

        show_kpis(df)
        plot_roas(df)
        plot_cpa(df)

        st.subheader("📍 Funnel Breakdown")
        selected_show = st.selectbox("Select Show ID", df["show_id"].dropna().unique())
        plot_conversion_funnel(df, selected_show)
else:
    st.info("Please upload one or more CSV files to begin.")