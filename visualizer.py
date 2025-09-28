"""
Visualizer
----------
Creates Streamlit visual components (KPIs, charts).
"""

import streamlit as st
import plotly.express as px

def show_kpi_cards(kpis: dict):
    """
    Display KPIs as metric cards in Streamlit.
    """
    cols = st.columns(len(kpis))
    for (key, value), col in zip(kpis.items(), cols):
        col.metric(label=key, value=value)

def plot_timeseries(sales_df, ads_df):
    """
    Plot sales and spend over time.
    """
    if "date" in ads_df.columns:
        fig = px.line(ads_df, x="date", y=["spend", "clicks"], title="Ads Spend & Clicks Over Time")
        st.plotly_chart(fig, use_container_width=True)

    if not sales_df.empty and "Date" in sales_df.columns:
        fig2 = px.line(sales_df, x="Date", y=sales_df.columns[-1], title="Sales Over Time")
        st.plotly_chart(fig2, use_container_width=True)

def plot_funnel(ads_df):
    """
    Simple funnel plot for Ads data.
    """
    if {"impressions", "clicks", "conversions"} <= set(ads_df.columns):
        funnel_data = {
            "stage": ["Impressions", "Clicks", "Conversions"],
            "value": [
                ads_df["impressions"].sum(),
                ads_df["clicks"].sum(),
                ads_df["conversions"].sum()
            ]
        }
        fig = px.funnel(funnel_data, x="value", y="stage", title="Ads Funnel")
        st.plotly_chart(fig, use_container_width=True)
