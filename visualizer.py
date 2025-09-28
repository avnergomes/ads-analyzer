
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def show_kpis(df: pd.DataFrame):
    st.subheader("📊 Health Indicators Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_spend = df['spend'].sum()
        st.metric("Total Spend ($)", f"{total_spend:,.0f}")

    with col2:
        total_conversions = df['conversions'].sum()
        st.metric("Total Conversions", int(total_conversions))

    with col3:
        cpa = total_spend / total_conversions if total_conversions else 0
        st.metric("CPA ($)", f"{cpa:,.2f}")

    st.divider()

def plot_roas(df: pd.DataFrame):
    df_grouped = df.groupby("show_id").agg({
        "spend": "sum",
        "conversions": "sum"
    }).reset_index()
    df_grouped["ROAS"] = df_grouped["conversions"] / df_grouped["spend"]

    fig, ax = plt.subplots()
    ax.bar(df_grouped["show_id"], df_grouped["ROAS"])
    ax.set_ylabel("ROAS")
    ax.set_title("ROAS by Show")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_cpa(df: pd.DataFrame):
    df_grouped = df.groupby("show_id").agg({
        "spend": "sum",
        "conversions": "sum"
    }).reset_index()
    df_grouped["CPA"] = df_grouped["spend"] / df_grouped["conversions"].replace(0, 1)

    fig, ax = plt.subplots()
    ax.bar(df_grouped["show_id"], df_grouped["CPA"])
    ax.set_ylabel("CPA")
    ax.set_title("CPA by Show")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_conversion_funnel(df: pd.DataFrame, show_id: str):
    filtered = df[df["show_id"] == show_id]
    steps = {
        "Clicks": filtered["clicks"].sum(),
        "LP Views": filtered["lp_views"].sum(),
        "Add to Cart": filtered["add_to_cart"].sum(),
        "Conversions": filtered["conversions"].sum()
    }

    fig, ax = plt.subplots()
    ax.bar(steps.keys(), steps.values())
    ax.set_ylabel("Count")
    ax.set_title(f"Funnel Breakdown - {show_id}")
    st.pyplot(fig)
