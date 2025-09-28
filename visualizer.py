import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_kpis(metrics_dict: dict):
    st.subheader("🎯 Show KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Days to Show", metrics_dict.get("days_to_show"))
    col1.metric("Tickets Sold", metrics_dict.get("sold"))
    col1.metric("Capacity", metrics_dict.get("capacity"))
    col2.metric("Remaining", metrics_dict.get("remaining"))
    col2.metric("Daily Sales Target", round(metrics_dict.get("daily_target", 0), 2))
    col2.metric("Ticket Cost", f"${metrics_dict.get('ticket_cost'):.2f}" if metrics_dict.get("ticket_cost") else "-")
    col3.metric("Total Spend", f"${metrics_dict.get('spend_total'):.2f}")
    col3.metric("ROAS", round(metrics_dict.get("roas", 0), 2) if metrics_dict.get("roas") else "-")

def plot_daily_sales(sales_df: pd.DataFrame, show_id: str):
    df = sales_df.copy()
    df = df[df["show_id"] == show_id]
    if "date" not in df.columns or "daily_sales" not in df.columns:
        st.info("No daily sales data available.")
        return
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["daily_sales"], marker="o")
    ax.set_title("Daily Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tickets Sold")
    ax.grid(True)
    st.pyplot(fig)

def plot_funnel_efficiency(funnel_metrics: dict):
    st.subheader("🔁 Funnel Efficiency")
    labels = list(funnel_metrics.keys())
    values = list(funnel_metrics.values())
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Per Ticket")
    ax.set_title("Funnel: Views / Add to Cart / Sales")
    st.pyplot(fig)
