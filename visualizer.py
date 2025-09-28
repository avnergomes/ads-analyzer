"""Visualization helpers rendered inside the Streamlit UI."""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


_DEF_METRIC_PLACEHOLDER = "-"


def _format_currency(value: float | None) -> str:
    if value is None:
        return _DEF_METRIC_PLACEHOLDER
    return f"$ {value:,.2f}"


def _format_number(value: float | None) -> str:
    if value is None:
        return _DEF_METRIC_PLACEHOLDER
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def show_kpis(metrics_dict: Mapping[str, float | None]) -> None:
    """Render the KPI cards for the selected show."""
    st.subheader("🎯 Show KPIs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Days until show", _format_number(metrics_dict.get("days_to_show")))
        st.metric("Tickets sold", _format_number(metrics_dict.get("sold")))
        st.metric("Capacity", _format_number(metrics_dict.get("capacity")))

    with col2:
        st.metric("Tickets remaining", _format_number(metrics_dict.get("remaining")))
        st.metric("Daily target", _format_number(metrics_dict.get("daily_target")))
        st.metric("Cost per ticket", _format_currency(metrics_dict.get("ticket_cost")))

    with col3:
        st.metric("Spend", _format_currency(metrics_dict.get("spend_total")))
        roas = metrics_dict.get("roas")
        st.metric("ROAS", _format_number(roas if roas is not None else None))

    st.divider()


def plot_daily_sales(sales_df: pd.DataFrame, show_id: str) -> None:
    if sales_df.empty:
        st.info("No daily sales information available.")
        return

    df = sales_df.copy()
    df = df[df["show_id"] == show_id]
    if df.empty or "date" not in df.columns or "daily_sales" not in df.columns:
        st.info("Spreadsheet is missing 'date' and 'daily_sales' columns required for the daily trend.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        st.info("No valid sales dates to display.")
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["date"], df["daily_sales"], marker="o", linewidth=2)
    ax.set_title("Daily sales trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tickets sold")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)


def plot_funnel_efficiency(funnel_metrics: Mapping[str, float | None]) -> None:
    st.subheader("🔁 Funnel efficiency")

    labels = list(funnel_metrics.keys())
    values = [value if value is not None else 0 for value in funnel_metrics.values()]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color="#5DADE2")
    ax.set_ylabel("Value")
    ax.set_title("Per-ticket indicators")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, funnel_metrics.values()):
        if value is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}",
                    ha="center", va="bottom", fontsize=8)

    st.pyplot(fig)


def plot_campaign_spend(df_ads: pd.DataFrame, show_id: str) -> None:
    st.subheader("💸 Spend by campaign")

    filtered = df_ads[df_ads["mapped_show_id"] == show_id]
    if filtered.empty:
        st.info("No spend recorded for campaigns mapped to this show.")
        return

    chart_data = (
        filtered.groupby("campaign_name")["amount_spent"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    chart_data.sort_values().plot(kind="barh", ax=ax, color="#58D68D")
    ax.set_xlabel("Spend (USD)")
    ax.set_ylabel("Campaign")
    ax.set_title("Top 10 campaigns by spend")
    st.pyplot(fig)
