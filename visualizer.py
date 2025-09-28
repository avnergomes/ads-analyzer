from __future__ import annotations

from typing import Mapping

import altair as alt
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

    base_chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("daily_sales:Q", title="Tickets sold"),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("daily_sales:Q", title="Tickets sold")],
        )
        .properties(title="Daily sales trend", height=260)
    )

    st.altair_chart(base_chart, use_container_width=True)


def plot_funnel_efficiency(funnel_metrics: Mapping[str, float | None]) -> None:
    st.subheader("🔁 Funnel efficiency")

    data = pd.DataFrame(
        [
            {"metric": label, "value": value if value is not None else 0}
            for label, value in funnel_metrics.items()
        ]
    )

    bars = (
        alt.Chart(data)
        .mark_bar(color="#5DADE2")
        .encode(
            x=alt.X("metric:N", title="Metric", sort=None),
            y=alt.Y("value:Q", title="Value"),
            tooltip=[alt.Tooltip("metric:N", title="Metric"), alt.Tooltip("value:Q", title="Value")],
        )
        .properties(title="Per-ticket indicators", height=260)
    )

    labels = bars.mark_text(align="center", baseline="bottom", dy=-4).encode(text=alt.Text("value:Q", format=".2f"))

    st.altair_chart(bars + labels, use_container_width=True)


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

    chart_df = chart_data.reset_index().rename(columns={"campaign_name": "campaign", "amount_spent": "spend"})

    bars = (
        alt.Chart(chart_df)
        .mark_bar(color="#58D68D")
        .encode(
            x=alt.X("spend:Q", title="Spend (USD)"),
            y=alt.Y("campaign:N", sort="-x", title="Campaign"),
            tooltip=[alt.Tooltip("campaign:N", title="Campaign"), alt.Tooltip("spend:Q", title="Spend (USD)")],
        )
        .properties(title="Top 10 campaigns by spend", height=320)
    )

    labels = bars.mark_text(align="left", dx=3).encode(text=alt.Text("spend:Q", format="$.2f"))

    st.altair_chart(bars + labels, use_container_width=True)
