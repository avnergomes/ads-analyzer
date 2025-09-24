import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import re

st.set_page_config(page_title="Meta Ads Funnel & Show Analysis", layout="wide")

st.title("📊 Meta Ads Funnel & Show Analysis")

# --- Upload Section ---
st.sidebar.header("Upload Meta Ads Files")
days_file = st.sidebar.file_uploader("Upload Days.csv", type="csv")
placement_device_file = st.sidebar.file_uploader("Upload Days + Placement + Device.csv", type="csv")
time_file = st.sidebar.file_uploader("Upload Days + Time.csv", type="csv")

# --- Google Sheet (Ticket Sales) ---
sheet_url = "https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/export?format=csv"

if days_file and placement_device_file and time_file:
    # --- Load CSVs ---
    days = pd.read_csv(days_file)
    placement_device = pd.read_csv(placement_device_file)
    time_of_day = pd.read_csv(time_file)

    # --- Clean column names ---
    def clean_columns(df):
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
                      .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        return df

    days = clean_columns(days)
    placement_device = clean_columns(placement_device)
    time_of_day = clean_columns(time_of_day)

    # --- Standardize keys ---
    for df in [days, placement_device, time_of_day]:
        df["ad_set_same"] = df["ad_set_name"].str.strip()
        df["reporting_starts"] = pd.to_datetime(df["reporting_starts"], errors="coerce")

    # --- Aggregations ---
    placement_device_base = placement_device.groupby(
        ["ad_set_same", "reporting_starts"], as_index=False
    ).agg({
        "impressions": "sum",
        "link_clicks": "sum",
        "amount_spent_usd": "sum",
        "impression_device": lambda x: list(pd.Series(x).dropna().unique()),
        "placement": lambda x: list(pd.Series(x).dropna().unique())
    })

    time_base = time_of_day.groupby(
        ["ad_set_same", "reporting_starts"], as_index=False
    ).agg({
        "impressions": "sum",
        "link_clicks": "sum",
        "amount_spent_usd": "sum",
        "time_of_day_viewers_time_zone": lambda x: list(pd.Series(x).dropna().unique())
    })

    merged = days.merge(
        placement_device_base,
        on=["ad_set_same", "reporting_starts"],
        how="left",
        suffixes=("", "_placement")
    ).merge(
        time_base,
        on=["ad_set_same", "reporting_starts"],
        how="left",
        suffixes=("", "_time")
    )

    # --- Funnel classification robust ---
    def classify_funnel_robust(name: str) -> str:
        if pd.isna(name):
            return "Unclassified"
        n = str(name).lower()
        if re.search(r"(f1|fun1|lpview|lpviews)", n):
            return "F1_LPView"
        elif re.search(r"(f2|fun2|addtocart)", n) and "conv" not in n:
            return "F2_AddtoCart"
        elif re.search(r"(f3|fun3)", n) or ("conv" in n and "addtocart" in n):
            return "F3_Conversion"
        elif "interest" in n:
            return "Legacy_Interest"
        elif "target" in n:
            return "Legacy_Target"
        else:
            return "Unclassified"

    merged["funnel"] = merged["ad_set_name"].apply(classify_funnel_robust)

    # --- Show parser extended ---
    def normalize_show_id_extended(name: str) -> str:
        if pd.isna(name):
            return "Unknown_Show"
        n = str(name).lower()
        base_match = re.match(r"([A-Z]{2,3}_[0-9]{4})", str(name))
        if base_match:
            return base_match.group(1)
        if "_dc_" in n or "dc" in n:
            return "WDC"
        if "_sea_" in n or "seattle" in n:
            return "SEA"
        if "_tr_" in n or "toronto" in n:
            return "TR"
        if "_pdx_" in n or "portland" in n:
            return "PDX"
        if "_edm_" in n or "edmonton" in n:
            return "EDM"
        if "_smf_" in n or "sacramento" in n:
            return "SMF"
        if "_cmh_" in n or "columbus" in n:
            return "CMH"
        if "sunrose" in n:
            return "SUN"
        if "upstairs" in n:
            return "UPS"
        return "Unknown_Show"

    merged["show_id"] = merged["ad_set_name"].apply(normalize_show_id_extended)

    # --- Ticket Sales Integration ---
    ticket_sales = pd.read_csv(sheet_url)
    ticket_sales.columns = ticket_sales.columns.str.strip().str.lower().str.replace(" ", "_")

    if "showid" in ticket_sales.columns:
        merged = merged.merge(ticket_sales, how="left", left_on="show_id", right_on="showid")

        # --- KPI calculations ---
        merged["ticket_cost"] = merged["show_budget"] / merged["total_sold"]
        merged["daily_sales_target"] = (merged["capacity"] - merged["total_sold"]) / merged["days_to_show"]
        merged["funnel_eff_clicks"] = merged["link_clicks"] / merged["total_sold"]
        merged["funnel_eff_lpviews"] = merged["impressions"] / merged["total_sold"]
        merged["cpa_daily"] = merged["amount_spent_usd"] / merged["total_sold"]
        merged["revenue"] = merged["avg_ticket_price"] * merged["capacity"]
        merged["roas"] = merged["revenue"] / merged["amount_spent_usd"]

    st.success("✅ Data cleaned, merged and integrated with Ticket Sales!")

    # --- Show selector ---
    show_list = merged["show_id"].dropna().unique().tolist()
    selected_show = st.sidebar.selectbox("Select Show", show_list)

    df_show = merged[merged["show_id"] == selected_show]

    # --- Health of Show KPIs ---
    st.subheader(f"Health of Show: {selected_show}")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🎟 Tickets Sold", int(df_show["total_sold"].dropna().mean()) if "total_sold" in df_show else 0)
    col2.metric("💰 Ticket Cost", round(df_show["ticket_cost"].dropna().mean(), 2) if "ticket_cost" in df_show else 0)
    col3.metric("📈 ROAS", round(df_show["roas"].dropna().mean(), 2) if "roas" in df_show else 0)
    col4.metric("📊 CPA Daily", round(df_show["cpa_daily"].dropna().mean(), 2) if "cpa_daily" in df_show else 0)
    col5.metric("🎯 Daily Target", round(df_show["daily_sales_target"].dropna().mean(), 2) if "daily_sales_target" in df_show else 0)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 Funnel Decay", "📱 Devices & Placements", "⏰ Time of Day", "🎭 Show Summary"]
    )

    # Funnel decay
    with tab1:
        st.subheader("Funnel Decay Over Time")
        funnel_trend = df_show.groupby(["reporting_starts", "funnel"], as_index=False).agg({
            "impressions": "sum",
            "link_clicks": "sum",
            "amount_spent_usd": "sum"
        })
        fig = px.line(funnel_trend, x="reporting_starts", y="impressions", color="funnel", title="Impressions over Time")
        st.plotly_chart(fig, use_container_width=True)

    # Devices & Placements
    with tab2:
        st.subheader("Device & Placement Breakdown")
        if "impression_device" in placement_device:
            device_counts = placement_device.explode("impression_device")["impression_device"].value_counts().reset_index()
            device_counts.columns = ["device", "count"]
            st.bar_chart(device_counts.set_index("device"))
        if "placement" in placement_device:
            placement_counts = placement_device.explode("placement")["placement"].value_counts().reset_index()
            placement_counts.columns = ["placement", "count"]
            st.bar_chart(placement_counts.set_index("placement"))

    # Time of Day
    with tab3:
        st.subheader("Performance by Time of Day")
        if "time_of_day_viewers_time_zone" in time_of_day:
            time_counts = time_of_day["time_of_day_viewers_time_zone"].value_counts().reset_index()
            time_counts.columns = ["time_slot", "count"]
            chart = alt.Chart(time_counts).mark_bar().encode(
                x="time_slot", y="count", tooltip=["time_slot", "count"]
            ).properties(width=800)
            st.altair_chart(chart, use_container_width=True)

    # Show summary
    with tab4:
        st.subheader("Ticket Sales by Show")
        if "showid" in ticket_sales.columns:
            show_sales = ticket_sales[["showid", "total_sold", "capacity", "avg_ticket_price"]]
            st.dataframe(show_sales)
            fig_sales = px.bar(show_sales, x="showid", y="total_sold", title="Tickets Sold per Show")
            st.plotly_chart(fig_sales, use_container_width=True)

else:
    st.info("⬆️ Please upload the three CSV files in the sidebar to begin.")
