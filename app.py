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

# --- Helper to clean column names ---
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

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

# --- Safe metric helper ---
def safe_metric(col, label, func):
    try:
        val = func()
        if pd.isna(val):
            val = "N/A"
    except Exception:
        val = "N/A"
    col.metric(label, val)

# --- Main App ---
if days_file and placement_device_file and time_file:
    # Load CSVs
    days = pd.read_csv(days_file)
    placement_device = pd.read_csv(placement_device_file)
    time_of_day = pd.read_csv(time_file)

    days = clean_columns(days)
    placement_device = clean_columns(placement_device)
    time_of_day = clean_columns(time_of_day)

    for df in [days, placement_device, time_of_day]:
        if "ad_set_name" in df.columns:
            df["ad_set_same"] = df["ad_set_name"].astype(str).str.strip()
        if "reporting_starts" in df.columns:
            df["reporting_starts"] = pd.to_datetime(df["reporting_starts"], errors="coerce")

    # Aggregations
    def safe_groupby(df, cols, agg_dict):
        if all(col in df.columns for col in cols):
            return df.groupby(cols, as_index=False).agg(agg_dict)
        return pd.DataFrame()

    placement_device_base = safe_groupby(
        placement_device, ["ad_set_same", "reporting_starts"],
        {"impressions": "sum", "link_clicks": "sum", "amount_spent_usd": "sum"}
    )
    time_base = safe_groupby(
        time_of_day, ["ad_set_same", "reporting_starts"],
        {"impressions": "sum", "link_clicks": "sum", "amount_spent_usd": "sum"}
    )

    merged = days.copy()
    if not placement_device_base.empty:
        merged = merged.merge(placement_device_base, on=["ad_set_same", "reporting_starts"], how="left", suffixes=("", "_placement"))
    if not time_base.empty:
        merged = merged.merge(time_base, on=["ad_set_same", "reporting_starts"], how="left", suffixes=("", "_time"))

    merged["funnel"] = merged.get("ad_set_name", "").apply(classify_funnel_robust)
    merged["show_id"] = merged.get("ad_set_name", "").apply(normalize_show_id_extended)

    # Try to load Ticket Sales Sheet
    try:
        ticket_sales = pd.read_csv(sheet_url)
        ticket_sales = ticket_sales.dropna(how="all")

        # Normalização de colunas
        ticket_sales.columns = ticket_sales.columns.str.strip().str.lower().str.replace(" ", "_")

        if "show_id" not in ticket_sales.columns and "showid" in ticket_sales.columns:
            ticket_sales = ticket_sales.rename(columns={"showid": "show_id"})

        # Limpeza numérica
        for col in ["sales_to_date", "atp"]:
            if col in ticket_sales.columns:
                ticket_sales[col] = (ticket_sales[col].astype(str)
                                     .str.replace(r"[^0-9.]", "", regex=True)
                                     .replace("", "0")
                                     .astype(float))
        for col in ["total_sold", "remaining", "sold_%", "capacity"]:
            if col in ticket_sales.columns:
                ticket_sales[col] = pd.to_numeric(ticket_sales[col], errors="coerce")

        # Datas
        if "show_date" in ticket_sales.columns and "report_date" in ticket_sales.columns:
            ticket_sales["show_date"] = pd.to_datetime(ticket_sales["show_date"], errors="coerce")
            ticket_sales["report_date"] = pd.to_datetime(ticket_sales["report_date"], errors="coerce")
            ticket_sales["days_to_show"] = (ticket_sales["show_date"] - ticket_sales["report_date"]).dt.days

        merged = merged.merge(ticket_sales, how="left", on="show_id")
        sales_available = True
    except Exception:
        st.warning("⚠️ Ticket Sales Sheet not accessible. Continuing with Ads data only.")
        sales_available = False

    st.success("✅ Data cleaned and merged!")

    # Show selector
    show_list = merged["show_id"].dropna().unique().tolist()
    selected_show = st.sidebar.selectbox("Select Show", show_list)
    df_show = merged[merged["show_id"] == selected_show]

    # Health of Show KPIs
    st.subheader(f"Health of Show: {selected_show}")
    col1, col2, col3, col4, col5 = st.columns(5)

    if sales_available and not df_show.empty:
        safe_metric(col1, "🎟 Tickets Sold", lambda: int(df_show["total_sold"].dropna().mean()))
        safe_metric(col2, "💰 Ticket Cost", lambda: round(df_show["sales_to_date"].dropna().mean() / df_show["total_sold"].dropna().mean(), 2))
        safe_metric(col3, "📈 ROAS", lambda: round(((df_show["atp"].dropna().mean() * df_show["capacity"].dropna().mean()) / df_show["amount_spent_usd"].dropna().mean()), 2))
        safe_metric(col4, "📊 CPA Daily", lambda: round(df_show["amount_spent_usd"].dropna().mean() / df_show["total_sold"].dropna().mean(), 2))
        safe_metric(col5, "🎯 Daily Target", lambda: round((df_show["capacity"].dropna().mean() - df_show["total_sold"].dropna().mean()) / df_show["days_to_show"].dropna().mean(), 2))
    else:
        col1.metric("🎟 Tickets Sold", "N/A")
        col2.metric("💰 Ticket Cost", "N/A")
        col3.metric("📈 ROAS", "N/A")
        col4.metric("📊 CPA Daily", "N/A")
        col5.metric("🎯 Daily Target", "N/A")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 Funnel Decay", "📱 Devices & Placements", "⏰ Time of Day", "🎭 Sales Summary"]
    )

    with tab1:
        st.subheader("Funnel Decay Over Time")
        if not df_show.empty:
            funnel_trend = df_show.groupby(["reporting_starts", "funnel"], as_index=False).agg({
                "impressions": "sum",
                "link_clicks": "sum",
                "amount_spent_usd": "sum"
            })
            if not funnel_trend.empty:
                fig = px.line(funnel_trend, x="reporting_starts", y="impressions", color="funnel", title="Impressions over Time")
                st.plotly_chart(fig, use_container_width=True)

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

    with tab3:
        st.subheader("Performance by Time of Day")
        if "time_of_day_viewers_time_zone" in time_of_day:
            time_counts = time_of_day["time_of_day_viewers_time_zone"].value_counts().reset_index()
            time_counts.columns = ["time_slot", "count"]
            chart = alt.Chart(time_counts).mark_bar().encode(
                x="time_slot", y="count", tooltip=["time_slot", "count"]
            ).properties(width=800)
            st.altair_chart(chart, use_container_width=True)

    with tab4:
        if sales_available:
            st.subheader("Ticket Sales by Show")
            sales_summary = ticket_sales[["show_id", "capacity", "total_sold", "remaining", "atp", "sales_to_date"]]
            st.dataframe(sales_summary)
            fig_sales = px.bar(sales_summary, x="show_id", y="total_sold", title="Tickets Sold per Show")
            st.plotly_chart(fig_sales, use_container_width=True)
        else:
            st.info("Sales data not available.")

else:
    st.info("⬆️ Please upload the three CSV files in the sidebar to begin.")
