import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import re
import requests
from io import StringIO

st.set_page_config(page_title="Meta Ads Funnel & Show Analysis", layout="wide")

st.title("📊 Meta Ads Funnel & Show Analysis")

# --- Upload Section ---
st.sidebar.header("Upload Meta Ads Files")
days_file = st.sidebar.file_uploader("Upload Days.csv", type="csv")
placement_device_file = st.sidebar.file_uploader("Upload Days + Placement + Device.csv", type="csv")
time_file = st.sidebar.file_uploader("Upload Days + Time.csv", type="csv")

# --- Google Sheet (Ticket Sales) ---
sheet_url = "https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/export?format=csv"

# --- Helpers ---
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

def clean_number(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    s = re.sub(r"[^0-9,.-]", "", s)
    if s.count(".") > 1:
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

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

def safe_metric(col, label, func):
    try:
        val = func()
        if pd.isna(val):
            val = "N/A"
    except Exception:
        val = "N/A"
    col.metric(label, val)

# --- Load mapping tables ---
@st.cache_data
def load_reference_tables():
    shows_ref = pd.read_excel("Shows.xlsx")
    cities_ref = pd.read_excel("City_Country_Region_Table.xlsx")

    shows_ref["show_norm"] = shows_ref["Show"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    cities_ref["city_norm"] = cities_ref["City"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    return shows_ref, cities_ref

shows_ref, cities_ref = load_reference_tables()

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
    merged["show_norm"] = merged.get("ad_set_name", "").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    # --- Ticket Sales ---
    try:
        r = requests.get(sheet_url)
        r.raise_for_status()
        ticket_sales = pd.read_csv(StringIO(r.text))

        ticket_sales = ticket_sales.dropna(how="all")
        ticket_sales.columns = (ticket_sales.columns.str.strip()
                                .str.lower()
                                .str.replace(" ", "_")
                                .str.replace(r"[^a-z0-9_]", "", regex=True))

        if "show_id" not in ticket_sales.columns and "showid" in ticket_sales.columns:
            ticket_sales = ticket_sales.rename(columns={"showid": "show_id"})

        ticket_sales = ticket_sales[ticket_sales["show_id"].notna()]
        ticket_sales = ticket_sales[~ticket_sales["show_id"]
            .str.contains("report|total|endrow|summary", case=False, na=False)]

        ticket_sales["show_norm"] = ticket_sales["show_id"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

        num_cols = ["sales_to_date", "atp", "total_sold", "remaining", "capacity"]
        for col in num_cols:
            if col in ticket_sales.columns:
                ticket_sales[col] = ticket_sales[col].apply(clean_number)
        existing_num_cols = [col for col in num_cols if col in ticket_sales.columns]
        ticket_sales[existing_num_cols] = ticket_sales[existing_num_cols].fillna(0)

        for dcol in ["show_date", "report_date"]:
            if dcol in ticket_sales.columns:
                ticket_sales[dcol] = pd.to_datetime(ticket_sales[dcol], errors="coerce")

        if "show_date" in ticket_sales.columns and "report_date" in ticket_sales.columns:
            ticket_sales["days_to_show"] = (ticket_sales["show_date"] - ticket_sales["report_date"]).dt.days

        ticket_sales = ticket_sales.reset_index(drop=True)
        sales_available = True

        # Debug expander
        with st.expander("🔎 Debug Ticket Sales Data"):
            st.dataframe(ticket_sales.head(20))
            st.write("Columns detected:", ticket_sales.columns.tolist())

    except Exception as e:
        st.warning(f"⚠️ Ticket Sales Sheet not accessible. Using Ads data only. Error: {e}")
        sales_available = False

    # --- Merge with mapping tables ---
    merged["show_norm"] = merged["show_norm"].fillna("")

    merged = merged.merge(
        shows_ref[["ShowID", "show_norm", "Capacity"]],
        on="show_norm", how="left"
    )

    merged = merged.merge(
        cities_ref[["City_Code", "city_norm"]],
        left_on="show_norm", right_on="city_norm", how="left"
    )

    if sales_available:
        merged = merged.merge(ticket_sales, how="left", on="show_norm")

    st.success("✅ Data cleaned, mapped, and merged!")

    show_list = merged["show_norm"].dropna().unique().tolist()
    selected_show = st.sidebar.selectbox("Select Show", show_list)
    df_show = merged[merged["show_norm"] == selected_show]

    # --- KPIs ---
    st.subheader(f"Health of Show: {selected_show}")
    col1, col2, col3, col4, col5 = st.columns(5)

    if sales_available and not df_show.empty:
        safe_metric(col1, "🎟 Tickets Sold", lambda: int(df_show["total_sold"].dropna().mean()))
        safe_metric(col2, "💰 Ticket Cost", lambda: round(df_show["sales_to_date"].dropna().mean() / df_show["total_sold"].dropna().mean(), 2))
        safe_metric(col3, "📈 ROAS", lambda: round(((df_show["atp"].dropna().mean() * df_show["capacity"].dropna().mean()) / df_show["amount_spent_usd"].dropna().mean()), 2))
        safe_metric(col4, "📊 CPA Daily", lambda: round(df_show["amount_spent_usd"].dropna().mean() / df_show["total_sold"].dropna().mean(), 2))
        safe_metric(col5, "🎯 Daily Target", lambda: round((df_show["capacity"].dropna().mean() - df_show["total_sold"].dropna().mean()) / df_show["days_to_show"].dropna().mean(), 2))
    else:
        for col, label in zip([col1, col2, col3, col4, col5],
                              ["🎟 Tickets Sold", "💰 Ticket Cost", "📈 ROAS", "📊 CPA Daily", "🎯 Daily Target"]):
            col.metric(label, "N/A")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Funnel Decay", "📱 Devices & Placements", "⏰ Time of Day", "🎭 Sales Summary"])

    with tab1:
        st.subheader("Funnel Decay Over Time")
        if not df_show.empty:
            funnel_trend = df_show.groupby(["reporting_starts", "funnel"], as_index=False).agg({
                "impressions": "sum",
                "link_clicks": "sum",
                "amount_spent_usd": "sum"
            })
            if not funnel_trend.empty:
                fig = px.line(funnel_trend, x="reporting_starts", y="impressions", color="funnel",
                              title="Impressions over Time", text="impressions")
                fig.update_traces(mode="lines+markers+text", textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Device & Placement Breakdown")
        if "impression_device" in placement_device:
            device_counts = placement_device["impression_device"].value_counts().reset_index()
            device_counts.columns = ["device", "count"]
            fig_dev = px.bar(device_counts, x="device", y="count", text="count")
            fig_dev.update_traces(textposition="outside")
            st.plotly_chart(fig_dev, use_container_width=True)
        if "placement" in placement_device:
            placement_counts = placement_device["placement"].value_counts().reset_index()
            placement_counts.columns = ["placement", "count"]
            fig_place = px.bar(placement_counts, x="placement", y="count", text="count")
            fig_place.update_traces(textposition="outside")
            st.plotly_chart(fig_place, use_container_width=True)

    with tab3:
        st.subheader("Performance by Time of Day")
        if "time_of_day_viewers_time_zone" in time_of_day:
            time_counts = time_of_day["time_of_day_viewers_time_zone"].value_counts().reset_index()
            time_counts.columns = ["time_slot", "count"]
            chart = alt.Chart(time_counts).mark_bar().encode(
                x="time_slot", y="count", tooltip=["time_slot", "count"]
            ).properties(width=800)
            text = chart.mark_text(dy=-10).encode(text="count")
            st.altair_chart(chart + text, use_container_width=True)

    with tab4:
        if sales_available:
            st.subheader("Ticket Sales by Show")
            sales_summary = ticket_sales[["show_norm", "capacity", "total_sold", "remaining", "atp", "sales_to_date"]]
            st.dataframe(sales_summary)
            fig_sales = px.bar(sales_summary, x="show_norm", y="total_sold", text="total_sold",
                               title="Tickets Sold per Show")
            fig_sales.update_traces(textposition="outside")
            st.plotly_chart(fig_sales, use_container_width=True)
        else:
            st.info("Sales data not available.")

else:
    st.info("⬆️ Please upload the three CSV files in the sidebar to begin.")
