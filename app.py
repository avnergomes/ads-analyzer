import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt

st.set_page_config(page_title="Meta Ads Funnel Analysis", layout="wide")

st.title("📊 Meta Ads Funnel & Show Analysis")

# --- Upload Section ---
st.sidebar.header("Upload Meta Ads Files")
days_file = st.sidebar.file_uploader("Upload Days.csv", type="csv")
placement_device_file = st.sidebar.file_uploader("Upload Days + Placement + Device.csv", type="csv")
time_file = st.sidebar.file_uploader("Upload Days + Time.csv", type="csv")

if days_file and placement_device_file and time_file:

    # --- Load files ---
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

    # --- Standardize ID ---
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

    # --- Funnel classification ---
    def classify_funnel(name: str) -> str:
        if pd.isna(name):
            return "Unclassified"
        name = str(name).lower()
        if "lpview" in name:
            return "F1_LPView"
        elif "addtocart" in name and "conv" not in name:
            return "F2_AddtoCart"
        elif "conv" in name and "addtocart" in name:
            return "F3_Conversion"
        else:
            return "Unclassified"

    merged["funnel"] = merged["ad_set_same"].apply(classify_funnel)

    st.success("✅ Data cleaned and merged successfully!")

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 Funnel Decay", "📱 Devices & Placements", "⏰ Time of Day", "🎭 Shows Summary"]
    )

    # --- Funnel Decay Curves ---
    with tab1:
        st.subheader("Funnel Decay Over Time")
        funnel_trend = merged.groupby(["reporting_starts", "funnel"], as_index=False).agg({
            "impressions": "sum",
            "link_clicks": "sum",
            "amount_spent_usd": "sum"
        })

        fig = px.line(
            funnel_trend,
            x="reporting_starts", y="impressions",
            color="funnel",
            title="Impressions over Time by Funnel"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Devices & Placements ---
    with tab2:
        st.subheader("Device & Placement Breakdown")
        device_counts = placement_device.explode("impression_device")["impression_device"].value_counts().reset_index()
        device_counts.columns = ["device", "count"]
        st.bar_chart(device_counts.set_index("device"))

        placement_counts = placement_device.explode("placement")["placement"].value_counts().reset_index()
        placement_counts.columns = ["placement", "count"]
        st.bar_chart(placement_counts.set_index("placement"))

    # --- Time of Day ---
    with tab3:
        st.subheader("Performance by Time of Day")
        time_counts = time_of_day["time_of_day_viewers_time_zone"].value_counts().reset_index()
        time_counts.columns = ["time_slot", "count"]
        chart = alt.Chart(time_counts).mark_bar().encode(
            x="time_slot",
            y="count",
            tooltip=["time_slot", "count"]
        ).properties(width=800)
        st.altair_chart(chart, use_container_width=True)

    # --- Show Summary ---
    with tab4:
        st.subheader("Ticket Sales by Show")
        show_sales = {
            "Toronto": 14194,
            "Portland": 887,
            "Seattle": 1739,
            "Edmonton": 2470,
            "Sacramento": 1727,
            "Columbus": 475,
            "SunRose": 100,
            "UpStairs": 160
        }
        sales_df = pd.DataFrame(list(show_sales.items()), columns=["Show", "Tickets Sold"])
        st.dataframe(sales_df)
        st.bar_chart(sales_df.set_index("Show"))

else:
    st.info("⬆️ Please upload the three CSV files in the sidebar to begin.")
