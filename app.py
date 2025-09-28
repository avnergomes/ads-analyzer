import streamlit as st
from data_loader import load_ads_data, load_sales_data_from_sheet, clean_ads_dataframe, clean_sales_dataframe
from mapper import map_campaigns, get_unmapped_campaigns
from metrics import summarize_show_metrics, compute_funnel_efficiency
from visualizer import show_kpis, plot_daily_sales, plot_funnel_efficiency

# App Config
st.set_page_config(page_title="Ads Performance Analyzer", layout="wide")
st.title("🎟️ Ads Performance Analyzer for Shows")

# Sidebar info
with st.sidebar:
    st.header("Data Sources")
    st.caption("Using fixed Google Sheet:")
    st.code("docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA")

# Use fixed export URL
sheet_url = "https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/export?format=csv"

# Load Meta Ads data
ads_df = load_ads_data()
try:
    ads_df = clean_ads_dataframe(ads_df)
except Exception as e:
    st.error(f"Error in ads CSV: {e}")
    st.stop()

# Load sales sheet
sales_df = None
try:
    sales_df = load_sales_data_from_sheet(sheet_url)
    sales_df = clean_sales_dataframe(sales_df)
except Exception as e:
    st.error(f"Failed to load or clean sales sheet: {e}")
    st.stop()

# 1. Map campaigns
ads_mapped = map_campaigns(ads_df, sales_df)

st.subheader("✅ Mapped Campaigns")
st.dataframe(ads_mapped[["campaign_name", "mapped_show_id"]].drop_duplicates(), use_container_width=True)

st.subheader("❌ Unmapped Campaigns")
unmapped = get_unmapped_campaigns(ads_mapped)
st.dataframe(unmapped[["campaign_name"]].drop_duplicates(), use_container_width=True)

# 2. Show selector
unique_shows = ads_mapped["mapped_show_id"].dropna().unique()
selected_show = st.selectbox("🎭 Select a Show", options=unique_shows)

if selected_show:
    show_ads = ads_mapped[ads_mapped["mapped_show_id"] == selected_show]
    show_sales = sales_df[sales_df["show_id"] == selected_show].iloc[0]

    # 3. KPIs
    spend_total = show_ads["amount_spent"].sum()
    metrics = summarize_show_metrics(show_sales, spend_total)
    show_kpis(metrics)

    # 4. Funnel efficiency
    funnel = {
        "clicks": show_ads["clicks"].sum(),
        "lpviews": show_ads["lpviews"].sum(),
        "addtocart": show_ads["addtocart"].sum(),
        "conversions": show_ads["conversions"].sum(),
        "tickets_sold": show_sales["total_sold"]
    }
    funnel_metrics = compute_funnel_efficiency(funnel)
    plot_funnel_efficiency(funnel_metrics)

    # 5. Daily sales trend
    plot_daily_sales(sales_df, selected_show)
