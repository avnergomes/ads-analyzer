import streamlit as st
from data_loader import load_ads_data, load_sales_data_from_sheet, clean_ads_dataframe, clean_sales_dataframe
from mapper import map_campaigns, get_unmapped_campaigns
from metrics import summarize_show_metrics, compute_funnel_efficiency
from visualizer import show_kpis, plot_daily_sales, plot_funnel_efficiency

# Streamlit App Config
st.set_page_config(page_title="Ads Performance Analyzer", layout="wide")

# Title
st.title("🎟️ Ads Performance Analyzer for Shows")

# Sidebar: User Input
with st.sidebar:
    st.header("Data Sources")
    sheet_url = st.text_input("Google Sheet URL")
    creds_path = st.text_input("Path to Google API credentials JSON", value="credentials.json")

# Load Meta Ads data
ads_df = load_ads_data()
ads_df = clean_ads_dataframe(ads_df)

# Load Sales Sheet
sales_df = None
if sheet_url and creds_path:
    try:
        sales_df = load_sales_data_from_sheet(sheet_url, creds_path)
        sales_df = clean_sales_dataframe(sales_df)
    except Exception as e:
        st.error(f"Failed to load sales sheet: {e}")

if sales_df is not None:
    # Step 1 - Map Campaigns to Shows
    ads_mapped = map_campaigns(ads_df, sales_df)

    st.subheader("Mapped Campaigns")
    st.dataframe(ads_mapped[["campaign_name", "mapped_show_id"]].drop_duplicates())

    st.subheader("Unmapped Campaigns")
    unmapped = get_unmapped_campaigns(ads_mapped)
    st.dataframe(unmapped[["campaign_name"]].drop_duplicates())

    # Step 2 - Show Selector
    unique_shows = ads_mapped["mapped_show_id"].dropna().unique()
    selected_show = st.selectbox("Select a Show", options=unique_shows)

    if selected_show:
        show_ads = ads_mapped[ads_mapped["mapped_show_id"] == selected_show]
        show_sales = sales_df[sales_df["show_id"] == selected_show].iloc[0]

        # Step 3 - Summary Metrics
        spend_total = show_ads["amount_spent"].sum()
        metrics = summarize_show_metrics(show_sales, spend_total)

        show_kpis(metrics)

        # Step 4 - Funnel Metrics
        funnel = {
            "clicks": show_ads["clicks"].sum(),
            "lpviews": show_ads["lpviews"].sum(),
            "addtocart": show_ads["addtocart"].sum(),
            "conversions": show_ads["conversions"].sum(),
            "tickets_sold": show_sales["total_sold"]
        }
        funnel_metrics = compute_funnel_efficiency(funnel)
        plot_funnel_efficiency(funnel_metrics)

        # Step 5 - Daily Sales Plot
        plot_daily_sales(sales_df, selected_show)
