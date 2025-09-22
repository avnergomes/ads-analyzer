import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

st.set_page_config(page_title="Meta Ads Funnel Analysis", layout="wide")
st.title("📊 Meta Ads Funnel Analysis Dashboard")

# --------------------------------
# Helper functions
# --------------------------------
def map_funnel(name, result_indicator=""):
    s = str(name).lower()
    ri = str(result_indicator).lower()
    if 'conv-addtocart' in s or 'conv_addtocart' in s or 'purchase' in ri or 'conversion' in ri:
        return "Funnel 3 (Conv-AddToCart)"
    if 'addtocart' in s or 'add_to_cart' in ri:
        return "Funnel 2 (AddToCart)"
    if 'lpview' in s or 'landing_page_view' in s or 'landing page view' in ri:
        return "Funnel 1 (LPView)"
    return "Unclassified"

city_map = {
    'toronto':'CA','calgary':'CA','edmonton':'CA',
    'portland':'US','seattle':'US','sacramento':'US','columbus':'US',
    'pdx':'US','sea':'US','smf':'US','cmh':'US','tr':'CA','edm':'CA'
}
def detect_country(name):
    s = str(name).lower()
    for token,country in city_map.items():
        if token in s:
            return country
    if "ca-" in s: return "CA"
    if "us-" in s: return "US"
    return "Unclassified"

def parse_hour(s):
    try:
        return int(str(s).split(":")[0])
    except:
        return None

def compute_decay(df):
    out = []
    for adset, g in df.groupby("ad_set_name"):
        g = g.sort_values("reporting_starts")
        base = g["cost_per_results"].replace([np.inf,-np.inf],np.nan).dropna().head(3).median()
        if pd.isna(base) or base <= 0: continue
        g["is_good"] = g["cost_per_results"] <= 1.3*base
        good_days, bad_run = 0, 0
        for ok in g["is_good"]:
            if ok:
                good_days += 1
                bad_run = 0
            else:
                bad_run += 1
                if bad_run >= 3: break
        out.append({
            "ad_set_name": adset,
            "funnel": g["funnel"].iloc[0],
            "country": g["country"].iloc[0],
            "good_days_before_drop": good_days
        })
    return pd.DataFrame(out)

# --------------------------------
# File uploaders
# --------------------------------
st.sidebar.header("Upload your Meta CSVs")
days_file = st.sidebar.file_uploader("Days.csv", type="csv")
days_time_file = st.sidebar.file_uploader("Days + Time.csv", type="csv")
days_pd_file = st.sidebar.file_uploader("Days + Placement + Device.csv", type="csv")

if days_file and days_time_file and days_pd_file:
    # Load
    days = pd.read_csv(days_file)
    days_time = pd.read_csv(days_time_file)
    days_pd = pd.read_csv(days_pd_file)

    # Clean column names
    for df in [days, days_time, days_pd]:
        df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_per_") for c in df.columns]
        if "reporting_starts" in df.columns:
            df["reporting_starts"] = pd.to_datetime(df["reporting_starts"], errors="coerce")

    # Funnel & country tagging
    for df in [days, days_time, days_pd]:
        if "result_indicator" not in df.columns:
            df["result_indicator"] = ""
        df["funnel"] = df.apply(lambda r: map_funnel(r.get("ad_set_name",""), r.get("result_indicator","")), axis=1)
        df["country"] = df["ad_set_name"].apply(detect_country)

    # --------------------------------
    # 1. Funnel Decay Analysis
    # --------------------------------
    st.subheader("⏳ Funnel Decay Analysis")
    decay_df = compute_decay(days)
    if not decay_df.empty:
        fig = px.box(decay_df, x="funnel", y="good_days_before_drop",
                     title="Distribution of Good Days Before Drop by Funnel",
                     points="all", color="funnel")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(decay_df.groupby("funnel")["good_days_before_drop"].describe())
    else:
        st.info("Not enough data to compute funnel decay in this window.")

    # --------------------------------
    # 2. Country × Funnel Overview
    # --------------------------------
    st.subheader("🌍 Country × Funnel Overview")
    country_funnel = days.groupby(["country","funnel"]).agg(
        spend=("amount_spent_(usd)","sum"),
        impressions=("impressions","sum"),
        results=("results","sum"),
        avg_cpm=("cpm_(cost_per_1,000_impressions)_(usd)","mean"),
        avg_cpr=("cost_per_results","mean")
    ).reset_index()
    fig = px.scatter(country_funnel, x="spend", y="results", size="impressions",
                     color="funnel", facet_col="country",
                     hover_data=["avg_cpr","avg_cpm"],
                     title="Spend vs Results by Country & Funnel")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------
    # 3. Time-of-Day Performance
    # --------------------------------
    st.subheader("🕒 Time-of-Day Performance")
    if "time_of_day_(viewer's_time_zone)" in days_time.columns:
        days_time["hour"] = days_time["time_of_day_(viewer's_time_zone)"].apply(parse_hour)
        hour_perf = days_time.groupby(["country","funnel","hour"]).apply(
            lambda x: np.average(x["cost_per_results"], weights=x.get("results", None))
            if "results" in x and x["results"].sum() > 0 else x["cost_per_results"].mean()
        ).reset_index(name="avg_cpr")

        country_sel = st.selectbox("Country (Time)", hour_perf["country"].unique())
        funnel_sel = st.selectbox("Funnel (Time)", hour_perf["funnel"].unique())
        filt = hour_perf[(hour_perf["country"]==country_sel)&(hour_perf["funnel"]==funnel_sel)]
        fig_t = px.line(filt, x="hour", y="avg_cpr", markers=True,
                        title=f"Avg Cost per Result by Hour ({country_sel}, {funnel_sel})")
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(filt)
    else:
        st.info("No time-of-day data available.")

    # --------------------------------
    # 4. Placement Performance
    # --------------------------------
    st.subheader("📱 Placement Performance")
    placement_perf = days_pd.groupby(["country","funnel","placement"]).agg(
        avg_cpr=("cost_per_results","mean"),
        results=("results","sum"),
        spend=("amount_spent_(usd)","sum")
    ).reset_index()
    country_sel_p = st.selectbox("Country (Placement)", placement_perf["country"].unique())
    funnel_sel_p = st.selectbox("Funnel (Placement)", placement_perf["funnel"].unique())
    filt_p = placement_perf[(placement_perf["country"]==country_sel_p)&(placement_perf["funnel"]==funnel_sel_p)]
    fig_p = px.bar(filt_p.sort_values("avg_cpr"), x="placement", y="avg_cpr",
                   color="spend", text="results",
                   title=f"Placement Efficiency ({country_sel_p}, {funnel_sel_p})")
    st.plotly_chart(fig_p, use_container_width=True)
    st.dataframe(filt_p)

    # --------------------------------
    # 5. Device Performance
    # --------------------------------
    st.subheader("💻 Device Performance")
    device_perf = days_pd.groupby(["country","funnel","impression_device"]).agg(
        avg_cpr=("cost_per_results","mean"),
        results=("results","sum"),
        spend=("amount_spent_(usd)","sum")
    ).reset_index()
    country_sel_d = st.selectbox("Country (Device)", device_perf["country"].unique())
    funnel_sel_d = st.selectbox("Funnel (Device)", device_perf["funnel"].unique())
    filt_d = device_perf[(device_perf["country"]==country_sel_d)&(device_perf["funnel"]==funnel_sel_d)]
    fig_d = px.bar(filt_d.sort_values("avg_cpr"), x="impression_device", y="avg_cpr",
                   color="spend", text="results",
                   title=f"Device Efficiency ({country_sel_d}, {funnel_sel_d})")
    st.plotly_chart(fig_d, use_container_width=True)
    st.dataframe(filt_d)

    st.success("✅ All analyses computed from raw exports. Explore interactively!")
else:
    st.info("⬆️ Please upload all three raw Meta exports to start.")
