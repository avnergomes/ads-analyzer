import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

st.set_page_config(page_title="Meta Ads Funnel Analysis", layout="wide")
st.title("📊 Meta Ads Funnel + Ticket Sales Dashboard")

# --------------------------
# Helper functions
# --------------------------
def map_funnel(name, result_indicator=""):
    s = str(name).lower()
    ri = str(result_indicator).lower()

    # --- Funnel 3: Conversion AddToCart
    if ("conv" in s and "addtocart" in s) or "conv_addtocart" in s or "conv-addtocart" in s:
        return "Funnel 3 (Conv-AddToCart)"
    if "conversion" in ri or "purchase" in ri:
        return "Funnel 3 (Conv-AddToCart)"

    # --- Funnel 2: AddToCart (but not conv)
    if "addtocart" in s and "conv" not in s:
        return "Funnel 2 (AddToCart)"
    if "add_to_cart" in ri:
        return "Funnel 2 (AddToCart)"

    # --- Funnel 1: LPView
    if any(x in s for x in ["lpview","lp_view"]) or "landing_page_view" in ri:
        return "Funnel 1 (LPView)"

    return "Unclassified"

def detect_legacy(name):
    s = str(name).lower()
    return "Interest/Target" if ("interest" in s or "target" in s) else "Standard"

city_map = {
    'toronto':'Toronto','calgary':'Calgary','edmonton':'Edmonton',
    'portland':'Portland','seattle':'Seattle','sacramento':'Sacramento','columbus':'Columbus',
    'pdx':'Portland','sea':'Seattle','smf':'Sacramento','cmh':'Columbus','tr':'Toronto','edm':'Edmonton'
}
def detect_city(name):
    s = str(name).lower()
    for token, city in city_map.items():
        if token in s:
            return city
    return "Other"

def detect_country(name):
    s = str(name).lower()
    if any(tok in s for tok in ["toronto","calgary","edmonton","tr","edm","ca-"]):
        return "CA"
    if any(tok in s for tok in ["portland","seattle","sacramento","columbus","pdx","sea","smf","cmh","us-"]):
        return "US"
    return "Unclassified"

def parse_hour(s):
    try: return int(str(s).split(":")[0])
    except: return None

def get_metric(df):
    if "cost_per_results" in df.columns and df["cost_per_results"].notna().any():
        return "cost_per_results"
    if "cpm_(cost_per_1,000_impressions)_(usd)" in df.columns:
        return "cpm_(cost_per_1,000_impressions)_(usd)"
    if "amount_spent_(usd)" in df.columns and "results" in df.columns:
        df["calc_cpr"] = df["amount_spent_(usd)"] / df["results"].replace(0,np.nan)
        return "calc_cpr"
    return None

def compute_decay(df, metric_col):
    out = []
    for adset, g in df.groupby("ad_set_name"):
        g = g.sort_values("reporting_starts")
        base = g[metric_col].replace([np.inf,-np.inf],np.nan).dropna().head(3).median()
        if pd.isna(base) or base <= 0: continue
        g["is_good"] = g[metric_col] <= 1.3*base
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
            "legacy": g["legacy"].iloc[0],
            "country": g["country"].iloc[0],
            "good_days_before_drop": good_days
        })
    return pd.DataFrame(out)

# --------------------------
# Upload files
# --------------------------
st.sidebar.header("Upload Meta Exports")
days_file = st.sidebar.file_uploader("Days.csv", type="csv")
days_time_file = st.sidebar.file_uploader("Days + Time.csv", type="csv")
days_pd_file = st.sidebar.file_uploader("Days + Placement + Device.csv", type="csv")
ticket_file = st.sidebar.file_uploader("Ticket Sales CSV (optional)", type="csv")

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

    # Funnel & geography
    for df in [days, days_time, days_pd]:
        if "result_indicator" not in df.columns: df["result_indicator"] = ""
        df["funnel"] = df.apply(lambda r: map_funnel(r.get("ad_set_name",""), r.get("result_indicator","")), axis=1)
        df["legacy"] = df["ad_set_name"].apply(detect_legacy)
        df["city"] = df["ad_set_name"].apply(detect_city)
        df["country"] = df["ad_set_name"].apply(detect_country)

    # Date filter
    min_date, max_date = pd.to_datetime(days["reporting_starts"]).min(), pd.to_datetime(days["reporting_starts"]).max()
    date_range = st.sidebar.date_input("Select analysis range", [min_date, max_date])
    mask = (days["reporting_starts"] >= pd.to_datetime(date_range[0])) & (days["reporting_starts"] <= pd.to_datetime(date_range[1]))
    days = days.loc[mask]
    days_time = days_time.loc[(days_time["reporting_starts"] >= pd.to_datetime(date_range[0])) & (days_time["reporting_starts"] <= pd.to_datetime(date_range[1]))]
    days_pd = days_pd.loc[(days_pd["reporting_starts"] >= pd.to_datetime(date_range[0])) & (days_pd["reporting_starts"] <= pd.to_datetime(date_range[1]))]

    # Choose metric
    metric_col = get_metric(days)

    # --------------------------------
    # 1. Funnel Decay
    # --------------------------------
    st.subheader("⏳ Funnel Decay Analysis")
    if metric_col:
        decay_df = compute_decay(days, metric_col)
        if not decay_df.empty:
            fig = px.box(decay_df, x="funnel", y="good_days_before_drop", color="legacy",
                         title="Good Days Before Performance Drop by Funnel", points="all")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(decay_df.groupby(["funnel","legacy"])["good_days_before_drop"].describe())
        else:
            st.info("Not enough data to compute decay.")
    else:
        st.warning("No valid performance metric found.")

    # --------------------------------
    # 2. Country × Funnel
    # --------------------------------
    st.subheader("🌍 Country × Funnel Overview")
    country_funnel = days.groupby(["country","funnel","legacy"]).agg(
        spend=("amount_spent_(usd)","sum"),
        impressions=("impressions","sum"),
        results=("results","sum"),
        avg_cpr=(metric_col,"mean")
    ).reset_index()
    fig = px.scatter(country_funnel, x="spend", y="results", size="impressions",
                     color="funnel", facet_col="country", symbol="legacy",
                     hover_data=["avg_cpr"],
                     title="Spend vs Results by Country, Funnel, and Legacy flag")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(country_funnel)

    # --------------------------------
    # 3. Time-of-Day
    # --------------------------------
    st.subheader("🕒 Time-of-Day Performance")
    if "time_of_day_(viewer's_time_zone)" in days_time.columns:
        days_time["hour"] = days_time["time_of_day_(viewer's_time_zone)"].apply(parse_hour)
        hour_perf = days_time.groupby(["country","funnel","legacy","hour"]).apply(
            lambda x: np.average(x[metric_col], weights=x["results"]) if "results" in x and x["results"].sum()>0 else x[metric_col].mean()
        ).reset_index(name="avg_cpr")
        country_sel = st.selectbox("Country (Time)", hour_perf["country"].unique())
        funnel_sel = st.selectbox("Funnel (Time)", hour_perf["funnel"].unique())
        filt = hour_perf[(hour_perf["country"]==country_sel)&(hour_perf["funnel"]==funnel_sel)]
        fig_t = px.line(filt, x="hour", y="avg_cpr", color="legacy", markers=True,
                        title=f"Avg CPR by Hour ({country_sel}, {funnel_sel})")
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(filt)

    # --------------------------------
    # 4. Placement
    # --------------------------------
    st.subheader("📱 Placement Performance")
    placement_perf = days_pd.groupby(["country","funnel","legacy","placement"]).agg(
        avg_cpr=(metric_col,"mean"), results=("results","sum"), spend=("amount_spent_(usd)","sum")
    ).reset_index()
    country_sel_p = st.selectbox("Country (Placement)", placement_perf["country"].unique())
    funnel_sel_p = st.selectbox("Funnel (Placement)", placement_perf["funnel"].unique())
    filt_p = placement_perf[(placement_perf["country"]==country_sel_p)&(placement_perf["funnel"]==funnel_sel_p)]
    fig_p = px.bar(filt_p.sort_values("avg_cpr"), x="placement", y="avg_cpr", color="legacy", text="results",
                   title=f"Placement Efficiency ({country_sel_p}, {funnel_sel_p})")
    st.plotly_chart(fig_p, use_container_width=True)
    st.dataframe(filt_p)

    # --------------------------------
    # 5. Devices
    # --------------------------------
    st.subheader("💻 Device Performance")
    device_perf = days_pd.groupby(["country","funnel","legacy","impression_device"]).agg(
        avg_cpr=(metric_col,"mean"), results=("results","sum"), spend=("amount_spent_(usd)","sum")
    ).reset_index()
    country_sel_d = st.selectbox("Country (Device)", device_perf["country"].unique())
    funnel_sel_d = st.selectbox("Funnel (Device)", device_perf["funnel"].unique())
    filt_d = device_perf[(device_perf["country"]==country_sel_d)&(device_perf["funnel"]==funnel_sel_d)]
    fig_d = px.bar(filt_d.sort_values("avg_cpr"), x="impression_device", y="avg_cpr", color="legacy", text="results",
                   title=f"Device Efficiency ({country_sel_d}, {funnel_sel_d})")
    st.plotly_chart(fig_d, use_container_width=True)
    st.dataframe(filt_d)

    # --------------------------------
    # 6. Ticket Sales
    # --------------------------------
    if ticket_file:
        st.subheader("🎟 Ticket Sales Integration")
        tickets = pd.read_csv(ticket_file)
        st.dataframe(tickets)

        city_ads = days.groupby("city").agg(
            spend=("amount_spent_(usd)","sum"), impressions=("impressions","sum"), results=("results","sum")
        ).reset_index()
        merged = pd.merge(tickets, city_ads, how="left", on="city")
        merged["tickets_per_$"] = merged["tickets_sold"] / merged["spend"].replace(0,np.nan)
        merged["tickets_per_1k_impr"] = merged["tickets_sold"] / (merged["impressions"]/1000).replace(0,np.nan)

        st.dataframe(merged)
        fig_ts = px.scatter(merged, x="spend", y="tickets_sold", size="impressions", color="city",
                            hover_data=["tickets_per_$","tickets_per_1k_impr"],
                            title="Ad Spend vs Ticket Sales by City")
        st.plotly_chart(fig_ts, use_container_width=True)
        fig_ts2 = px.bar(merged, x="city", y="tickets_per_$", color="tickets_sold", text="tickets_sold",
                         title="Tickets per $ Spent by City")
        st.plotly_chart(fig_ts2, use_container_width=True)

    st.success("✅ Analysis complete. Adjust filters in sidebar.")
else:
    st.info("⬆️ Upload the three raw Meta exports to get started.")
