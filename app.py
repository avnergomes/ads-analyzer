import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Meta Ads Funnel Analysis", layout="wide")
st.title("📊 Meta Ads Funnel + Ticket Sales Dashboard")


# --------------------------
# Helper functions
# --------------------------
def map_funnel(name: str, result_indicator: str = "") -> str:
    """Map ad set names and result indicators to a funnel stage."""
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
    if any(x in s for x in ["lpview", "lp_view"]) or "landing_page_view" in ri:
        return "Funnel 1 (LPView)"

    return "Unclassified"


def detect_legacy(name: str) -> str:
    s = str(name).lower()
    return "Interest/Target" if ("interest" in s or "target" in s) else "Standard"


city_map = {
    r"toronto|tr_": "Toronto",
    r"calgary": "Calgary",
    r"edmonton|edm": "Edmonton",
    r"portland|pdx": "Portland",
    r"seattle|sea": "Seattle",
    r"sacramento|smf": "Sacramento",
    r"columbus|cmh": "Columbus",
}


def detect_city(name: str) -> str:
    s = str(name).lower()
    for pattern, city in city_map.items():
        if re.search(pattern, s):
            return city
    return "Other"


def detect_country(name: str) -> str:
    s = str(name).lower()

    # Explicit prefixes like IG_CA, FB_US, etc.
    if "_ca" in s or "-ca" in s or " ca-" in s:
        return "CA"
    if "_us" in s or "-us" in s or " us-" in s:
        return "US"

    # Fallback: look for city tokens
    if re.search(r"toronto|calgary|edmonton|tr_|edm", s):
        return "CA"
    if re.search(r"portland|seattle|sacramento|columbus|pdx|sea|smf|cmh", s):
        return "US"

    return "Unclassified"


def parse_hour(value: str) -> Optional[int]:
    try:
        s = str(value)
        if ":" in s:
            return int(s.split(":", 1)[0])
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("/", "_per_", regex=False)
        .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        regex = re.compile(pattern)
        for col in df.columns:
            if regex.search(col):
                return col
    return None


def ensure_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ad_set_name" not in df.columns:
        alt = find_column(df, [r"adset"])
        if alt:
            df["ad_set_name"] = df[alt]
        else:
            df["ad_set_name"] = "Unknown"
    if "campaign_name" not in df.columns:
        alt = find_column(df, [r"campaign"])
        if alt:
            df["campaign_name"] = df[alt]
    if "ad_name" not in df.columns:
        alt = find_column(df, [r"(^|_)ad_name($|_)"])
        if alt:
            df["ad_name"] = df[alt]
    if "result_indicator" not in df.columns:
        df["result_indicator"] = ""
    return df


COLUMN_PATTERNS: Dict[str, List[str]] = {
    "spend": [r"amount[_]?spent", r"total_spend", r"(^|_)spend($|_)"],
    "results": [r"(^|_)results($|_)", r"(^|_)conversions($|_)", r"(^|_)purchases($|_)"],
    "impressions": [r"(^|_)impressions($|_)"],
    "clicks": [r"link[_]?clicks", r"(^|_)clicks($|_)"],
    "reach": [r"(^|_)reach($|_)"],
    "frequency": [r"(^|_)frequency($|_)"],
    "purchases": [r"(^|_)purchases($|_)"],
    "value": [r"(^|_)purchase_value($|_)", r"(^|_)total_value($|_)", r"(^|_)revenue($|_)"],
    "cpm": [r"(^|_)cpm", r"cost_per_1_000"],
    "time": [r"time_of_day", r"hour"],
    "placement": [r"placement"],
    "device": [r"impression_device", r"device_type", r"device"],
}


def detect_core_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {key: find_column(df, patterns) for key, patterns in COLUMN_PATTERNS.items()}


def classify_metric(column_name: Optional[str]) -> str:
    if not column_name:
        return "unknown"
    col = column_name.lower()
    if "calc_cpm" in col or "cpm" in col or "cost_per_1_000" in col:
        return "cpm"
    if any(term in col for term in ["calc_cpr", "cost_per", "cpa", "cpp"]):
        return "cost_per_result"
    if any(term in col for term in ["ctr", "rate"]):
        return "rate"
    return "value"


def ensure_metric_column(
    df: pd.DataFrame, metric_candidate: Optional[str], core_cols: Dict[str, Optional[str]]
) -> Optional[str]:
    if metric_candidate and metric_candidate in df.columns:
        return metric_candidate

    metric_type = classify_metric(metric_candidate)
    spend_col = core_cols.get("spend")
    results_col = core_cols.get("results")
    impressions_col = core_cols.get("impressions")

    if metric_type == "cpm" and spend_col and impressions_col:
        calc_name = "calc_cpm"
        if calc_name not in df.columns:
            impressions = df[impressions_col].replace(0, np.nan)
            df[calc_name] = df[spend_col] / (impressions / 1000)
        return calc_name

    if spend_col and results_col:
        calc_name = "calc_cpr"
        if calc_name not in df.columns:
            df[calc_name] = df[spend_col] / df[results_col].replace(0, np.nan)
        return calc_name

    if metric_candidate and metric_candidate in df.columns:
        return metric_candidate

    return metric_candidate if metric_candidate in df.columns else None


def get_metric(df: pd.DataFrame, core_cols: Dict[str, Optional[str]]) -> Optional[str]:
    metric_patterns = [
        r"cost_per_result[s]?",
        r"cost_per_add[_]?to[_]?cart",
        r"cost_per_purchase",
        r"cost_per_conversion",
        r"(^|_)cpa($|_)",
        r"(^|_)cpp($|_)",
    ]
    for pattern in metric_patterns:
        col = find_column(df, [pattern])
        if col and df[col].notna().any():
            return col

    cpm_col = core_cols.get("cpm")
    if cpm_col and cpm_col in df.columns and df[cpm_col].notna().any():
        return cpm_col

    spend_col = core_cols.get("spend")
    results_col = core_cols.get("results")
    if spend_col and results_col:
        calc_name = "calc_cpr"
        if calc_name not in df.columns:
            df[calc_name] = df[spend_col] / df[results_col].replace(0, np.nan)
        return calc_name

    return None


def weighted_average(series: pd.Series, weights: Optional[pd.Series]) -> float:
    if weights is None:
        return float(series.mean()) if not series.empty else float("nan")
    mask = (~series.isna()) & (~weights.isna())
    if not mask.any():
        return float("nan")
    series = series[mask]
    weights = weights[mask]
    total_weight = weights.sum()
    if total_weight == 0:
        return float("nan")
    return float(np.average(series, weights=weights))


def format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"${value:,.2f}" if abs(value) < 1000 else f"${value:,.0f}"


def format_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def format_rate(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:,.2f}%"


def format_metric(value: Optional[float], metric_type: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    if metric_type in {"cost_per_result", "cpm", "value"}:
        return format_currency(value)
    if metric_type == "rate":
        return format_rate(value)
    return format_number(value)


def labelize_metric(column_name: Optional[str]) -> str:
    if not column_name:
        return "Metric"
    if column_name == "calc_cpr":
        return "Cost per Result (calc)"
    if column_name == "calc_cpm":
        return "CPM (calc)"
    return column_name.replace("_", " ").title()


def apply_segment_filters(
    df: pd.DataFrame, funnels: List[str], countries: List[str], legacies: List[str]
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    if funnels:
        mask &= df["funnel"].isin(funnels)
    if countries:
        mask &= df["country"].isin(countries)
    if legacies:
        mask &= df["legacy"].isin(legacies)
    return df.loc[mask].copy()


def compute_metric_value(
    df: pd.DataFrame,
    metric_col: Optional[str],
    metric_type: str,
    core_cols: Dict[str, Optional[str]],
) -> Optional[float]:
    if df.empty or not metric_col:
        return None
    spend_col = core_cols.get("spend")
    results_col = core_cols.get("results")
    impressions_col = core_cols.get("impressions")

    if metric_type == "cost_per_result" and spend_col and results_col:
        total_results = df[results_col].replace(0, np.nan).sum()
        if total_results > 0:
            return df[spend_col].sum() / total_results
        return None

    if metric_type == "cpm" and spend_col and impressions_col:
        total_impressions = df[impressions_col].replace(0, np.nan).sum()
        if total_impressions > 0:
            return df[spend_col].sum() / (total_impressions / 1000)
        return None

    if metric_col in df.columns:
        return float(df[metric_col].mean())
    return None


def compute_weight_column(metric_type: str, core_cols: Dict[str, Optional[str]]) -> Optional[str]:
    if metric_type == "cost_per_result":
        return core_cols.get("results")
    if metric_type == "cpm":
        return core_cols.get("impressions")
    if metric_type == "rate":
        return core_cols.get("impressions")
    return None


# --------------------------
# Upload files
# --------------------------
st.sidebar.header("Upload Meta Exports")
days_file = st.sidebar.file_uploader("Days.csv", type="csv")
days_time_file = st.sidebar.file_uploader("Days + Time.csv", type="csv")
days_pd_file = st.sidebar.file_uploader("Days + Placement + Device.csv", type="csv")
ticket_file = st.sidebar.file_uploader("Ticket Sales CSV (optional)", type="csv")


@st.cache_data(show_spinner=False)
def load_meta_file(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    df = normalize_columns(df)
    df = ensure_identifier_columns(df)
    for col in ["reporting_starts", "reporting_ends"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


if days_file and days_time_file and days_pd_file:
    days = load_meta_file(days_file)
    days_time = load_meta_file(days_time_file)
    days_pd = load_meta_file(days_pd_file)

    dataframes = [days, days_time, days_pd]
    for df in dataframes:
        df["funnel"] = df.apply(
            lambda r: map_funnel(r.get("ad_set_name", ""), r.get("result_indicator", "")), axis=1
        )
        df["legacy"] = df["ad_set_name"].apply(detect_legacy)
        df["city"] = df["ad_set_name"].apply(detect_city)
        df["country"] = df["ad_set_name"].apply(detect_country)

    core_days = detect_core_columns(days)
    core_days_time = detect_core_columns(days_time)
    core_days_pd = detect_core_columns(days_pd)

    metric_col = get_metric(days, core_days)
    metric_type = classify_metric(metric_col)
    metric_time_col = ensure_metric_column(days_time, metric_col, core_days_time)
    metric_pd_col = ensure_metric_column(days_pd, metric_col, core_days_pd)
    metric_label = labelize_metric(metric_col or metric_time_col or metric_pd_col)

    date_min = pd.to_datetime(days.get("reporting_starts")).min()
    date_max = pd.to_datetime(days.get("reporting_starts")).max()
    if pd.isna(date_min) or pd.isna(date_max):
        st.error("No valid reporting dates detected in the Days.csv file.")
        st.stop()

    default_range = (date_min.date(), date_max.date())
    date_range = st.sidebar.date_input("Select analysis range", default_range)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date = pd.to_datetime(date_range)
        end_date = start_date

    mask_days = (days["reporting_starts"] >= start_date) & (days["reporting_starts"] <= end_date)
    days = days.loc[mask_days].copy()

    if "reporting_starts" in days_time.columns:
        mask_time = (days_time["reporting_starts"] >= start_date) & (
            days_time["reporting_starts"] <= end_date
        )
        days_time = days_time.loc[mask_time].copy()
    else:
        days_time = days_time.copy()

    if "reporting_starts" in days_pd.columns:
        mask_pd = (days_pd["reporting_starts"] >= start_date) & (
            days_pd["reporting_starts"] <= end_date
        )
        days_pd = days_pd.loc[mask_pd].copy()
    else:
        days_pd = days_pd.copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation")
    funnel_options = sorted(days["funnel"].dropna().unique())
    country_options = sorted(days["country"].dropna().unique())
    legacy_options = sorted(days["legacy"].dropna().unique())

    selected_funnels = st.sidebar.multiselect("Funnel", funnel_options, default=funnel_options)
    selected_countries = st.sidebar.multiselect("Country", country_options, default=country_options)
    selected_legacies = st.sidebar.multiselect("Legacy", legacy_options, default=legacy_options)

    days = apply_segment_filters(days, selected_funnels, selected_countries, selected_legacies)
    days_time = apply_segment_filters(days_time, selected_funnels, selected_countries, selected_legacies)
    days_pd = apply_segment_filters(days_pd, selected_funnels, selected_countries, selected_legacies)

    if days.empty:
        st.warning("No rows remain after applying the selected filters. Adjust the filters to see results.")
        st.stop()

    spend_col = core_days.get("spend")
    results_col = core_days.get("results")
    impressions_col = core_days.get("impressions")
    clicks_col = core_days.get("clicks")
    reach_col = core_days.get("reach")
    frequency_col = core_days.get("frequency")

    total_spend = days[spend_col].sum() if spend_col else np.nan
    total_results = days[results_col].sum() if results_col else np.nan
    total_impressions = days[impressions_col].sum() if impressions_col else np.nan
    total_clicks = days[clicks_col].sum() if clicks_col else np.nan
    total_reach = days[reach_col].sum() if reach_col else np.nan

    avg_metric = compute_metric_value(days, metric_col, metric_type, core_days)
    result_rate = None
    if (
        results_col
        and impressions_col
        and not pd.isna(total_impressions)
        and total_impressions > 0
    ):
        result_rate = total_results / total_impressions

    summary_cols = st.columns(4)
    summary_cols[0].metric("Total Spend", format_currency(total_spend))
    summary_cols[1].metric("Total Results", format_number(total_results))
    summary_cols[2].metric(f"Avg {metric_label}", format_metric(avg_metric, metric_type))
    if result_rate is not None:
        summary_cols[3].metric("Result Rate", format_rate(result_rate))
    elif impressions_col:
        summary_cols[3].metric("Impressions", format_number(total_impressions))
    else:
        summary_cols[3].metric("Reach", format_number(total_reach))

    extra_cols = st.columns(3)
    if impressions_col and not pd.isna(total_impressions):
        extra_cols[0].metric("Impressions", format_number(total_impressions))
    if (
        clicks_col
        and impressions_col
        and not pd.isna(total_impressions)
        and total_impressions > 0
    ):
        extra_cols[1].metric("CTR", format_rate(total_clicks / total_impressions))
    if frequency_col:
        extra_cols[2].metric("Avg Frequency", format_number(days[frequency_col].mean()))

    # --------------------------------
    # Daily performance
    # --------------------------------
    st.subheader("📈 Daily Performance Trend")
    daily_df = days.dropna(subset=["reporting_starts"]).copy()
    daily_df["reporting_date"] = daily_df["reporting_starts"].dt.date
    daily_agg = {}
    if spend_col:
        daily_agg["spend"] = (spend_col, "sum")
    if results_col:
        daily_agg["results"] = (results_col, "sum")
    if impressions_col:
        daily_agg["impressions"] = (impressions_col, "sum")
    daily_summary = pd.DataFrame()
    if daily_agg:
        daily_summary = daily_df.groupby("reporting_date").agg(**daily_agg).reset_index()

    if not daily_summary.empty:
        if metric_type == "cost_per_result" and {"spend", "results"}.issubset(daily_summary.columns):
            daily_summary["metric"] = daily_summary["spend"] / daily_summary["results"].replace(0, np.nan)
        elif metric_type == "cpm" and {"spend", "impressions"}.issubset(daily_summary.columns):
            daily_summary["metric"] = daily_summary["spend"] / (daily_summary["impressions"] / 1000)
        elif metric_col and metric_col in days.columns:
            metric_daily = (
                days.groupby("reporting_starts")[metric_col]
                .mean()
                .reset_index()
            )
            metric_daily["reporting_date"] = metric_daily["reporting_starts"].dt.date
            daily_summary = daily_summary.merge(metric_daily[["reporting_date", metric_col]], on="reporting_date", how="left")
            daily_summary["metric"] = daily_summary[metric_col]

        if "spend" in daily_summary.columns or "results" in daily_summary.columns:
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            if "spend" in daily_summary.columns:
                fig_trend.add_trace(
                    go.Bar(
                        x=daily_summary["reporting_date"],
                        y=daily_summary["spend"],
                        name="Spend",
                        marker_color="#636EFA",
                    ),
                    secondary_y=False,
                )
            if "results" in daily_summary.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=daily_summary["reporting_date"],
                        y=daily_summary["results"],
                        name="Results",
                        mode="lines+markers",
                        line=dict(color="#EF553B"),
                    ),
                    secondary_y=True,
                )
            fig_trend.update_layout(
                title="Spend vs Results", barmode="group", hovermode="x unified"
            )
            fig_trend.update_xaxes(title="Date")
            fig_trend.update_yaxes(title="Spend", secondary_y=False)
            fig_trend.update_yaxes(title="Results", secondary_y=True)
            st.plotly_chart(fig_trend, use_container_width=True)

        if "metric" in daily_summary.columns and daily_summary["metric"].notna().any():
            fig_metric = px.line(
                daily_summary,
                x="reporting_date",
                y="metric",
                markers=True,
                title=f"Daily {metric_label}",
            )
            fig_metric.update_layout(hovermode="x unified")
            st.plotly_chart(fig_metric, use_container_width=True)
    else:
        st.info("Not enough date data to render the daily performance trend.")

    # --------------------------------
    # Funnel overview
    # --------------------------------
    st.subheader("🛒 Funnel Overview")
    funnel_agg = {}
    if spend_col:
        funnel_agg["spend"] = (spend_col, "sum")
    if results_col:
        funnel_agg["results"] = (results_col, "sum")
    if impressions_col:
        funnel_agg["impressions"] = (impressions_col, "sum")
    if clicks_col:
        funnel_agg["clicks"] = (clicks_col, "sum")

    funnel_summary = pd.DataFrame()
    if funnel_agg:
        funnel_summary = (
            days.groupby(["funnel", "legacy", "country"]).agg(**funnel_agg).reset_index()
        )

    if not funnel_summary.empty:
        if metric_type == "cost_per_result" and {"spend", "results"}.issubset(funnel_summary.columns):
            funnel_summary[metric_label] = funnel_summary["spend"] / funnel_summary["results"].replace(0, np.nan)
        elif metric_type == "cpm" and {"spend", "impressions"}.issubset(funnel_summary.columns):
            funnel_summary[metric_label] = funnel_summary["spend"] / (funnel_summary["impressions"] / 1000)
        elif metric_col and metric_col in days.columns:
            metric_group = (
                days.groupby(["funnel", "legacy", "country"])[metric_col]
                .mean()
                .reset_index()
            )
            funnel_summary = funnel_summary.merge(metric_group, on=["funnel", "legacy", "country"], how="left")
            funnel_summary[metric_label] = funnel_summary[metric_col]

        if clicks_col and impressions_col and "impressions" in funnel_summary.columns:
            funnel_summary["CTR"] = funnel_summary["clicks"] / funnel_summary["impressions"].replace(0, np.nan)

        plot_x = "spend" if "spend" in funnel_summary.columns else None
        plot_y = None
        if "results" in funnel_summary.columns:
            plot_y = "results"
        elif metric_label in funnel_summary.columns:
            plot_y = metric_label

        if plot_x and plot_y:
            scatter_args = {
                "data_frame": funnel_summary,
                "x": plot_x,
                "y": plot_y,
                "color": "funnel",
                "symbol": "legacy",
                "facet_col": "country",
                "title": "Spend vs Results by Country, Funnel, and Legacy",
            }
            if "impressions" in funnel_summary.columns:
                scatter_args["size"] = "impressions"
            if metric_label in funnel_summary.columns:
                scatter_args["hover_data"] = [metric_label]
            fig_funnel = px.scatter(**scatter_args)
            st.plotly_chart(fig_funnel, use_container_width=True)
        st.dataframe(funnel_summary.sort_values("spend", ascending=False) if "spend" in funnel_summary.columns else funnel_summary)
    else:
        st.info("No funnel data available after filtering.")

    # --------------------------------
    # Country overview table
    # --------------------------------
    st.subheader("🌍 Country Snapshot")
    country_agg = {}
    if spend_col:
        country_agg["spend"] = (spend_col, "sum")
    if results_col:
        country_agg["results"] = (results_col, "sum")
    if impressions_col:
        country_agg["impressions"] = (impressions_col, "sum")
    country_summary = pd.DataFrame()
    if country_agg:
        country_summary = days.groupby("country").agg(**country_agg).reset_index()
    if not country_summary.empty:
        if metric_type == "cost_per_result" and {"spend", "results"}.issubset(country_summary.columns):
            country_summary[metric_label] = country_summary["spend"] / country_summary["results"].replace(0, np.nan)
        elif metric_type == "cpm" and {"spend", "impressions"}.issubset(country_summary.columns):
            country_summary[metric_label] = country_summary["spend"] / (country_summary["impressions"] / 1000)
        st.dataframe(country_summary)
    else:
        st.info("No country level data available.")

    # --------------------------------
    # Campaign and Ad Set leaderboards
    # --------------------------------
    st.subheader("🏆 Leaderboards")
    leaderboard_agg = {}
    if spend_col:
        leaderboard_agg["spend"] = (spend_col, "sum")
    if results_col:
        leaderboard_agg["results"] = (results_col, "sum")
    if impressions_col:
        leaderboard_agg["impressions"] = (impressions_col, "sum")

    adset_summary = pd.DataFrame()
    if leaderboard_agg:
        adset_summary = (
            days.groupby(["ad_set_name", "funnel", "legacy", "country"]).agg(**leaderboard_agg).reset_index()
        )
    if not adset_summary.empty:
        if metric_type == "cost_per_result" and {"spend", "results"}.issubset(adset_summary.columns):
            adset_summary[metric_label] = adset_summary["spend"] / adset_summary["results"].replace(0, np.nan)
        elif metric_type == "cpm" and {"spend", "impressions"}.issubset(adset_summary.columns):
            adset_summary[metric_label] = adset_summary["spend"] / (adset_summary["impressions"] / 1000)
        elif metric_col and metric_col in days.columns:
            metric_group = (
                days.groupby(["ad_set_name", "funnel", "legacy", "country"])[metric_col]
                .mean()
                .reset_index()
            )
            adset_summary = adset_summary.merge(metric_group, on=["ad_set_name", "funnel", "legacy", "country"], how="left")
            adset_summary[metric_label] = adset_summary[metric_col]

        best_results = adset_summary.sort_values("results", ascending=False) if "results" in adset_summary.columns else adset_summary
        best_efficiency = adset_summary.sort_values(metric_label, ascending=(metric_type != "rate")) if metric_label in adset_summary.columns else adset_summary

        tab1, tab2 = st.tabs(["Top Results", "Best Efficiency"])
        with tab1:
            st.dataframe(best_results.head(15))
        with tab2:
            st.dataframe(best_efficiency.head(15))
    else:
        st.info("No ad set level data available.")

    # --------------------------------
    # Time-of-Day Analysis
    # --------------------------------
    st.subheader("🕒 Time-of-Day Performance")
    time_col = find_column(days_time, COLUMN_PATTERNS["time"])
    results_time_col = core_days_time.get("results")
    weight_col_time = compute_weight_column(metric_type, core_days_time)
    if time_col and metric_time_col and metric_time_col in days_time.columns:
        days_time["hour"] = days_time[time_col].apply(parse_hour)
        hour_perf = (
            days_time.dropna(subset=["hour"])
            .groupby(["country", "funnel", "legacy", "hour"])
            .apply(lambda g: weighted_average(g[metric_time_col], g[weight_col_time]) if weight_col_time and weight_col_time in g.columns else g[metric_time_col].mean())
            .reset_index(name="avg_metric")
        )
        if results_time_col and results_time_col in days_time.columns:
            volume = (
                days_time.groupby(["country", "funnel", "legacy", "hour"])[results_time_col]
                .sum()
                .reset_index(name="volume")
            )
            hour_perf = hour_perf.merge(volume, on=["country", "funnel", "legacy", "hour"], how="left")
        if not hour_perf.empty:
            country_sel = st.selectbox("Country (Time)", sorted(hour_perf["country"].unique()))
            funnel_sel = st.selectbox("Funnel (Time)", sorted(hour_perf["funnel"].unique()))
            filt = hour_perf[(hour_perf["country"] == country_sel) & (hour_perf["funnel"] == funnel_sel)]
            fig_time = px.line(
                filt,
                x="hour",
                y="avg_metric",
                color="legacy",
                markers=True,
                hover_data=["volume"] if "volume" in filt.columns else None,
                title=f"{metric_label} by Hour ({country_sel}, {funnel_sel})",
            )
            fig_time.update_xaxes(dtick=1)
            st.plotly_chart(fig_time, use_container_width=True)
            st.dataframe(filt)
        else:
            st.info("Not enough hourly data after filtering.")
    else:
        st.info("Time-of-day information not available in the uploaded report.")

    # --------------------------------
    # Placement Performance
    # --------------------------------
    st.subheader("📱 Placement Performance")
    placement_col = find_column(days_pd, COLUMN_PATTERNS["placement"])
    results_pd_col = core_days_pd.get("results")
    weight_col_pd = compute_weight_column(metric_type, core_days_pd)
    if placement_col and metric_pd_col and metric_pd_col in days_pd.columns:
        group_cols = ["country", "funnel", "legacy", placement_col]
        base_agg = {metric_pd_col: "mean"}
        if results_pd_col:
            base_agg[results_pd_col] = "sum"
        placement_perf = (
            days_pd.groupby(group_cols)
            .agg(base_agg)
            .reset_index()
        )
        if weight_col_pd and weight_col_pd in days_pd.columns:
            weighted = (
                days_pd.groupby(group_cols)
                .apply(lambda g: weighted_average(g[metric_pd_col], g[weight_col_pd]))
                .reset_index(name="avg_metric")
            )
            placement_perf = placement_perf.drop(columns=[metric_pd_col]).merge(
                weighted, on=group_cols, how="left"
            )
        else:
            placement_perf.rename(columns={metric_pd_col: "avg_metric"}, inplace=True)
        if results_pd_col and results_pd_col in placement_perf.columns:
            placement_perf.rename(columns={results_pd_col: "results"}, inplace=True)

        country_sel_p = st.selectbox("Country (Placement)", sorted(placement_perf["country"].unique()))
        funnel_sel_p = st.selectbox("Funnel (Placement)", sorted(placement_perf["funnel"].unique()))
        filt_p = placement_perf[(placement_perf["country"] == country_sel_p) & (placement_perf["funnel"] == funnel_sel_p)]
        fig_p = px.bar(
            filt_p.sort_values("avg_metric", ascending=(metric_type == "rate")),
            x=placement_col,
            y="avg_metric",
            color="legacy",
            text="results" if "results" in filt_p.columns else None,
            title=f"{metric_label} by Placement ({country_sel_p}, {funnel_sel_p})",
        )
        st.plotly_chart(fig_p, use_container_width=True)
        st.dataframe(filt_p)
    else:
        st.info("Placement information not available in the uploaded report.")

    # --------------------------------
    # Device Performance
    # --------------------------------
    st.subheader("💻 Device Performance")
    device_col = find_column(days_pd, COLUMN_PATTERNS["device"])
    if device_col and metric_pd_col and metric_pd_col in days_pd.columns:
        device_group_cols = ["country", "funnel", "legacy", device_col]
        base_agg = {metric_pd_col: "mean"}
        if results_pd_col:
            base_agg[results_pd_col] = "sum"
        device_perf = (
            days_pd.groupby(device_group_cols)
            .agg(base_agg)
            .reset_index()
        )
        if weight_col_pd and weight_col_pd in days_pd.columns:
            weighted = (
                days_pd.groupby(device_group_cols)
                .apply(lambda g: weighted_average(g[metric_pd_col], g[weight_col_pd]))
                .reset_index(name="avg_metric")
            )
            device_perf = device_perf.drop(columns=[metric_pd_col]).merge(
                weighted, on=device_group_cols, how="left"
            )
        else:
            device_perf.rename(columns={metric_pd_col: "avg_metric"}, inplace=True)
        if results_pd_col and results_pd_col in device_perf.columns:
            device_perf.rename(columns={results_pd_col: "results"}, inplace=True)

        country_sel_d = st.selectbox("Country (Device)", sorted(device_perf["country"].unique()))
        funnel_sel_d = st.selectbox("Funnel (Device)", sorted(device_perf["funnel"].unique()))
        filt_d = device_perf[(device_perf["country"] == country_sel_d) & (device_perf["funnel"] == funnel_sel_d)]
        fig_d = px.bar(
            filt_d.sort_values("avg_metric", ascending=(metric_type == "rate")),
            x=device_col,
            y="avg_metric",
            color="legacy",
            text="results" if "results" in filt_d.columns else None,
            title=f"{metric_label} by Device ({country_sel_d}, {funnel_sel_d})",
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.dataframe(filt_d)
    else:
        st.info("Device information not available in the uploaded report.")

    # --------------------------------
    # City performance snapshot
    # --------------------------------
    st.subheader("🏙 City Performance Snapshot")
    city_agg = {}
    if spend_col:
        city_agg["spend"] = (spend_col, "sum")
    if results_col:
        city_agg["results"] = (results_col, "sum")
    if impressions_col:
        city_agg["impressions"] = (impressions_col, "sum")
    city_summary = pd.DataFrame()
    if city_agg:
        city_summary = days.groupby("city").agg(**city_agg).reset_index()
    if not city_summary.empty:
        if metric_type == "cost_per_result" and {"spend", "results"}.issubset(city_summary.columns):
            city_summary[metric_label] = city_summary["spend"] / city_summary["results"].replace(0, np.nan)
        elif metric_type == "cpm" and {"spend", "impressions"}.issubset(city_summary.columns):
            city_summary[metric_label] = city_summary["spend"] / (city_summary["impressions"] / 1000)
        plot_x = "spend" if "spend" in city_summary.columns else None
        plot_y = None
        if "results" in city_summary.columns:
            plot_y = "results"
        elif metric_label in city_summary.columns:
            plot_y = metric_label

        if plot_x and plot_y:
            scatter_args = {
                "data_frame": city_summary,
                "x": plot_x,
                "y": plot_y,
                "color": "city",
                "title": "City Level Performance",
            }
            if "impressions" in city_summary.columns:
                scatter_args["size"] = "impressions"
            if metric_label in city_summary.columns:
                scatter_args["hover_data"] = [metric_label]
            fig_city = px.scatter(**scatter_args)
            st.plotly_chart(fig_city, use_container_width=True)
        st.dataframe(city_summary)
    else:
        st.info("City level view not available.")

    # --------------------------------
    # Ticket Sales Integration
    # --------------------------------
    if ticket_file:
        st.subheader("🎟 Ticket Sales Integration")
        tickets = load_meta_file(ticket_file)
        if "city" not in tickets.columns:
            st.warning("Ticket file must contain a 'city' column to merge with ad performance.")
        else:
            ticket_merge_agg = {}
            if spend_col:
                ticket_merge_agg["spend"] = (spend_col, "sum")
            if impressions_col:
                ticket_merge_agg["impressions"] = (impressions_col, "sum")
            if results_col:
                ticket_merge_agg["results"] = (results_col, "sum")
            if ticket_merge_agg:
                city_ads = days.groupby("city").agg(**ticket_merge_agg).reset_index()
            else:
                city_ads = days[["city"]].drop_duplicates()
            merged = pd.merge(tickets, city_ads, how="left", on="city")
            if spend_col and "spend" in merged.columns:
                merged["tickets_per_$"] = merged["tickets_sold"] / merged["spend"].replace(0, np.nan)
            if impressions_col and "impressions" in merged.columns:
                merged["tickets_per_1k_impr"] = merged["tickets_sold"] / (
                    merged["impressions"].replace(0, np.nan) / 1000
                )
            st.dataframe(merged)
            if "spend" in merged.columns:
                fig_ts = px.scatter(
                    merged,
                    x="spend",
                    y="tickets_sold",
                    size="impressions" if "impressions" in merged.columns else None,
                    color="city",
                    hover_data=["tickets_per_$", "tickets_per_1k_impr"] if "tickets_per_$" in merged.columns else None,
                    title="Ad Spend vs Ticket Sales by City",
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            if "tickets_per_$" in merged.columns:
                fig_ts2 = px.bar(
                    merged,
                    x="city",
                    y="tickets_per_$",
                    color="tickets_sold",
                    text="tickets_sold",
                    title="Tickets per $ Spent by City",
                )
                st.plotly_chart(fig_ts2, use_container_width=True)

    st.success("✅ Analysis complete. Adjust filters in sidebar as needed.")
else:
    st.info("⬆️ Upload the three raw Meta exports to get started.")
