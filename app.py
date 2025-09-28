"""
Unified Streamlit Dashboard
Sales + Ads Integration + Regex Mapping Funnel Analysis
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from public_sheets_connector import PublicSheetsConnector

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Unified Ads Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Regex Mapping Funnel setup
# ---------------------------------------------------------------------------
MAPPING_FILE = Path("campaign_mapping_fixed.csv")
CLASSIFICATION_TEXT_FIELDS = ("ad_set_name","campaign_name","ad_name")

NUMERIC_CANDIDATES = [
    "amount_spent_(usd)",
    "impressions",
    "results",
    "clicks",
    "cpm",
    "ctr_(link_click-through_rate)",
    "cost_per_results"
]

class MappingRule:
    __slots__ = ("compiled","mapping_type","mapping_value")
    def __init__(self, pattern:str, mapping_type:str, mapping_value:str)->None:
        self.compiled = re.compile(pattern, re.IGNORECASE)
        self.mapping_type = mapping_type.lower()
        self.mapping_value = mapping_value.strip()
    def matches(self,text:str)->bool:
        return bool(self.compiled.search(text))

def build_mapping_rules(table:pd.DataFrame)->List[MappingRule]:
    rules=[]
    for _,rule in table.iterrows():
        pattern=str(rule.get("regex_pattern","")).strip()
        mapping_type=str(rule.get("mapping_type",""))
        mapping_value=str(rule.get("mapping_value",""))
        if not pattern: continue
        try: rules.append(MappingRule(pattern,mapping_type,mapping_value))
        except re.error as exc: st.warning(f"⚠️ Invalid regex: {pattern} ({exc})")
    return rules

def normalize_text_parts(row:pd.Series,fields:Iterable[str])->str:
    parts=[str(row.get(f)).lower() for f in fields if pd.notna(row.get(f))]
    return " ".join(parts)

def apply_classification(row:pd.Series,rules:List[MappingRule])->pd.Series:
    text=normalize_text_parts(row,CLASSIFICATION_TEXT_FIELDS)
    show,funnel=None,None
    legacy=[]
    for r in rules:
        if not text or not r.matches(text): continue
        if r.mapping_type=="show" and not show: show=r.mapping_value or "Unknown"
        elif r.mapping_type=="funnel" and not funnel: funnel=r.mapping_value
        elif r.mapping_type=="legacy" and r.mapping_value: legacy.append(r.mapping_value)
    show=show or "Unknown"
    funnel=funnel or "Unclassified"
    legacy_label=", ".join(sorted(set(legacy))) if legacy else None
    return pd.Series({"show":show,"funnel":funnel,"legacy_label":legacy_label})

def sanitize_columns(df:pd.DataFrame)->pd.DataFrame:
    renamed={c:c.strip().lower().replace(" ","_") for c in df.columns}
    df=df.rename(columns=renamed)
    for col in NUMERIC_CANDIDATES:
        if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
    if "reporting_starts" in df.columns:
        df["reporting_starts"]=pd.to_datetime(df["reporting_starts"],errors="coerce")
    return df

def compute_decay(df:pd.DataFrame)->pd.DataFrame:
    if "ad_set_name" not in df.columns or "cost_per_results" not in df.columns:
        return pd.DataFrame()
    results=[]
    grouped=df.sort_values("reporting_starts").groupby("ad_set_name",dropna=True)
    for adset,g in grouped:
        g=g.copy()
        baseline=g["cost_per_results"].replace([np.inf,-np.inf],np.nan).dropna().head(3).median()
        if pd.isna(baseline) or baseline<=0: continue
        g["is_good"]=g["cost_per_results"]<=1.3*baseline
        good_days=0; bad=0
        for ok in g["is_good"]:
            if ok: good_days+=1; bad=0
            else: bad+=1
            if bad>=3: break
        results.append({"ad_set_name":adset,
                        "funnel":g["funnel"].iloc[0] if "funnel" in g else "Unclassified",
                        "show":g["show"].iloc[0] if "show" in g else "Unknown",
                        "good_days_before_drop":good_days})
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.title("📊 Unified Ads Analyzer")

    sheets_connector=PublicSheetsConnector()

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    # Sales data (Google Sheets)
    if st.sidebar.button("Load Sales Data"):
        with st.spinner("Loading sales data..."):
            sales_data=sheets_connector.load_data()
            st.session_state.sales_data=sales_data if sales_data is not None else None

    # Ads data (normalized)
    ads_file=st.sidebar.file_uploader("Upload Ads File",type=["csv","xlsx"])
    if ads_file:
        ads_data=pd.read_csv(ads_file) if ads_file.name.endswith(".csv") else pd.read_excel(ads_file)
        st.session_state.ads_data=sanitize_columns(ads_data)

    # Funnel regex files
    days_file=st.sidebar.file_uploader("Days.csv",type="csv")
    days_time_file=st.sidebar.file_uploader("Days + Time.csv",type="csv")
    days_pd_file=st.sidebar.file_uploader("Days + Placement + Device.csv",type="csv")

    mapping_rules=[]
    if MAPPING_FILE.exists():
        mapping_df=pd.read_csv(MAPPING_FILE).fillna("")
        mapping_rules=build_mapping_rules(mapping_df)

    # Tabs
    tab1,tab2,tab3,tab4,tab5=st.tabs(["🎫 Sales","📈 Ads","🔗 Integration","📊 Funnel (Regex Mapping)","🔍 Raw Data"])

    with tab1:
        sales_df=st.session_state.get("sales_data")
        if sales_df is not None:
            st.dataframe(sales_df,use_container_width=True)
        else:
            st.info("Click 'Load Sales Data' to fetch from Google Sheets")

    with tab2:
        ads_df=st.session_state.get("ads_data")
        if ads_df is not None:
            st.dataframe(ads_df,use_container_width=True)
        else:
            st.info("Upload ads data file")

    with tab3:
        sales_df=st.session_state.get("sales_data")
        ads_df=st.session_state.get("ads_data")
        if sales_df is not None and ads_df is not None:
            st.write("Integration analysis between ads and sales (to be expanded).")
        else:
            st.info("Need both sales and ads data.")

    with tab4:
        uploaded_dfs={}
        def load_csv(upload,name):
            if upload is None: return
            df=pd.read_csv(upload)
            df=sanitize_columns(df)
            if mapping_rules is not None:
                classified=df.apply(lambda r:apply_classification(r,mapping_rules),axis=1)
                for c in classified.columns: df[c]=classified[c]
            uploaded_dfs[name]=df

        load_csv(days_file,"days")
        load_csv(days_time_file,"days_time")
        load_csv(days_pd_file,"days_pd")

        if not uploaded_dfs:
            st.info("Upload Meta Ads export files.")
        else:
            if "days" in uploaded_dfs:
                decay=compute_decay(uploaded_dfs["days"])
                if not decay.empty:
                    st.subheader("⏳ Funnel Decay Analysis")
                    st.dataframe(decay)

    with tab5:
        if "sales_data" in st.session_state:
            st.subheader("Sales Data")
            st.dataframe(st.session_state.sales_data)
        if "ads_data" in st.session_state:
            st.subheader("Ads Data")
            st.dataframe(st.session_state.ads_data)

    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit | Unified Ads Analyzer*")

if __name__=="__main__":
    main()
