import pandas as pd
import os
import glob
import streamlit as st

@st.cache_data(show_spinner=False)
def load_ads_data(samples_dir="samples"):
    all_files = glob.glob(os.path.join(samples_dir, "*.csv"))
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = os.path.basename(file)
            df_list.append(df)
        except Exception as e:
            st.warning(f"Error reading {file}: {e}")
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_sales_data_from_sheet(sheet_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"Failed to load sheet: {e}")
        return pd.DataFrame()

def clean_ads_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "campaign_name" not in df.columns:
        raise ValueError("Missing 'campaign_name' column in ads CSV")
    return df.dropna(subset=["campaign_name"])

def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "show_id" not in df.columns:
        raise ValueError("Missing 'show_id' column in sales data")
    return df[df["show_id"].notnull()]
