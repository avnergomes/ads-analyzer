import pandas as pd
import os
import glob
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials


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
            st.warning(f"Erro ao ler {file}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_sales_data_from_sheet(sheet_url: str, credentials_json_path: str) -> pd.DataFrame:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_json_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = worksheet.get_all_records()

    df = pd.DataFrame(data)
    return df


def clean_ads_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["campaign_name"], errors="ignore")
    return df


def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df["show_id"].notnull()]
    return df
