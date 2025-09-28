from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:  # pragma: no cover - optional dependency during local testing
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:  # pragma: no cover - handled gracefully when missing
    gspread = None
    ServiceAccountCredentials = None

ADS_COLUMN_ALIASES = {
    "campaign_name": ["campaign name", "campaign", "ad campaign"],
    "amount_spent": [
        "amount spent",
        "amount spent (usd)",
        "amount spent usd",
        "spend",
        "amount_spent_usd",
    ],
    "clicks": ["link clicks", "clicks", "clicks_all"],
    "lp_views": ["landing page views", "landing page view", "lp views", "landing_page_views"],
    "add_to_cart": ["adds to cart", "add to cart", "adds_to_cart"],
    "conversions": ["results", "conversions", "purchases"],
}

_SHEET_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def _rename_ads_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, variations in ADS_COLUMN_ALIASES.items():
        for variation in variations:
            if variation in df.columns:
                rename_map[variation] = canonical
                break
    return df.rename(columns=rename_map)


def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        df[column] = series.fillna(0.0)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


@st.cache_data(show_spinner=False)
def load_ads_data_from_folder(folder: str | Path = "samples") -> pd.DataFrame:
    """Load all CSV files from *folder* and concatenate them."""
    folder_path = Path(folder)
    if not folder_path.exists():
        st.warning(f"Folder '{folder_path}' was not found.")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(folder_path.glob("*.csv")):
        try:
            frames.append(_read_csv(csv_path))
        except Exception as exc:
            st.warning(f"Error while reading '{csv_path.name}': {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ads_data_from_files(files: Iterable[UploadedFile]) -> pd.DataFrame:
    """Load ads data from the uploaded files provided by Streamlit."""
    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            frames.append(df)
        except Exception as exc:
            st.warning(f"Error while processing '{file.name}': {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_sales_data_from_sheet(sheet_url: str, credentials_json_path: str) -> pd.DataFrame:
    """Fetch the first worksheet from a Google Sheet using service credentials."""
    if gspread is None or ServiceAccountCredentials is None:
        raise RuntimeError(
            "Google Sheets support requires 'gspread' and 'oauth2client' packages to be installed."
        )

    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_json_path, _SHEET_SCOPE)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_url(sheet_url)
    worksheet = spreadsheet.get_worksheet(0)
    records = worksheet.get_all_records()
    return pd.DataFrame(records)


def clean_ads_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = _normalize_columns(df)
    df = _rename_ads_columns(df)
    df = df.loc[:, ~df.columns.duplicated()]

    required_columns = ["campaign_name", "amount_spent", "clicks", "lp_views", "add_to_cart", "conversions"]
    for column in required_columns:
        if column not in df.columns:
            df[column] = 0

    df = df.dropna(subset=["campaign_name"])
    df = _ensure_numeric(df, [col for col in required_columns if col != "campaign_name"])
    return df


def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = _normalize_columns(df)
    if "show_id" in df.columns:
        df = df[df["show_id"].notnull()]
    return df
