"""
Data Loader
-----------
Handles reading of Google Sheets sales data and Meta Ads uploads.
"""

import pandas as pd

def load_sales_from_sheets(sheet_url: str):
    """
    Load sales data from a public Google Sheets (CSV export).
    Args:
        sheet_url: str - export CSV URL of Google Sheets.
    Returns:
        pd.DataFrame
    """
    try:
        return pd.read_csv(sheet_url)
    except Exception as e:
        print(f"Error loading Google Sheets: {e}")
        return pd.DataFrame()

def load_ads_file(file):
    """
    Load Meta Ads CSV or XLSX uploaded file.
    Args:
        file: Uploaded file (Streamlit object).
    Returns:
        pd.DataFrame or None
    """
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")
        else:
            return None
    except Exception as e:
        print(f"Error loading Ads file: {e}")
        return pd.DataFrame()
