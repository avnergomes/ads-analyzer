"""
Utility to fetch and clean Sales data from a public Google Sheet
"""

import io
import numpy as np
import pandas as pd
import requests

class PublicSheetsConnector:
    def __init__(self):
        # Public Google Sheets URL (export as CSV)
        self.sheet_url = (
            "https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/export?format=csv&gid=0"
        )

    def load_data(self) -> pd.DataFrame | None:
        """Fetch sales data from Google Sheets and clean it"""
        try:
            resp = requests.get(self.sheet_url)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            return self._clean_sales_data(df)
        except Exception as e:
            print(f"Error fetching Google Sheet: {e}")
            return None

    def _clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize sales data from Google Sheets"""
        if df is None or df.empty:
            return df

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Remove section headers like "September", "October" etc.
        if "show_id" in df.columns:
            df = df[df["show_id"].notna() & (df["show_id"] != "None")]

        # Drop completely empty rows
        df = df.dropna(how="all")

        # Convert dates
        for col in ["show_date", "report_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numerics
        num_cols = [
            "capacity",
            "venue_holds",
            "wheelchair_&_companions",
            "camera",
            "artist's_hold",
            "kills",
            "yesterday",
            "today's_sold",
            "sales_to_date",
            "total_sold",
            "remaining",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create occupancy_rate if missing
        if "sales_to_date" in df.columns and "capacity" in df.columns:
            df["occupancy_rate"] = np.where(
                df["capacity"] > 0,
                (df["sales_to_date"] / df["capacity"]) * 100,
                None
            )

        return df.reset_index(drop=True)


    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Quick summary of sales data"""
        if df is None or df.empty:
            return {}
        return {
            "avg_occupancy": df["occupancy_rate"].mean() if "occupancy_rate" in df.columns else 0,
            "unique_cities": df["city"].nunique() if "city" in df.columns else 0
        }
