"""
Utility to fetch and clean Sales data from a public Google Sheet
"""

import pandas as pd
import requests
import io

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
        """Standardize and clean sales DataFrame"""
        if df is None or df.empty:
            return df

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Expected final schema
        col_map = {
            "show_date": ["show_date", "date", "data"],
            "city": ["city", "cidade", "local"],
            "capacity": ["capacity", "capacidade"],
            "total_sold": ["total_sold", "vendidos"],
            "sales_to_date": ["sales_to_date", "receita", "revenue"],
            "today_sold": ["today_sold", "vendidos_hoje"],
            "occupancy_rate": ["occupancy_rate", "ocupacao"],
            "avg_ticket_price": ["avg_ticket_price", "ticket_medio", "preco_medio"]
        }

        clean_df = df.copy()
        for std_col, aliases in col_map.items():
            for alias in aliases:
                if alias in clean_df.columns and std_col not in clean_df.columns:
                    clean_df = clean_df.rename(columns={alias: std_col})
                    break

        # Convert dates
        if "show_date" in clean_df.columns:
            clean_df["show_date"] = pd.to_datetime(clean_df["show_date"], errors="coerce")

        # Numeric conversions
        for col in ["capacity","total_sold","sales_to_date","today_sold","occupancy_rate","avg_ticket_price"]:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        return clean_df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Quick summary of sales data"""
        if df is None or df.empty:
            return {}
        return {
            "avg_occupancy": df["occupancy_rate"].mean() if "occupancy_rate" in df.columns else 0,
            "unique_cities": df["city"].nunique() if "city" in df.columns else 0
        }
