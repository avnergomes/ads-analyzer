"""
Metrics
-------
Computes key performance indicators (KPIs) from sales + ads data.
"""

import pandas as pd

def compute_kpis(sales_df: pd.DataFrame, ads_df: pd.DataFrame) -> dict:
    """
    Compute KPIs combining sales and ads data.
    """
    impressions = ads_df.get("impressions", pd.Series()).sum()
    clicks = ads_df.get("clicks", pd.Series()).sum()
    spend = ads_df.get("spend", pd.Series()).sum()
    conversions = ads_df.get("conversions", pd.Series()).sum()
    total_sales = sales_df.iloc[:, -1].sum() if not sales_df.empty else 0

    ctr = (clicks / impressions * 100) if impressions else 0
    cpc = (spend / clicks) if clicks else 0
    cpa = (spend / conversions) if conversions else 0
    roas = (total_sales / spend) if spend else 0

    return {
        "Impressions": impressions,
        "Clicks": clicks,
        "Spend": spend,
        "Conversions": conversions,
        "Sales (Sheets)": total_sales,
        "CTR (%)": round(ctr, 2),
        "CPC": round(cpc, 2),
        "CPA": round(cpa, 2),
        "ROAS": round(roas, 2)
    }
