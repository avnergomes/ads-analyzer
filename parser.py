"""
Parser
------
Normalizes column names for consistency across different Meta Ads files.
"""

import re

# Robust column mapping
COLUMN_MAP = {
    "impressions": ["impressions", "impr", "views", "visualizações"],
    "clicks": ["clicks", "click", "cliques", "total clicks"],
    "spend": ["spend", "cost", "gasto", "investment"],
    "conversions": ["conversions", "purchases", "sales", "transactions"],
    "date": ["date", "day", "data"]
}

def normalize_columns(df):
    """
    Normalize dataframe columns using COLUMN_MAP.
    Unknown columns remain unchanged.
    """
    new_columns = {}
    for col in df.columns:
        normalized = None
        col_lower = col.strip().lower()
        for key, patterns in COLUMN_MAP.items():
            for pat in patterns:
                if re.search(rf"\b{pat}\b", col_lower):
                    normalized = key
                    break
            if normalized:
                break
        new_columns[col] = normalized if normalized else col_lower

    return df.rename(columns=new_columns)
