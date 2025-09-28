import re
import pandas as pd
from fuzzywuzzy import process

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()

def match_campaign_to_show(campaign_name, sales_df, show_col="show_id", city_col="city"):
    if not isinstance(campaign_name, str):
        return None
    norm_campaign = normalize_text(campaign_name)
    candidates = sales_df[show_col].dropna().unique()
    for candidate in candidates:
        if normalize_text(candidate) in norm_campaign:
            return candidate
    for _, row in sales_df.iterrows():
        city = normalize_text(str(row.get(city_col, "")))
        if city and city in norm_campaign:
            return row[show_col]
    match, score = process.extractOne(campaign_name, sales_df[show_col].dropna().astype(str).tolist())
    return match if score > 85 else None

def map_campaigns(df_ads: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    df_ads = df_ads.copy()
    df_ads["mapped_show_id"] = df_ads["campaign_name"].apply(
        lambda name: match_campaign_to_show(name, df_sales)
    )
    return df_ads

def get_unmapped_campaigns(df_mapped: pd.DataFrame) -> pd.DataFrame:
    return df_mapped[df_mapped["mapped_show_id"].isna()]
