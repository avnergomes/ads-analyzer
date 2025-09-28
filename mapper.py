"""Campaign to show mapping helpers."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fuzzywuzzy import process

from parser import contains_any, extract_city_tokens, normalize_show_id

_MANUAL_MAPPING_FILE = "campaign_mapping_fixed.csv"


@lru_cache(maxsize=1)
def _load_manual_mapping(path: str | Path = _MANUAL_MAPPING_FILE) -> tuple[dict[str, str], list[tuple[str, str]]]:
    direct: dict[str, str] = {}
    regex_rules: list[tuple[str, str]] = []

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        return direct, regex_rules

    df = pd.read_csv(csv_path)
    lower_cols = {col.lower(): col for col in df.columns}

    if {"regex_pattern", "mapping_value"}.issubset(lower_cols):
        pattern_col = lower_cols["regex_pattern"]
        value_col = lower_cols["mapping_value"]
        for _, row in df.iterrows():
            pattern = str(row.get(pattern_col, "")).strip()
            value = str(row.get(value_col, "")).strip()
            if pattern and value:
                regex_rules.append((pattern, value))
    else:
        campaign_col = next((col for col in df.columns if col.lower().startswith("campaign")), None)
        show_col = next((col for col in df.columns if "show" in col.lower()), None)
        if campaign_col and show_col:
            for _, row in df.iterrows():
                campaign = str(row[campaign_col]).strip()
                show_id = str(row[show_col]).strip()
                if campaign and show_id:
                    direct[campaign.lower()] = show_id

    return direct, regex_rules


def match_campaign_to_show(
    campaign_name: str,
    sales_df: pd.DataFrame,
    *,
    manual_mapping_path: str | Path = _MANUAL_MAPPING_FILE,
    score_threshold: int = 85,
) -> Optional[str]:
    """Return the most likely show identifier for *campaign_name*.

    The strategy combines manual overrides, ID normalization, city token
    detection and fuzzy matching to cover the majority of naming patterns.
    """
    if not isinstance(campaign_name, str) or campaign_name.strip() == "":
        return None

    direct_mapping, regex_rules = _load_manual_mapping(manual_mapping_path)
    manual_hit = direct_mapping.get(campaign_name.lower())
    if manual_hit:
        return manual_hit

    for pattern, show_id in regex_rules:
        try:
            if re.search(pattern, campaign_name, flags=re.IGNORECASE):
                return show_id
        except re.error:
            continue

    normalized_campaign = normalize_show_id(campaign_name)
    if normalized_campaign:
        for raw_show in sales_df.get("show_id", pd.Series(dtype=str)).dropna().astype(str):
            if normalize_show_id(raw_show) == normalized_campaign:
                return raw_show

    campaign_tokens = extract_city_tokens(campaign_name)
    if not sales_df.empty:
        for _, row in sales_df.iterrows():
            show_id = str(row.get("show_id", ""))
            normalized_show = normalize_show_id(show_id)
            if normalized_campaign and normalized_show and normalized_show in normalized_campaign:
                return show_id

            city_candidates = set()
            for col in ("city", "market", "location", "region"):
                city_candidates |= extract_city_tokens(row.get(col))

            if campaign_tokens and campaign_tokens & city_candidates:
                return show_id or None

    show_candidates = sales_df.get("show_id")
    if show_candidates is not None and not show_candidates.empty:
        show_list = show_candidates.dropna().astype(str).tolist()
        if show_list:
            match = process.extractOne(campaign_name, show_list)
            if match and match[1] >= score_threshold:
                return match[0]

    city_candidates = sales_df.get("city")
    if city_candidates is not None and not city_candidates.empty:
        city_list = city_candidates.dropna().astype(str).tolist()
        if city_list:
            match = process.extractOne(campaign_name, city_list)
            if match and match[1] >= 90:
                city_match = match[0]
                show_row = sales_df[city_candidates == city_match]
                if not show_row.empty:
                    return str(show_row.iloc[0].get("show_id", "")) or None

    if contains_any(campaign_name, ("awareness", "prospecting")):
        return None

    return None


def map_campaigns(
    df_ads: pd.DataFrame,
    df_sales: pd.DataFrame,
    *,
    manual_mapping_path: str | Path = _MANUAL_MAPPING_FILE,
) -> pd.DataFrame:
    """Append a ``mapped_show_id`` column to the Ads dataframe using manual overrides when provided."""
    df = df_ads.copy()
    df["mapped_show_id"] = df["campaign_name"].apply(
        lambda name: match_campaign_to_show(
            name,
            df_sales,
            manual_mapping_path=manual_mapping_path,
        )
    )
    return df


def get_unmapped_campaigns(df_mapped: pd.DataFrame) -> pd.DataFrame:
    """Return campaigns that could not be matched to a show."""
    return df_mapped[df_mapped["mapped_show_id"].isna() | (df_mapped["mapped_show_id"] == "")]
