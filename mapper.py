import re
import pandas as pd
from fuzzywuzzy import process

# Regras de nomes equivalentes para estágios de funil
FUNNEL_ALIASES = {
    "F1": ["F1", "Fun1", "LPViews", "LPViews_F1", "LPViews_Fun1"],
    "F2": ["F2", "Fun2", "AddToCart", "AddToCart_F2"],
    "F3": ["F3", "Fun3", "CONV_AddtoCart", "CONV_F3"],
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def extract_possible_ids(campaign_name):
    """
    Extrai possíveis candidatos a showId com base em padrões como:
    - WDC_0927_S2
    - DC0927
    - Ottawa #2
    """
    campaign = campaign_name.lower()
    ids = set()

    # Captura ID tipo ABC_1234_S2
    matches = re.findall(r"[a-z]{2,4}_?\d{3,4}(_s\d)?", campaign)
    ids.update(matches)

    # Captura sufixos tipo "#2", "2nd", "second", "s2"
    if "#2" in campaign or "2nd" in campaign or "s2" in campaign:
        ids.add("s2")
    if "#3" in campaign or "3rd" in campaign or "s3" in campaign:
        ids.add("s3")

    return list(ids)


def match_campaign_to_show(campaign_name, sales_df, show_col="show_id", city_col="city"):
    """
    Tenta casar uma campanha a um show, usando vários níveis:
    1. Match direto por show_id
    2. Match por nome da cidade
    3. Fuzzy match
    """
    if not isinstance(campaign_name, str):
        return None

    norm_campaign = normalize_text(campaign_name)
    candidates = sales_df[show_col].dropna().unique()

    # Tentativa 1: match exato por show_id
    for candidate in candidates:
        if normalize_text(candidate) in norm_campaign:
            return candidate

    # Tentativa 2: match por cidade
    for _, row in sales_df.iterrows():
        city = normalize_text(str(row.get(city_col, "")))
        if city and city in norm_campaign:
            return row[show_col]

    # Tentativa 3: fuzzy match por show_id
    show_ids = sales_df[show_col].dropna().astype(str).tolist()
    match, score = process.extractOne(campaign_name, show_ids)
    if score > 85:
        return match

    return None


def map_campaigns(df_ads: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma coluna 'mapped_show_id' ao dataframe de campanhas,
    com o show correspondente se encontrado.
    """
    df_ads = df_ads.copy()
    df_ads["mapped_show_id"] = df_ads["campaign_name"].apply(
        lambda name: match_campaign_to_show(name, df_sales)
    )
    return df_ads


def get_unmapped_campaigns(df_mapped: pd.DataFrame) -> pd.DataFrame:
    return df_mapped[df_mapped["mapped_show_id"].isna()]
