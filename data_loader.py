import os
import pandas as pd
from parser import normalize_show_id, normalize_funnel_label

def load_ads_data(folder_path: str) -> pd.DataFrame:
    \"\"\"Load and merge all campaign CSVs in a folder\"\"\"
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['source_file'] = filename
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    # Normalize field names
    df.columns = [col.strip().lower().replace('\\n', '').replace('  ', ' ') for col in df.columns]

    # Create normalized show ID
    if 'campaign name' in df.columns:
        df['show_id'] = df['campaign name'].apply(normalize_show_id)
    elif 'campaign_name' in df.columns:
        df['show_id'] = df['campaign_name'].apply(normalize_show_id)

    # Funnel stage mapping
    funnel_map = {
        'clicks': ['link clicks', 'clicks_all'],
        'lp_views': ['landing page views', 'landing_page_views_rate_per_link_clicks'],
        'add_to_cart': ['adds to cart', 'adds_to_cart'],
        'conversions': ['results', 'conversions'],
        'spend': ['amount spent (usd)', 'amount_spent_usd'],
    }

    for key, variants in funnel_map.items():
        for var in variants:
            if var in df.columns:
                df[key] = df[var]
                break
        else:
            df[key] = 0

    return df
