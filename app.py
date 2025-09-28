import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, timedelta
import requests
import io
from typing import Dict, List, Set, Tuple


# ====================================
# TEXT NORMALIZATION & MATCH UTILITIES
# ====================================

MONTH_ABBR = ["", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def normalize_upper(value) -> str:
    """Normalize value to uppercase string preserving internal spaces."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    return re.sub(r"\s+", " ", text.strip().upper())


def normalize_compact(value) -> str:
    """Uppercase alphanumeric string with all separators removed."""
    return re.sub(r"[^A-Z0-9]", "", normalize_upper(value))


def sanitize_column_names(columns: List[str]) -> List[str]:
    """Standardize column names to snake_case while handling duplicates."""
    seen: Dict[str, int] = {}
    normalized: List[str] = []
    for original in columns:
        if original is None:
            original = ""
        base = normalize_upper(original)
        base = base.lower()
        base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        if not base:
            base = "column"
        count = seen.get(base, 0)
        if count:
            normalized.append(f"{base}_{count+1}")
        else:
            normalized.append(base)
        seen[base] = count + 1
    return normalized


def extract_time_tokens_from_string(text: str) -> Set[str]:
    """Extract time tokens like 4PM, 19:30 from arbitrary string."""
    tokens: Set[str] = set()
    if not text:
        return tokens
    text_upper = normalize_upper(text)
    for match in re.finditer(r"(\d{1,2})(?::?(\d{2}))?\s*(AM|PM)", text_upper):
        hour = int(match.group(1))
        minute = match.group(2)
        ampm = match.group(3)
        hour_12 = hour if 1 <= hour <= 12 else hour % 12 or 12
        tokens.add(f"{hour_12}{ampm}")
        tokens.add(f"{hour_12} {ampm}")
        if minute:
            tokens.add(f"{hour_12}:{minute}{ampm}")
            tokens.add(f"{hour_12}:{minute} {ampm}")
        tokens.add(f"{hour_12:02d}{ampm}")
    for match in re.finditer(r"\b(\d{1,2})[:h](\d{2})\b", text_upper):
        tokens.add(f"{match.group(1)}:{match.group(2)}")
        tokens.add(f"{match.group(1)}{match.group(2)}")
    return tokens


def extract_sequence_tokens(text: str) -> Set[str]:
    """Extract sequence identifiers (#2, S3) from strings."""
    tokens: Set[str] = set()
    if not text:
        return tokens
    text_upper = normalize_upper(text)
    for match in re.finditer(r"[#_\- ]S?(\d+)", text_upper):
        seq = match.group(1)
        tokens.add(f"S{seq}")
        tokens.add(f"#{seq}")
        tokens.add(seq)
    return tokens


def generate_date_tokens(show_date: pd.Timestamp) -> Tuple[Set[str], Set[str]]:
    """Generate multiple textual representations for a show date."""
    tokens: Set[str] = set()
    compact_tokens: Set[str] = set()
    if pd.isna(show_date):
        return tokens, compact_tokens

    month = show_date.month
    day = show_date.day
    year = show_date.year
    month_abbr = MONTH_ABBR[month]

    candidates = {
        f"{month_abbr}{day}",
        f"{month_abbr}{day:02d}",
        f"{month_abbr} {day}",
        f"{month_abbr} {day:02d}",
        f"{day}{month_abbr}",
        f"{day:02d}{month_abbr}",
        f"{day} {month_abbr}",
        f"{day:02d} {month_abbr}",
        f"{month:02d}/{day:02d}",
        f"{month}/{day}",
        f"{month:02d}-{day:02d}",
        f"{month}-{day}",
        f"{month_abbr}{year}",
        f"{month_abbr}{str(year)[-2:]}",
        f"{month_abbr} {year}",
        f"{month_abbr} {str(year)[-2:]}",
        show_date.strftime("%Y-%m-%d"),
        show_date.strftime("%m%d"),
        show_date.strftime("%m/%d"),
        show_date.strftime("%d"),
    }

    for candidate in candidates:
        normalized = normalize_upper(candidate)
        if normalized:
            tokens.add(normalized)
            compact_tokens.add(normalize_compact(normalized))

    return tokens, compact_tokens


def create_show_identifier_from_name(name: str, show_date: pd.Timestamp | None = None) -> str:
    """Fallback show identifier derived from show name and optional date."""
    base = normalize_compact(name)
    if not base:
        base = "SHOW"
    if show_date and not pd.isna(show_date):
        base = f"{base}_{show_date.strftime('%m%d')}"
    return base


def build_show_catalog(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Create a catalog of shows with tokens for mapping Meta Ads data."""
    catalog_rows: List[Dict] = []
    seen_ids: Set[str] = set()

    for _, row in sales_df.iterrows():
        show_id = row.get('Show ID') or row.get('ShowID')
        show_name = row.get('Show Name') or row.get('Show') or show_id
        show_date = row.get('Show Date')
        country = row.get('Country') or row.get('Market')

        if pd.isna(show_id) or not str(show_id).strip():
            show_id = create_show_identifier_from_name(show_name, show_date)

        if pd.isna(show_date):
            show_date = None

        show_id = str(show_id).strip()
        if show_id in seen_ids:
            continue
        seen_ids.add(show_id)

        show_name_str = str(show_name).strip() if show_name is not None else show_id

        city_token_source = show_name_str
        city_token_source = re.sub(r"^\d+\s*[\.#-]?\s*", "", city_token_source)
        city_token_source = city_token_source.split('#')[0]
        city_token_source = city_token_source.split('(')[0]
        city_token = normalize_compact(city_token_source)

        date_tokens, date_tokens_compact = generate_date_tokens(show_date) if show_date else (set(), set())

        time_tokens = set()
        if isinstance(show_date, pd.Timestamp) and not pd.isna(show_date):
            hour = show_date.hour
            minute = show_date.minute
            if not pd.isna(hour):
                ampm = 'AM' if hour < 12 else 'PM'
                hour_12 = hour % 12 or 12
                time_tokens.add(f"{hour_12}{ampm}")
                time_tokens.add(f"{hour_12}:{minute:02d}{ampm}")
                time_tokens.add(f"{hour_12:02d}{ampm}")
        time_tokens |= extract_time_tokens_from_string(show_name_str)

        sequence_tokens = extract_sequence_tokens(show_name_str)
        sequence_tokens |= extract_sequence_tokens(show_id)

        show_record = {
            'show_id': show_id,
            'show_name': show_name_str,
            'show_date': show_date,
            'country': country if pd.notna(country) else None,
            'city_token': normalize_upper(city_token_source),
            'city_token_compact': city_token,
            'show_id_tokens': {normalize_upper(show_id), normalize_compact(show_id)},
            'date_tokens': date_tokens,
            'date_tokens_compact': date_tokens_compact,
            'time_tokens': {normalize_upper(t) for t in time_tokens},
            'time_tokens_compact': {normalize_compact(t) for t in time_tokens},
            'sequence_tokens': {normalize_upper(t) for t in sequence_tokens},
            'sequence_tokens_compact': {normalize_compact(t) for t in sequence_tokens},
        }

        catalog_rows.append(show_record)

    return pd.DataFrame(catalog_rows)


def token_in_text(token: str, text_upper: str, text_compact: str) -> bool:
    """Check if token exists in either spaced or compact text forms."""
    if not token:
        return False
    normalized_token = normalize_upper(token)
    compact_token = normalize_compact(token)
    return (normalized_token and normalized_token in text_upper) or (
        compact_token and compact_token in text_compact
    )


def score_show_match(text_upper: str, text_compact: str, show_row: Dict) -> Tuple[int, List[str]]:
    """Score how well a meta row matches a show entry."""
    score = 0
    reasons: List[str] = []

    # Direct show ID tokens
    for token in show_row['show_id_tokens']:
        if token and token in {text_upper, text_compact}:
            score += 80
            reasons.append("Exact show ID match")
            break
        if token and token in text_upper:
            score += 60
            reasons.append("Matched show ID token")
            break
        if token and token in text_compact:
            score += 50
            reasons.append("Matched compact show ID token")
            break

    # City tokens
    if show_row['city_token'] and token_in_text(show_row['city_token'], text_upper, text_compact):
        score += 30
        reasons.append("Matched city name")

    # Date tokens
    date_hits = [token for token in show_row['date_tokens'] if token_in_text(token, text_upper, text_compact)]
    if date_hits:
        score += 20 + 5 * (len(date_hits) - 1)
        reasons.append(f"Matched date tokens: {', '.join(sorted(set(date_hits)))[:60]}")

    # Time tokens
    time_hits = [token for token in show_row['time_tokens'] if token_in_text(token, text_upper, text_compact)]
    if time_hits:
        score += 12 + 3 * (len(time_hits) - 1)
        reasons.append(f"Matched time tokens: {', '.join(sorted(set(time_hits)))[:60]}")

    # Sequence tokens (#2, S3)
    sequence_hits = [token for token in show_row['sequence_tokens'] if token_in_text(token, text_upper, text_compact)]
    if sequence_hits:
        score += 8
        reasons.append(f"Matched sequence tokens: {', '.join(sorted(set(sequence_hits)))[:60]}")

    return score, reasons


def compute_confidence(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    if score >= 30:
        return "Low"
    return "Very Low"


def map_meta_to_shows(meta_df: pd.DataFrame, show_catalog: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map meta ads campaigns to shows using fuzzy token matching."""
    if meta_df.empty or show_catalog.empty:
        return pd.DataFrame(), meta_df

    catalog_records = show_catalog.to_dict('records')
    mapped_rows: List[Dict] = []
    unmatched_rows: List[Dict] = []

    text_columns = [
        col for col in meta_df.columns
        if any(keyword in col for keyword in ['campaign', 'ad_set', 'adset', 'ad name', 'ad_name'])
    ]

    for _, row in meta_df.iterrows():
        combined_text_parts = [normalize_upper(row.get(col, '')) for col in text_columns]
        combined_text = " ".join(part for part in combined_text_parts if part)
        text_upper = combined_text
        text_compact = normalize_compact(combined_text)

        best_match = None
        best_score = 0
        best_reasons: List[str] = []

        for show_row in catalog_records:
            score, reasons = score_show_match(text_upper, text_compact, show_row)
            if score > best_score:
                best_match = show_row
                best_score = score
                best_reasons = reasons

        row_dict = row.to_dict()
        if best_match and best_score >= 30:
            row_dict['show_id'] = best_match['show_id']
            row_dict['show_name'] = best_match['show_name']
            row_dict['country'] = best_match.get('country')
            row_dict['mapping_score'] = best_score
            row_dict['mapping_confidence'] = compute_confidence(best_score)
            row_dict['mapping_notes'] = "; ".join(best_reasons)
            mapped_rows.append(row_dict)
        else:
            row_dict['mapping_score'] = best_score
            row_dict['mapping_confidence'] = compute_confidence(best_score)
            row_dict['mapping_notes'] = "No confident show match"
            unmatched_rows.append(row_dict)

    mapped_df = pd.DataFrame(mapped_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)
    return mapped_df, unmatched_df


def standardize_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize sales DataFrame structure."""
    sales_clean = sales_df.copy()
    sales_clean.columns = [col.strip() if isinstance(col, str) else col for col in sales_clean.columns]

    rename_map = {}
    for col in sales_clean.columns:
        lower = normalize_upper(col)
        if lower in {"SHOW ID", "SHOW_ID", "SHOWID"}:
            rename_map[col] = "Show ID"
        elif lower in {"SHOW NAME", "SHOW", "NAME"}:
            rename_map[col] = "Show Name"
        elif lower in {"SHOW DATE", "SHOW_DATE", "DATE"}:
            rename_map[col] = "Show Date"
        elif lower in {"REPORT DATE", "REPORT_DATE"}:
            rename_map[col] = "Report Date"
        elif lower in {"CAPACITY"}:
            rename_map[col] = "Capacity"
        elif lower in {"TOTAL SOLD", "TOTAL_SOLD", "SOLD"}:
            rename_map[col] = "Total Sold"
        elif lower in {"REMAINING"}:
            rename_map[col] = "Remaining"
        elif lower in {"ATP", "AVG TICKET PRICE", "AVERAGE TICKET PRICE"}:
            rename_map[col] = "ATP"
        elif lower in {"COUNTRY", "MARKET"}:
            rename_map[col] = "Country"

    sales_clean = sales_clean.rename(columns=rename_map)

    date_columns = ['Show Date', 'Report Date']
    for col in date_columns:
        if col in sales_clean.columns:
            sales_clean[col] = pd.to_datetime(sales_clean[col], errors='coerce')

    numeric_cols = ['Capacity', 'Total Sold', 'Remaining', 'ATP']
    for col in numeric_cols:
        if col in sales_clean.columns:
            if col == 'ATP':
                sales_clean[col] = sales_clean[col].astype(str).str.replace(r'[$,₹£€]', '', regex=True)
            sales_clean[col] = pd.to_numeric(sales_clean[col], errors='coerce')

    if 'Show ID' not in sales_clean.columns and 'Show Name' in sales_clean.columns:
        sales_clean['Show ID'] = sales_clean.apply(
            lambda row: create_show_identifier_from_name(row.get('Show Name'), row.get('Show Date')),
            axis=1
        )

    return sales_clean


def standardize_meta_data(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize meta ads dataframe columns for consistent processing."""
    if meta_df.empty:
        return meta_df

    meta_clean = meta_df.copy()
    meta_clean.columns = sanitize_column_names(list(meta_clean.columns))
    # Ensure key textual columns are kept as string
    for text_col in ['campaign_name', 'ad_set_name', 'ad_name']:
        if text_col in meta_clean.columns:
            meta_clean[text_col] = meta_clean[text_col].astype(str)

    return meta_clean

# Streamlit Cloud Configuration
st.set_page_config(
    page_title="DiA - Show Analytics Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS optimized for cloud deployment
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .health-good { color: #28a745; }
    .health-warning { color: #ffc107; }
    .health-critical { color: #dc3545; }
    .show-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
@@ -43,52 +428,54 @@ st.markdown("""
        .show-header h2 { font-size: 1.2rem; }
        .metric-card { padding: 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CLOUD-OPTIMIZED DATA PROCESSING
# ================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_google_sheet_data(sheet_url):
    """Load data from Google Sheets URL - Cloud optimized with error handling"""
    try:
        if 'docs.google.com' not in sheet_url:
            st.error("Please provide a valid Google Sheets URL")
            return None
        
        # Extract sheet ID
        if '/d/' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        else:
            st.error("Invalid Google Sheets URL format")
            return None
        
        # Convert to CSV export URL using provided gid when available
        gid_match = re.search(r"gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else "0"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Load with timeout and error handling
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Basic data validation
        if df.empty:
            st.error("Loaded data is empty. Please check the Google Sheet.")
            return None
        
        return df
        
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again or check your connection.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Error loading Google Sheet: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

@@ -155,97 +542,83 @@ def parse_funnel_enhanced(campaign_name, result_indicator=""):
        'F1': ['F1', 'FUN1', 'LPVIEW', 'LANDING'],
        'F2': ['F2', 'FUN2', 'ADDTOCART', 'ATC', 'CART'],
        'F3': ['F3', 'FUN3', 'PURCHASE', 'CONV', 'CHECKOUT', 'SALES']
    }
    
    # Check result indicator first
    for funnel, keywords in funnel_keywords.items():
        for keyword in keywords:
            if keyword in result:
                return funnel
    
    # Check campaign name
    for funnel, keywords in funnel_keywords.items():
        for keyword in keywords:
            if keyword in name:
                return funnel
    
    # Legacy patterns
    if 'INTEREST' in name or 'TARGET' in name:
        return 'Legacy'
    
    return "Unclassified"

@st.cache_data
def process_data_optimized(sales_df, meta_df):
    """Optimized data processing with dynamic show mapping."""
    try:
        sales_clean = standardize_sales_data(sales_df)
        show_catalog = build_show_catalog(sales_clean)

        if meta_df is None or meta_df.empty:
            return sales_clean, pd.DataFrame(), pd.DataFrame(), show_catalog

        meta_clean = standardize_meta_data(meta_df)

        if 'funnel' not in meta_clean.columns:
            if 'ad_set_name' in meta_clean.columns or 'adset_name' in meta_clean.columns:
                meta_clean['funnel'] = meta_clean.apply(
                    lambda row: parse_funnel_enhanced(
                        row.get('ad_set_name', row.get('adset_name', '')),
                        row.get('result_indicator', '')
                    ), axis=1
                )
            else:
                meta_clean['funnel'] = meta_clean.get('result_indicator', pd.Series(dtype=str)).apply(parse_funnel_enhanced)

        mapped_df, unmatched_df = map_meta_to_shows(meta_clean, show_catalog)

        if not mapped_df.empty and 'country' in mapped_df.columns:
            mapped_df['country'] = mapped_df['country'].fillna(mapped_df['show_id'].apply(classify_country))
        elif not mapped_df.empty:
            mapped_df['country'] = mapped_df['show_id'].apply(classify_country)

        return sales_clean, mapped_df, unmatched_df, show_catalog

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return sales_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def classify_country(show_id):
    """Optimized country classification"""
    if pd.isna(show_id):
        return "Unknown"
    
    city_code = str(show_id).split('_')[0]
    
    country_mapping = {
        'WDC': 'US', 'DAL': 'US', 'IAH': 'US', 'SMF': 'US',
        'TR': 'CA', 'OTW': 'CA',
        'BLR': 'IN', 'MOB': 'IN', 'DEL': 'IN'
    }
    
    return country_mapping.get(city_code, "Unknown")

def calculate_metrics_optimized(show_sales, show_ads):
    """Optimized metrics calculation for cloud performance"""
    if show_sales.empty:
        return None
    
    try:
        latest = show_sales.iloc[-1]
        
        # Basic metrics with error handling
@@ -599,147 +972,237 @@ def main():
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Sources")
        
        # Google Sheets Integration
        st.subheader("🌐 Google Sheets Integration")
        sheet_url = st.text_input(
            "Online Ticket Sale Sheet URL",
            value="https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/edit?gid=0#gid=0",
            help="Paste the Google Sheets URL for sales data"
        )
        
        if st.button("🔄 Load from Google Sheets", type="primary"):
            with st.spinner("Loading data from Google Sheets..."):
                sales_data = load_google_sheet_data(sheet_url)
                if sales_data is not None:
                    st.session_state['sales_data'] = sales_data
                    st.session_state.data_loaded = True
                    st.success("✅ Sales data loaded successfully!")
                    st.rerun()
        
        st.divider()
        
        # File Upload
        st.subheader("📁 File Upload")
        meta_files = st.file_uploader(
            "Upload Meta Ads Data (up to 3 CSV files)",
            type="csv",
            key="meta",
            help="Upload one or more Meta Ads exports to consolidate",
            accept_multiple_files=True
        )

        if meta_files:
            selected_files = meta_files[:3]
            if len(meta_files) > 3:
                st.warning("⚠️ Only the first 3 files were loaded. Please upload a maximum of 3 files at a time.")

            meta_frames = []
            for uploaded_file in selected_files:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                df['source_file'] = uploaded_file.name
                meta_frames.append(df)

            if meta_frames:
                meta_data = pd.concat(meta_frames, ignore_index=True)
                st.session_state['meta_data'] = meta_data
                st.session_state['meta_sources'] = [file.name for file in selected_files]
                st.success(f"✅ Loaded {len(selected_files)} Meta Ads file(s) with {len(meta_data):,} rows")

        # Data status
        st.subheader("📊 Data Status")
        sales_status = "✅ Loaded" if 'sales_data' in st.session_state else "❌ Not loaded"
        if 'meta_data' in st.session_state:
            source_count = len(st.session_state.get('meta_sources', []))
            source_note = f" ({source_count} file{'s' if source_count != 1 else ''})" if source_count else ""
            meta_status = f"✅ Loaded{source_note}"
        else:
            meta_status = "❌ Not loaded"

        st.write(f"**Sales Data:** {sales_status}")
        st.write(f"**Meta Ads Data:** {meta_status}")

        if st.button("🗑️ Clear All Data"):
            for key in ['sales_data', 'meta_data', 'meta_sources']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.data_loaded = False
            st.rerun()

    # Main content
    if 'sales_data' in st.session_state and 'meta_data' in st.session_state:
        sales_data = st.session_state['sales_data']
        meta_data = st.session_state['meta_data']

        # Process data
        with st.spinner("🔄 Processing and mapping data..."):
            sales_clean, meta_mapped, meta_unmatched, show_catalog = process_data_optimized(sales_data, meta_data)

        # Show mapping results
        if not meta_mapped.empty or (meta_unmatched is not None and not meta_unmatched.empty):
            total_campaigns = len(meta_data)
            mapped_campaigns = len(meta_mapped)
            unmatched_campaigns = len(meta_unmatched) if meta_unmatched is not None else 0
            mapping_rate = (mapped_campaigns / total_campaigns * 100) if total_campaigns else 0

            # Mapping status in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Campaigns", f"{total_campaigns:,}")
            with col2:
                st.metric("Mapped", f"{mapped_campaigns:,}")
            with col3:
                st.metric("Mapping Rate", f"{mapping_rate:.1f}%")
            with col4:
                unique_shows = meta_mapped['show_id'].nunique() if not meta_mapped.empty else 0
                st.metric("Unique Shows", unique_shows)

            if unmatched_campaigns:
                st.warning(f"⚠️ {unmatched_campaigns} campaign row(s) could not be confidently mapped. Review them in the mapping overview below.")

            # Mapping overview tables
            with st.expander("🔍 Mapping Overview", expanded=False):
                if not meta_mapped.empty:
                    mapping_display_cols = [
                        col for col in [
                            'source_file', 'campaign_name', 'ad_set_name', 'ad_name',
                            'show_id', 'show_name', 'mapping_confidence', 'mapping_score', 'mapping_notes'
                        ] if col in meta_mapped.columns
                    ]
                    st.subheader("Mapped Campaigns")
                    st.dataframe(
                        meta_mapped[mapping_display_cols].sort_values('mapping_score', ascending=False),
                        use_container_width=True
                    )

                    spend_col = next((col for col in meta_mapped.columns if 'amount_spent' in col), None)
                    results_col = 'results' if 'results' in meta_mapped.columns else None

                    summary_cols = ['show_id', 'show_name']
                    agg_dict = {'mapping_score': 'mean', 'mapping_confidence': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown', 'source_file': lambda x: ', '.join(sorted(set(filter(None, x))))}
                    if spend_col:
                        agg_dict[spend_col] = 'sum'
                    if results_col:
                        agg_dict[results_col] = 'sum'

                    mapping_summary = meta_mapped.groupby(summary_cols).agg(agg_dict).reset_index()
                    rename_dict = {
                        'mapping_score': 'Avg. Score',
                        'mapping_confidence': 'Top Confidence',
                        'source_file': 'Source Files'
                    }
                    if spend_col:
                        rename_dict[spend_col] = 'Total Spend'
                    if results_col:
                        rename_dict[results_col] = 'Total Results'
                    mapping_summary = mapping_summary.rename(columns=rename_dict)

                    st.subheader("Show-Level Summary")
                    st.dataframe(mapping_summary, use_container_width=True)

                if meta_unmatched is not None and not meta_unmatched.empty:
                    unmatched_display_cols = [
                        col for col in [
                            'source_file', 'campaign_name', 'ad_set_name', 'ad_name', 'mapping_score', 'mapping_notes'
                        ] if col in meta_unmatched.columns
                    ]
                    st.subheader("Unmatched Campaigns")
                    st.dataframe(meta_unmatched[unmatched_display_cols], use_container_width=True)

            # Show selection
            show_options = pd.DataFrame()
            if isinstance(show_catalog, pd.DataFrame) and not show_catalog.empty:
                show_options = show_catalog[['show_id', 'show_name']].drop_duplicates()
            elif not meta_mapped.empty:
                show_options = meta_mapped[['show_id', 'show_name']].drop_duplicates()

            if show_options.empty and not meta_mapped.empty:
                show_options = meta_mapped[['show_id', 'show_name']].drop_duplicates()

            if show_options.empty:
                st.info("No shows available for selection yet. Please review the mapping overview above.")
            else:
                show_options['label'] = show_options.apply(
                    lambda row: f"{row['show_name']} ({row['show_id']})" if pd.notna(row['show_name']) else row['show_id'],
                    axis=1
                )
                show_options = show_options.sort_values('label')
                available_labels = show_options['label'].tolist()

                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_show = st.selectbox(
                        "🎪 Select Show for Analysis",
                        available_labels,
                        key="show_select"
                    )
                with col2:
                    if st.button("🔄 Refresh"):
                        st.cache_data.clear()
                        st.rerun()

                if selected_show:
                    show_id_value = show_options.loc[show_options['label'] == selected_show, 'show_id'].iloc[0]
                    # Filter data for selected show
                    show_sales = sales_clean[sales_clean['Show ID'] == show_id_value].copy()
                    show_ads = meta_mapped[meta_mapped['show_id'] == show_id_value].copy()

                    # Calculate metrics
                    metrics = calculate_metrics_optimized(show_sales, show_ads)

                    if metrics:
                        # Show header
                        country = show_ads['country'].iloc[0] if not show_ads.empty else 'Unknown'
                        st.markdown(f"""
                        <div class="show-header">
                            <h2>🎭 {metrics['show_name']} ({metrics['show_id']})</h2>
                            <p><strong>Capacity:</strong> {metrics['capacity']:,} |
                               <strong>Days to Show:</strong> {metrics['days_to_show']} |
                               <strong>Country:</strong> {country}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Health Dashboard
                        st.subheader("🏥 Show Health Dashboard")
                        health_fig = create_health_dashboard_optimized(metrics)
                        if health_fig:
                            st.plotly_chart(health_fig, use_container_width=True)
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric(
                            "📊 Sold", 
                            f"{metrics['total_sold']:,}", 
                            f"{metrics['sold_percentage']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "💰 Revenue", 
                            f"${metrics['revenue']:,.0f}",
                            f"ATP: ${metrics['atp']:.0f}" if metrics['atp'] > 0 else None
                        )
                    
                    with col3:
                        roas_delta = f"{metrics['roas']:.1f}x ROAS" if metrics['roas'] > 0 else None
                        st.metric(
                            "💸 Ad Spend", 
                            f"${metrics['total_spend']:,.0f}",
                            roas_delta
                        )
@@ -798,104 +1261,104 @@ def main():
                            potential_revenue = metrics['atp'] * metrics['capacity'] if metrics['atp'] > 0 else 0
                            revenue_completion = (metrics['revenue'] / potential_revenue * 100) if potential_revenue > 0 else 0
                            st.metric("Revenue Progress", f"{revenue_completion:.1f}%",
                                    f"${metrics['revenue']:,.0f} / ${potential_revenue:,.0f}")
                    
                    # Funnel Analysis
                    if not show_ads.empty:
                        st.subheader("🔄 Funnel Performance Analysis")
                        funnel_fig = create_funnel_analysis_optimized(show_ads)
                        if funnel_fig:
                            st.plotly_chart(funnel_fig, use_container_width=True)
                        
                        # Performance Timeline
                        st.subheader("📈 Performance Over Time")
                        timeline_fig = create_performance_timeline_optimized(show_ads)
                        if timeline_fig:
                            st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Advanced Analytics
                    with st.expander("📊 Advanced Analytics & Raw Data"):
                        tab1, tab2, tab3 = st.tabs(["🗺️ Mapping Report", "📈 Sales Data", "📱 Ads Data"])
                        
                        with tab1:
                            st.subheader("Data Mapping Analysis")
                            
                            if not meta_mapped.empty:
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Show distribution
                                    show_distribution = meta_mapped['show_id'].value_counts().head(10)
                                    st.subheader("Show Distribution (Top 10)")
                                    st.bar_chart(show_distribution)

                                with col2:
                                    # Funnel distribution
                                    funnel_distribution = meta_mapped['funnel'].value_counts()
                                    st.subheader("Funnel Distribution")
                                    st.bar_chart(funnel_distribution)

                                # Unmapped campaigns sample
                                if meta_unmatched is not None and not meta_unmatched.empty:
                                    st.subheader("Unmatched Campaigns (Sample)")
                                    sample_cols = [col for col in ['campaign_name', 'ad_set_name', 'source_file'] if col in meta_unmatched.columns]
                                    st.dataframe(meta_unmatched[sample_cols].head(10))
                        
                        with tab2:
                            st.subheader("Sales Data")
                            st.dataframe(show_sales)
                        
                        with tab3:
                            st.subheader("Ads Data") 
                            st.dataframe(show_ads)
                
                else:
                    st.error("❌ Unable to calculate metrics for this show. Please check data quality.")
        
        else:
            st.warning("⚠️ No shows found in Meta ads data. Please check mapping configuration.")

            # Debug information
            if not meta_data.empty:
                debug_meta = standardize_meta_data(meta_data)
                if 'ad_set_name' in debug_meta.columns:
                    st.subheader("🔍 Campaign Mapping Debug")
                    st.write("Sample campaign names and their mapping results:")

                    sample_campaigns = debug_meta['ad_set_name'].dropna().unique()[:10]
                    debug_data = []

                    for campaign in sample_campaigns:
                        parsed_id = parse_show_id_enhanced(campaign)
                        parsed_funnel = parse_funnel_enhanced(campaign)
                        debug_data.append({
                            'Campaign Name': campaign,
                            'Show ID': parsed_id,
                            'Funnel': parsed_funnel
                        })

                    st.dataframe(pd.DataFrame(debug_data))
    
    else:
        # Welcome screen with instructions
        st.info("""
        ## 🚀 Welcome to DiA v2.0 - Enhanced Show Analytics
        
        ### 📋 Quick Start Guide
        
        #### Step 1: Load Your Data
        1. **Google Sheets**: Use the provided URL in the sidebar for automatic loading
        2. **File Upload**: Upload your Meta Ads CSV data
        
        #### Step 2: Analyze Your Shows
        - View **Health Dashboard** with 5 key indicators
        - Monitor **Sales Velocity** vs daily targets  
        - Track **ROAS** and **Cost per Ticket**
        - Analyze **Funnel Performance**
        
        ### 🎯 Key Features
        
        #### 🏥 Health Dashboard
        **5 Quick Visual Indicators** for instant decision making:
        - 🟢 **Green**: Excellent performance
        - 🟡 **Yellow**: Needs attention
        - 🔴 **Red**: Immediate action required
        
        #### 📊 Enhanced Metrics
        All metrics from **Online Ticket Sale Sheet (Column N)**:
        - **Ticket Cost** = Show Budget ÷ Total Tickets Sold
        - **Daily Sales Target** vs Last 7-day Average
        - **Funnel Efficiency**: LP Views/Cart Adds/Conversions per ticket
        - **ROAS**: Revenue ÷ Ad Spend
        
        #### 🗺️ Smart Mapping
        Automatically recognizes show variations:
        - `WDC_0927`, `WDC_0927_S2`, `S3`, `S4`
        - "Washington DC #2" → `WDC_0927_S2`
        - Legacy patterns: "US-DC-Sales-2024 - Interest"
        
        ### 💡 Pro Tips
        - **Health Dashboard** gives instant go/no-go decisions
        - **Red indicators** = immediate action needed
        - **Yellow indicators** = monitor closely
        - **Green indicators** = performing well
        
        ---
        *Ready to get started? Load your data using the sidebar! 🚀*
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**DiA v2.0** - Cloud Analytics")
    with col2:
        st.markdown("🎭 Show Performance Dashboard")
    with col3:
        if st.button("ℹ️ About"):
            st.info("""
            **DiA v2.0** - Enhanced Show Analytics Dashboard
            
            ✨ **New in v2.0:**
            - 🏥 Health Dashboard with 5 visual indicators
            - 🎯 Metrics from Online Ticket Sale Sheet
            - 🗺️ Enhanced show ID parsing with fallback logic
            - 🔄 Improved funnel detection for legacy campaigns
            - ☁️ Optimized for Streamlit Community Cloud
            - 📱 Mobile-responsive design
            
            Built for fast decisions and actionable insights.
            """)

if __name__ == "__main__":
    main()
