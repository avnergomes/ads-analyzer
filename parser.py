import re

def normalize_show_id(raw_name: str) -> str:
    \"\"\"Extract consistent show ID from raw campaign or show name\"\"\"
    if not isinstance(raw_name, str):
        return ""
    
    # Common show ID patterns: CITY_DATE(_S2), remove suffixes
    base = re.sub(r"_S\\d+$", "", raw_name)  # Remove _S2, _S3 etc
    base = re.sub(r"[^\\w\\-]", "", base)    # Remove special characters
    base = base.upper().strip()

    # Try fallback patterns
    fallback = re.findall(r"[A-Z]{2,}-[A-Z]{2,}-\\w+", raw_name)
    if fallback:
        return fallback[0]

    return base

def normalize_funnel_label(raw_label: str) -> str:
    \"\"\"Normalize funnel step naming to consistent format\"\"\"
    if not isinstance(raw_label, str):
        return ""

    raw = raw_label.lower()
    if "click" in raw:
        return "Clicks"
    elif "lpview" in raw or "landing" in raw:
        return "LP Views"
    elif "add" in raw and "cart" in raw:
        return "Add to Cart"
    elif "conversion" in raw or "result" in raw:
        return "Conversions"
    return raw_label.title()