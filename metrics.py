import pandas as pd
from datetime import datetime

def calculate_days_to_show(show_date_str):
    try:
        show_date = pd.to_datetime(show_date_str)
        today = pd.to_datetime(datetime.today().date())
        return max((show_date - today).days, 0)
    except Exception:
        return None

def compute_ticket_cost(spend_total, tickets_sold):
    return spend_total / tickets_sold if tickets_sold else None

def compute_roas(atp, capacity, spend_total):
    return (atp * capacity) / spend_total if spend_total else None

def compute_daily_sales_target(tickets_remaining, days_left):
    return tickets_remaining / days_left if days_left else None

def compute_funnel_efficiency(df_ads_grouped):
    result = {}
    clicks = df_ads_grouped.get("clicks", 0)
    lp_views = df_ads_grouped.get("lpviews", 0)
    add_to_cart = df_ads_grouped.get("addtocart", 0)
    tickets_sold = df_ads_grouped.get("tickets_sold", 0)
    safe_div = lambda x, y: round(x / y, 2) if y else None
    result["clicks_per_ticket"] = safe_div(clicks, tickets_sold)
    result["lpviews_per_ticket"] = safe_div(lp_views, tickets_sold)
    result["addtocart_per_ticket"] = safe_div(add_to_cart, tickets_sold)
    return result

def summarize_show_metrics(show_row, spend_total):
    try:
        sold = int(show_row.get("total_sold", 0))
        capacity = int(show_row.get("capacity", 0))
        atp = float(show_row.get("atp", 0))
        date = show_row.get("date")
        days_left = calculate_days_to_show(date)
        remaining = capacity - sold
        return {
            "days_to_show": days_left,
            "sold": sold,
            "capacity": capacity,
            "remaining": remaining,
            "ticket_cost": compute_ticket_cost(spend_total, sold),
            "daily_target": compute_daily_sales_target(remaining, days_left),
            "roas": compute_roas(atp, capacity, spend_total),
            "spend_total": spend_total
        }
    except Exception:
        return {}
