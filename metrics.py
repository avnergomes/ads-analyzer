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
    if tickets_sold == 0:
        return None
    return spend_total / tickets_sold


def compute_roas(atp, capacity, spend_total):
    if spend_total == 0:
        return None
    return (atp * capacity) / spend_total


def compute_daily_sales_target(tickets_remaining, days_left):
    if days_left == 0:
        return None
    return tickets_remaining / days_left


def compute_funnel_efficiency(df_ads_grouped):
    """
    Espera um DataFrame já filtrado por show_id.
    Retorna: dict com Clicks per Sale, LP Views per Sale, AddToCart per Sale, etc.
    """
    result = {}

    clicks = df_ads_grouped.get("clicks", 0)
    lp_views = df_ads_grouped.get("lpviews", 0)
    add_to_cart = df_ads_grouped.get("addtocart", 0)
    conv = df_ads_grouped.get("conversions", 0)
    tickets_sold = df_ads_grouped.get("tickets_sold", 0)

    def safe_div(x, y):
        return round(x / y, 2) if y else None

    result["clicks_per_ticket"] = safe_div(clicks, tickets_sold)
    result["lpviews_per_ticket"] = safe_div(lp_views, tickets_sold)
    result["addtocart_per_ticket"] = safe_div(add_to_cart, tickets_sold)

    return result


def summarize_show_metrics(show_row, spend_total):
    """
    show_row: linha do DataFrame da planilha de vendas
    spend_total: total gasto nas campanhas daquele show
    """
    try:
        sold = int(show_row.get("total_sold", 0))
        capacity = int(show_row.get("capacity", 0))
        atp = float(show_row.get("atp", 0))

        date = show_row.get("date")
        days_left = calculate_days_to_show(date)
        remaining = capacity - sold
        ticket_cost = compute_ticket_cost(spend_total, sold)
        daily_target = compute_daily_sales_target(remaining, days_left)
        roas = compute_roas(atp, capacity, spend_total)

        return {
            "days_to_show": days_left,
            "sold": sold,
            "capacity": capacity,
            "remaining": remaining,
            "ticket_cost": ticket_cost,
            "daily_target": daily_target,
            "roas": roas,
            "spend_total": spend_total
        }

    except Exception:
        return {}
