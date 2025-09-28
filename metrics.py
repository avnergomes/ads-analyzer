"""Business metrics helpers used by the Streamlit dashboards."""
from __future__ import annotations

from datetime import datetime
from typing import Mapping, MutableMapping, Optional

import pandas as pd


def calculate_days_to_show(show_date: object) -> Optional[int]:
    """Return the number of days until the show date (0 if past)."""
    try:
        show_ts = pd.to_datetime(show_date)
        if pd.isna(show_ts):
            return None
        today = pd.Timestamp(datetime.today().date())
        delta = (show_ts.normalize() - today).days
        return max(int(delta), 0)
    except Exception:
        return None


def compute_ticket_cost(spend_total: float, tickets_sold: float) -> Optional[float]:
    if not tickets_sold:
        return None
    return float(spend_total) / float(tickets_sold)


def compute_roas(atp: float, capacity: float, spend_total: float) -> Optional[float]:
    potential_revenue = float(atp or 0) * float(capacity or 0)
    if not spend_total:
        return None
    return potential_revenue / float(spend_total)


def compute_daily_sales_target(tickets_remaining: float, days_left: Optional[int]) -> Optional[float]:
    if not days_left:
        return None
    if days_left <= 0:
        return None
    return float(tickets_remaining) / float(days_left)


def summarize_show_metrics(show_row: Mapping[str, object], spend_total: float) -> MutableMapping[str, Optional[float]]:
    """Build the KPI dictionary shown in the dashboard for a single show."""
    sold = float(show_row.get("total_sold", 0) or 0)
    capacity = float(show_row.get("capacity", 0) or 0)
    atp = float(show_row.get("atp", 0) or 0)
    days_left = calculate_days_to_show(show_row.get("date"))
    remaining = max(capacity - sold, 0)

    return {
        "days_to_show": days_left,
        "sold": sold,
        "capacity": capacity,
        "remaining": remaining,
        "ticket_cost": compute_ticket_cost(spend_total, sold),
        "daily_target": compute_daily_sales_target(remaining, days_left),
        "roas": compute_roas(atp, capacity, spend_total),
        "spend_total": spend_total,
    }


def compute_funnel_efficiency(
    show_ads_df: pd.DataFrame,
    show_sales_row: Mapping[str, object] | None = None,
) -> MutableMapping[str, Optional[float]]:
    """Calculate per-ticket funnel efficiency for a selected show."""
    aggregated = show_ads_df[[
        "clicks",
        "lp_views",
        "add_to_cart",
        "conversions",
    ]].sum(numeric_only=True)

    tickets_sold = None
    if show_sales_row is not None:
        tickets_sold = float(show_sales_row.get("total_sold", 0) or 0)
    if not tickets_sold:
        tickets_sold = float(show_ads_df.get("conversions", pd.Series(dtype=float)).sum())

    def _safe_div(value: float, base: float) -> Optional[float]:
        if not base:
            return None
        return round(float(value) / float(base), 2)

    return {
        "Clicks per ticket": _safe_div(aggregated.get("clicks", 0), tickets_sold),
        "LP views per ticket": _safe_div(aggregated.get("lp_views", 0), tickets_sold),
        "Add to cart per ticket": _safe_div(aggregated.get("add_to_cart", 0), tickets_sold),
        "Conversions": float(aggregated.get("conversions", 0)),
        "Tickets sold": tickets_sold,
    }
