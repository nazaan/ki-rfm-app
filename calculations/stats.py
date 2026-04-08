"""
calculations/stats.py
---------------------
Quick Stats bar calculations.

Input:  clean transaction DataFrame from data_loader
        + customer-level RFM DataFrame from rfm_engine + segments

Output: a single dict consumed directly by the UI stats bar.

Metrics:
    - Total revenue
    - Total customers
    - Average Order Value (AOV)
    - Purchase frequency (avg orders per customer)
    - Churn rate (% customers with no purchase in last N days)
    - Revenue from top 5% of customers
    - Average customer lifetime (days)
    - Repeat purchase rate (% customers with > 1 order)
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_quick_stats(
    transactions_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    churn_days: int = 90,
    reference_date: Optional[pd.Timestamp] = None,
) -> dict:
    if reference_date is None:
        reference_date = transactions_df["order_date"].max() + pd.Timedelta(days=1)

    total_customers    = len(rfm_df)
    total_transactions = len(transactions_df)
    total_revenue      = transactions_df["order_value"].sum()

    return {
        "total_customers":         total_customers,
        "total_transactions":      total_transactions,
        "total_revenue":           round(total_revenue, 2),
        "avg_order_value":         _avg_order_value(transactions_df),
        "median_order_value":      round(transactions_df["order_value"].median(), 2),
        "avg_orders_per_customer": _avg_orders_per_customer(rfm_df),
        "repeat_purchase_rate":    _repeat_purchase_rate(rfm_df),
        "churn_rate":              _churn_rate(rfm_df, churn_days),
        "churn_days_threshold":    churn_days,
        "avg_days_since":          round(rfm_df["days_since"].mean(), 0),
        "top5_revenue_pct":        _top_n_revenue_pct(rfm_df, pct=0.05),
        "top20_revenue_pct":       _top_n_revenue_pct(rfm_df, pct=0.20),
        "avg_lifetime_days":       round(rfm_df["lifetime_days"].mean(), 0),
        "date_range_start":        transactions_df["order_date"].min().strftime("%d %b %Y"),
        "date_range_end":          transactions_df["order_date"].max().strftime("%d %b %Y"),
        "reference_date":          reference_date.strftime("%d %b %Y"),
    }


def _avg_order_value(transactions_df: pd.DataFrame) -> float:
    if len(transactions_df) == 0:
        return 0.0
    return round(transactions_df["order_value"].mean(), 2)


def _avg_orders_per_customer(rfm_df: pd.DataFrame) -> float:
    if len(rfm_df) == 0:
        return 0.0
    return round(rfm_df["order_count"].mean(), 1)


def _repeat_purchase_rate(rfm_df: pd.DataFrame) -> float:
    if len(rfm_df) == 0:
        return 0.0
    repeat = (rfm_df["order_count"] > 1).sum()
    return round(repeat / len(rfm_df) * 100, 1)


def _churn_rate(rfm_df: pd.DataFrame, churn_days: int) -> float:
    if len(rfm_df) == 0:
        return 0.0
    churned = (rfm_df["days_since"] > churn_days).sum()
    return round(churned / len(rfm_df) * 100, 1)


def _top_n_revenue_pct(rfm_df: pd.DataFrame, pct: float = 0.05) -> float:
    if len(rfm_df) == 0 or rfm_df["total_spend"].sum() == 0:
        return 0.0
    n = max(1, int(np.ceil(len(rfm_df) * pct)))
    top_revenue = rfm_df["total_spend"].nlargest(n).sum()
    return round(top_revenue / rfm_df["total_spend"].sum() * 100, 1)


def compute_health_score(stats: dict, segment_summary: pd.DataFrame) -> dict:
    scores = []

    rpr = stats.get("repeat_purchase_rate", 0)
    scores.append(min(rpr / 50 * 100, 100))

    churn = stats.get("churn_rate", 100)
    scores.append(max(0, (1 - churn / 80) * 100))

    seg   = segment_summary.set_index("segment")
    total = segment_summary["count"].sum()
    if total > 0:
        champ_pct = seg.loc["Champions", "count"] / total * 100 if "Champions" in seg.index else 0
        scores.append(min(champ_pct / 20 * 100, 100))
    else:
        scores.append(0)

    at_risk_count = 0
    for seg_name in ["At Risk", "Cannot Lose"]:
        if seg_name in seg.index:
            at_risk_count += seg.loc[seg_name, "count"]
    at_risk_pct = at_risk_count / total * 100 if total > 0 else 0
    scores.append(max(0, (1 - at_risk_pct / 30) * 100))

    top5 = stats.get("top5_revenue_pct", 100)
    scores.append(max(0, (1 - (top5 - 30) / 50) * 100))

    health = round(np.mean(scores), 0)

    if health >= 75:
        verdict, color = "Strong",      "#10BB82"
    elif health >= 50:
        verdict, color = "Moderate",    "#E8A838"
    elif health >= 25:
        verdict, color = "Needs Work",  "#D46A17"
    else:
        verdict, color = "Critical",    "#B03A2E"

    return {"score": int(health), "verdict": verdict, "color": color}


def fmt_currency(value: float, symbol: str = "$") -> str:
    if value >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"
    return f"{symbol}{value:,.2f}"

def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"

def fmt_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{int(value):,}"
