"""
calculations/rfm_engine.py
--------------------------
Core RFM scoring logic.

Takes the clean DataFrame from data_loader and returns a customer-level
DataFrame with R, F, M raw values and 1-5 scores.

Input DataFrame columns (guaranteed by data_loader):
    customer_id | customer_name | order_date | order_value

Output DataFrame columns:
    customer_id | customer_name
    last_order_date | days_since        ← Recency raw
    order_count                         ← Frequency raw
    total_spend                         ← Monetary raw
    r_score | f_score | m_score         ← 1–5 scores
    rfm_total                           ← sum of scores (3–15)
    rfm_score_str                       ← e.g. "5-4-3" for sorting/display
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def compute_rfm(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    recency_weight: float = 1.0,
    frequency_weight: float = 1.0,
    monetary_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Main entry point. Takes clean transaction DataFrame, returns
    customer-level RFM DataFrame.

    Args:
        df:               Clean DataFrame from data_loader
        reference_date:   Date to calculate recency from.
                          Defaults to max(order_date) + 1 day.
                          UI should let user override this.
        recency_weight:   Multiplier for R score (default 1.0)
        frequency_weight: Multiplier for F score (default 1.0)
        monetary_weight:  Multiplier for M score (default 1.0)

    Returns:
        Customer-level DataFrame with all RFM columns.
    """
    if reference_date is None:
        reference_date = df["order_date"].max() + pd.Timedelta(days=1)

    # ── Step 1: Aggregate to customer level ───────────────────────────────
    customers = _aggregate(df, reference_date)

    # ── Step 2: Score each dimension 1–5 ──────────────────────────────────
    customers["r_score"] = _percentrank_score(
        customers["days_since"], invert=True   # lower days = better
    )
    customers["f_score"] = _percentrank_score(
        customers["order_count"], invert=False
    )
    customers["m_score"] = _percentrank_score(
        customers["total_spend"], invert=False
    )

    # ── Step 3: Weighted total ─────────────────────────────────────────────
    customers["rfm_total"] = (
        customers["r_score"] * recency_weight
        + customers["f_score"] * frequency_weight
        + customers["m_score"] * monetary_weight
    ).round(2)

    # ── Step 4: Score string for display / matrix lookup ──────────────────
    customers["rfm_score_str"] = (
        customers["r_score"].astype(str) + "-"
        + customers["f_score"].astype(str) + "-"
        + customers["m_score"].astype(str)
    )

    return customers.reset_index(drop=True)


# ── Aggregation ────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """
    Collapse transaction rows into one row per customer.
    Keeps the most recent customer_name if multiple exist.
    """
    agg = (
        df.groupby("customer_id", sort=False)
        .agg(
            customer_name=("customer_name", _most_frequent_name),
            last_order_date=("order_date", "max"),
            order_count=("order_date", "count"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            first_order_date=("order_date", "min"),
        )
        .reset_index()
    )

    # Recency: days between last purchase and reference date
    agg["days_since"] = (
        reference_date - agg["last_order_date"]
    ).dt.days.clip(lower=0)

    # Customer lifetime in days (useful for stats sheet later)
    agg["lifetime_days"] = (
        agg["last_order_date"] - agg["first_order_date"]
    ).dt.days.clip(lower=0)

    return agg


def _most_frequent_name(names: pd.Series) -> str:
    """Return the most common non-empty name for a customer."""
    clean = names[names.str.strip() != ""]
    if clean.empty:
        return ""
    return clean.mode().iloc[0]


# ── PERCENTRANK scoring ────────────────────────────────────────────────────

def _percentrank_score(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Score a numeric series from 1 to 5 using percentile ranking.
    Mirrors Excel's PERCENTRANK logic exactly.

    invert=True  → lower raw value gets higher score (used for recency:
                   fewer days since last purchase = better)
    invert=False → higher raw value gets higher score (frequency, monetary)

    Edge cases handled:
    - Single unique value → everyone gets score 3
    - Ties → all tied values get the same score
    """
    n = len(series)

    if n == 0:
        return pd.Series([], dtype=int)

    # Single unique value — no meaningful ranking possible
    if series.nunique() == 1:
        return pd.Series([3] * n, index=series.index)

    # Percentile rank 0.0 – 1.0 for each value
    # method="average" matches Excel PERCENTRANK for ties
    pct = series.rank(method="average", pct=True)

    if invert:
        pct = 1.0 - pct

    # Scale to 1–5
    # multiply by 5, ceil, then clip to [1,5]
    scores = np.ceil(pct * 5).clip(1, 5).astype(int)

    return scores


# ── Convenience: re-score with custom weights ──────────────────────────────

def recompute_weighted_total(
    rfm_df: pd.DataFrame,
    r_weight: float,
    f_weight: float,
    m_weight: float,
) -> pd.DataFrame:
    """
    Recalculate rfm_total without re-aggregating.
    Used by the UI sliders — fast because scores are already computed.
    """
    df = rfm_df.copy()
    df["rfm_total"] = (
        df["r_score"] * r_weight
        + df["f_score"] * f_weight
        + df["m_score"] * m_weight
    ).round(2)
    return df


# ── Matrix helper: 5x5 recency vs frequency grid ──────────────────────────

def build_rfm_matrix(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a 5x5 pivot table of customer counts.
    Rows = R score (5 at top = most recent)
    Cols = F score (1 at left = least frequent)

    Used by the interactive matrix in the UI.
    """
    matrix = (
        rfm_df.groupby(["r_score", "f_score"])
        .size()
        .reset_index(name="count")
        .pivot(index="r_score", columns="f_score", values="count")
        .reindex(index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5])
        .fillna(0)
        .astype(int)
    )
    return matrix


def get_matrix_cell_customers(
    rfm_df: pd.DataFrame,
    r: int,
    f: int,
) -> pd.DataFrame:
    """
    Returns all customers in a given R/F cell of the matrix.
    Called when user clicks a cell in the UI.
    """
    return (
        rfm_df[(rfm_df["r_score"] == r) & (rfm_df["f_score"] == f)][
            ["customer_id", "customer_name", "r_score", "f_score",
             "m_score", "rfm_total", "days_since", "order_count", "total_spend"]
        ]
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )
