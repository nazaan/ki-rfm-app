"""
calculations/segments.py
------------------------
Assigns RFM segment labels to customers based on their R, F, M scores.

Kept separate from rfm_engine.py so segment rules can be changed
independently of the scoring logic.

Input:  DataFrame from rfm_engine.compute_rfm() — must have r_score,
        f_score, m_score columns.
Output: Same DataFrame with added columns:
            segment       ← string label
            segment_color ← hex for UI display
            segment_rank  ← int 1-11 (1=best) for sorting
"""

import pandas as pd
from typing import Optional


# ── Segment definitions ────────────────────────────────────────────────────
# Order matters: first matching rule wins.
# Each rule is a dict with:
#   name:      display label
#   r_min/max: R score range (inclusive)
#   f_min/max: F score range (inclusive)
#   m_min/max: M score range (inclusive)
#   rank:      1 = best segment (Champions), 11 = worst (Lost)
#   color:     hex background for UI badges
#   text:      hex text color for UI badges
#   action:    short recommended action for Monday Morning Brief

SEGMENT_RULES = [
    {
        "name":   "Champions",
        "r_min": 4, "r_max": 5,
        "f_min": 4, "f_max": 5,
        "m_min": 4, "m_max": 5,
        "rank":   1,
        "color":  "#064F35",
        "text":   "#FFFFFF",
        "action": "Reward them, ask for reviews, and offer early access to new products.",
    },
    {
        "name":   "Loyal Customers",
        "r_min": 2, "r_max": 5,
        "f_min": 3, "f_max": 5,
        "m_min": 3, "m_max": 5,
        "rank":   2,
        "color":  "#0A7A52",
        "text":   "#FFFFFF",
        "action": "Enrol in a loyalty programme. Upsell to premium tier.",
    },
    {
        "name":   "Potential Loyalists",
        "r_min": 3, "r_max": 5,
        "f_min": 1, "f_max": 3,
        "m_min": 1, "m_max": 3,
        "rank":   3,
        "color":  "#10BB82",
        "text":   "#FFFFFF",
        "action": "Offer a membership or subscription. Build the relationship early.",
    },
    {
        "name":   "Recent Customers",
        "r_min": 4, "r_max": 5,
        "f_min": 1, "f_max": 1,
        "m_min": 1, "m_max": 2,
        "rank":   4,
        "color":  "#009084",
        "text":   "#FFFFFF",
        "action": "Onboard well. Send a welcome sequence showing product value.",
    },
    {
        "name":   "Promising",
        "r_min": 3, "r_max": 4,
        "f_min": 1, "f_max": 1,
        "m_min": 1, "m_max": 1,
        "rank":   5,
        "color":  "#4DB89E",
        "text":   "#1A1A2E",
        "action": "Build brand awareness. Offer an introductory discount to drive second purchase.",
    },
    {
        "name":   "Needs Attention",
        "r_min": 3, "r_max": 4,
        "f_min": 2, "f_max": 3,
        "m_min": 2, "m_max": 3,
        "rank":   6,
        "color":  "#D4AC0D",
        "text":   "#1A1A2E",
        "action": "Send a limited-time offer. Re-engage before they drift further.",
    },
    {
        "name":   "About To Sleep",
        "r_min": 2, "r_max": 3,
        "f_min": 1, "f_max": 2,
        "m_min": 1, "m_max": 2,
        "rank":   7,
        "color":  "#E8A838",
        "text":   "#1A1A2E",
        "action": "Share your most popular products. Discount offer to re-activate.",
    },
    {
        "name":   "At Risk",
        "r_min": 1, "r_max": 2,
        "f_min": 3, "f_max": 5,
        "m_min": 3, "m_max": 5,
        "rank":   8,
        "color":  "#D46A17",
        "text":   "#FFFFFF",
        "action": "Send a personalised win-back email. Reference their past purchases.",
    },
    {
        "name":   "Cannot Lose",
        "r_min": 1, "r_max": 1,
        "f_min": 4, "f_max": 5,
        "m_min": 4, "m_max": 5,
        "rank":   9,
        "color":  "#B03A2E",
        "text":   "#FFFFFF",
        "action": "Urgent: call or email personally. Offer renewal or exclusive deal.",
    },
    {
        "name":   "Hibernating",
        "r_min": 1, "r_max": 2,
        "f_min": 1, "f_max": 2,
        "m_min": 1, "m_max": 2,
        "rank":   10,
        "color":  "#7B241C",
        "text":   "#FFFFFF",
        "action": "Suggest other relevant products. Low effort campaign only.",
    },
    {
        "name":   "Lost",
        "r_min": 1, "r_max": 5,
        "f_min": 1, "f_max": 5,
        "m_min": 1, "m_max": 5,
        "rank":   11,
        "color":  "#4A0E0E",
        "text":   "#FFFFFF",
        "action": "Mass reactivation campaign or accept churn. Low priority.",
    },
]

# Quick lookup dicts built once at import time
_SEGMENT_COLOR_MAP  = {s["name"]: s["color"]  for s in SEGMENT_RULES}
_SEGMENT_TEXT_MAP   = {s["name"]: s["text"]   for s in SEGMENT_RULES}
_SEGMENT_RANK_MAP   = {s["name"]: s["rank"]   for s in SEGMENT_RULES}
_SEGMENT_ACTION_MAP = {s["name"]: s["action"] for s in SEGMENT_RULES}

# Ordered list of segment names (best → worst) for UI dropdowns/charts
SEGMENT_NAMES_ORDERED = [s["name"] for s in SEGMENT_RULES]


# ── Main entry point ───────────────────────────────────────────────────────

def assign_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the output of rfm_engine.compute_rfm() and adds segment columns.
    Returns the same DataFrame with added columns:
        segment, segment_color, segment_text, segment_rank, segment_action
    """
    df = rfm_df.copy()

    df["segment"] = df.apply(
        lambda row: _classify(row["r_score"], row["f_score"], row["m_score"]),
        axis=1,
    )

    df["segment_color"]  = df["segment"].map(_SEGMENT_COLOR_MAP)
    df["segment_text"]   = df["segment"].map(_SEGMENT_TEXT_MAP)
    df["segment_rank"]   = df["segment"].map(_SEGMENT_RANK_MAP)
    df["segment_action"] = df["segment"].map(_SEGMENT_ACTION_MAP)

    return df


def _classify(r: int, f: int, m: int) -> str:
    """
    Return the first matching segment name for given R, F, M scores.
    Falls through to 'Lost' if nothing else matches.
    """
    for rule in SEGMENT_RULES:
        if (
            rule["r_min"] <= r <= rule["r_max"]
            and rule["f_min"] <= f <= rule["f_max"]
            and rule["m_min"] <= m <= rule["m_max"]
        ):
            return rule["name"]
    return "Lost"


# ── Segment summary for dashboard ─────────────────────────────────────────

def get_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame with one row per segment.
    Includes all 11 segments even if count is 0 — keeps dashboard stable.
    """
    if "segment" not in rfm_df.columns:
        rfm_df = assign_segments(rfm_df)

    total_customers = len(rfm_df)
    total_revenue   = rfm_df["total_spend"].sum()

    rows = []
    for seg_def in SEGMENT_RULES:
        name   = seg_def["name"]
        subset = rfm_df[rfm_df["segment"] == name]
        count  = len(subset)

        rows.append({
            "segment":       name,
            "count":         count,
            "pct_customers": round(count / total_customers * 100, 1) if total_customers > 0 else 0.0,
            "avg_rfm":       round(subset["rfm_total"].mean(), 1)    if count > 0 else 0.0,
            "avg_r":         round(subset["r_score"].mean(), 1)      if count > 0 else 0.0,
            "avg_f":         round(subset["f_score"].mean(), 1)      if count > 0 else 0.0,
            "avg_m":         round(subset["m_score"].mean(), 1)      if count > 0 else 0.0,
            "avg_spend":     round(subset["total_spend"].mean(), 2)  if count > 0 else 0.0,
            "total_revenue": round(subset["total_spend"].sum(), 2),
            "pct_revenue":   round(subset["total_spend"].sum() / total_revenue * 100, 1) if total_revenue > 0 else 0.0,
            "color":         seg_def["color"],
            "text":          seg_def["text"],
            "rank":          seg_def["rank"],
            "action":        seg_def["action"],
        })

    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


# ── Helpers for export / filtering ────────────────────────────────────────

def get_customers_by_segment(
    rfm_df: pd.DataFrame,
    segment_name: str,
) -> pd.DataFrame:
    """
    Returns all customers in a given segment, sorted by total_spend desc.
    Used by CSV export and action cards.
    """
    if "segment" not in rfm_df.columns:
        rfm_df = assign_segments(rfm_df)

    return (
        rfm_df[rfm_df["segment"] == segment_name]
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )


def get_high_value_at_risk(
    rfm_df: pd.DataFrame,
    min_spend: float = 0.0,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns At Risk + Cannot Lose customers above a spend threshold.
    Used by the Monday Morning Brief and win-back action card.
    """
    if "segment" not in rfm_df.columns:
        rfm_df = assign_segments(rfm_df)

    at_risk_segs = ["At Risk", "Cannot Lose"]
    result = (
        rfm_df[
            rfm_df["segment"].isin(at_risk_segs)
            & (rfm_df["total_spend"] >= min_spend)
        ]
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )

    if top_n:
        result = result.head(top_n)

    return result


def segment_color(name: str) -> str:
    return _SEGMENT_COLOR_MAP.get(name, "#6B7E75")


def segment_text_color(name: str) -> str:
    return _SEGMENT_TEXT_MAP.get(name, "#FFFFFF")
