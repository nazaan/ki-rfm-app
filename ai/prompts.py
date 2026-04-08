"""
ai/prompts.py
-------------
All prompt templates in one place.

Rules:
  - System prompts define the AI's role and output format strictly.
  - User prompts inject the actual data.
  - No business logic here — just string construction.
  - All functions return (system_prompt, user_prompt) tuples.
"""

import pandas as pd
from calculations.stats import fmt_currency, fmt_pct, fmt_number


# ── Monday Morning Brief ───────────────────────────────────────────────────

def monday_morning_brief_prompt(
    stats: dict,
    segment_summary: pd.DataFrame,
    health: dict,
    top_at_risk: pd.DataFrame,
) -> tuple[str, str]:
    """
    Generates the main actionable brief.
    Tone: direct, concrete, Monday morning — no fluff.
    """

    system = """You are a concise, data-driven customer retention analyst.
Your job is to write a short Monday Morning Brief for a business owner.

Rules:
- Maximum 250 words.
- No preamble. Start directly with the insight.
- Use plain English. No jargon.
- Be specific: use the actual numbers provided.
- Give exactly 3 actions, each on its own line starting with ▶
- Each action must include: who to target, what to do, and when.
- End with one sentence on the biggest risk if nothing is done.
- Do not invent data. Only use the numbers given."""

    # Build at-risk summary
    if len(top_at_risk) > 0:
        at_risk_lines = "\n".join([
            f"  - {row['customer_name'] or row['customer_id']}: "
            f"${row['total_spend']:,.0f} lifetime spend, "
            f"{int(row['days_since'])} days since last purchase"
            for _, row in top_at_risk.head(5).iterrows()
        ])
    else:
        at_risk_lines = "  None identified."

    # Build segment snapshot — only non-zero segments
    active_segs = segment_summary[segment_summary["count"] > 0]
    seg_lines = "\n".join([
        f"  {row['segment']}: {row['count']} customers "
        f"({row['pct_customers']}% of base, "
        f"avg spend {fmt_currency(row['avg_spend'])})"
        for _, row in active_segs.iterrows()
    ])

    user = f"""Here is today's customer data snapshot:

HEALTH SCORE: {health['score']}/100 — {health['verdict']}

KEY METRICS:
  Total customers:      {fmt_number(stats['total_customers'])}
  Total revenue:        {fmt_currency(stats['total_revenue'])}
  Avg order value:      {fmt_currency(stats['avg_order_value'])}
  Repeat purchase rate: {fmt_pct(stats['repeat_purchase_rate'])}
  Churn rate ({stats['churn_days_threshold']} days): {fmt_pct(stats['churn_rate'])}
  Top 5% revenue share: {fmt_pct(stats['top5_revenue_pct'])}

SEGMENT BREAKDOWN:
{seg_lines}

HIGHEST VALUE AT-RISK CUSTOMERS:
{at_risk_lines}

Write the Monday Morning Brief now."""

    return system, user


# ── Customer persona writer ────────────────────────────────────────────────

def segment_persona_prompt(
    segment_name: str,
    segment_row: pd.Series,
    sample_customers: pd.DataFrame,
) -> tuple[str, str]:
    """
    Writes a short marketing persona for a segment.
    Agency tier feature.
    """

    system = """You are a sharp marketing strategist who writes customer personas.
Write a persona for the given customer segment in exactly this format:

**Persona Name:** [Give them a memorable name like "The Deal Seeker" or "The Brand Loyalist"]
**Who They Are:** [2 sentences describing their buying behaviour based on the data]
**What They Want:** [1 sentence on their motivation]
**What To Do:** [2 concrete marketing actions]
**What NOT To Do:** [1 thing that will push them away]

Keep it under 150 words. Be specific. Use the data provided."""

    # Sample customer names for context
    if len(sample_customers) > 0 and "customer_name" in sample_customers.columns:
        names = sample_customers["customer_name"].head(3).tolist()
        names = [n for n in names if n and str(n).strip()]
        sample_str = f"Example customers: {', '.join(names)}" if names else ""
    else:
        sample_str = ""

    user = f"""Segment: {segment_name}

DATA:
  Customers in segment:  {int(segment_row.get('count', 0))}
  Avg RFM score:         {segment_row.get('avg_rfm', 0):.1f} / 15
  Avg recency score:     {segment_row.get('avg_r', 0):.1f} / 5
  Avg frequency score:   {segment_row.get('avg_f', 0):.1f} / 5
  Avg monetary score:    {segment_row.get('avg_m', 0):.1f} / 5
  Avg spend per customer:{fmt_currency(segment_row.get('avg_spend', 0))}
  % of total revenue:    {fmt_pct(segment_row.get('pct_revenue', 0))}
{sample_str}

Write the persona now."""

    return system, user


# ── Win-back email subject lines ───────────────────────────────────────────

def winback_subject_lines_prompt(
    segment_name: str,
    avg_days_since: float,
    avg_spend: float,
    count: int,
) -> tuple[str, str]:
    """
    Generates 5 win-back email subject lines for a segment.
    Practical, copywriter-quality output.
    """

    system = """You are an email marketing copywriter specialising in customer retention.
Generate exactly 5 email subject lines for a win-back campaign.

Rules:
- Each subject line on its own numbered line.
- Mix of emotional, curiosity, and direct approaches.
- Keep each under 50 characters where possible.
- No clickbait. No emojis unless it genuinely helps.
- Make them feel personal, not mass-email.
- Do not add any other text — just the 5 subject lines."""

    user = f"""Write 5 win-back email subject lines for this segment:

Segment:              {segment_name}
Customers:            {count}
Avg days since last purchase: {int(avg_days_since)} days
Avg customer spend:   {fmt_currency(avg_spend)}

These customers were once engaged buyers who haven't returned.
The email goal is to get them to make one more purchase."""

    return system, user


# ── Churn risk explanation ─────────────────────────────────────────────────

def churn_explanation_prompt(
    stats: dict,
    segment_summary: pd.DataFrame,
) -> tuple[str, str]:
    """
    Plain-English explanation of why customers are churning,
    based on the data pattern.
    """

    system = """You are a customer analytics expert.
Analyse the data and give a plain-English diagnosis of the churn situation.

Format:
**Diagnosis:** [1-2 sentences on what the data pattern suggests]
**Likely Cause:** [1-2 sentences on probable business reasons]
**Priority Fix:** [1 concrete action to reduce churn this week]

Under 100 words. Be direct. Only use the data provided."""

    active = segment_summary[segment_summary["count"] > 0]
    seg_lines = "\n".join([
        f"  {r['segment']}: {r['count']} ({r['pct_customers']}%)"
        for _, r in active.iterrows()
    ])

    user = f"""Customer base data:

  Total customers:     {fmt_number(stats['total_customers'])}
  Churn rate:          {fmt_pct(stats['churn_rate'])} (>{stats['churn_days_threshold']} days inactive)
  Repeat purchase rate:{fmt_pct(stats['repeat_purchase_rate'])}
  Avg days since buy:  {int(stats['avg_days_since'])} days
  Avg order value:     {fmt_currency(stats['avg_order_value'])}

Segment distribution:
{seg_lines}

Diagnose the churn situation."""

    return system, user
