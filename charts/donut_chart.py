"""
charts/donut_chart.py
---------------------
Donut chart: revenue share by segment.
"""

import plotly.graph_objects as go
import pandas as pd


def build_donut_chart(segment_summary: pd.DataFrame) -> go.Figure:
    df = segment_summary[segment_summary["total_revenue"] > 0].copy()
    if df.empty:
        return _empty_figure("No revenue data to display.")

    hover = [
        f"<b>{row['segment']}</b><br>"
        f"Revenue: ${row['total_revenue']:,.2f}<br>"
        f"Share: {row['pct_revenue']}%<br>"
        f"Customers: {row['count']}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=df["segment"].tolist(),
        values=df["total_revenue"].tolist(),
        hole=0.55,
        marker=dict(colors=df["color"].tolist(), line=dict(color="#FFFFFF", width=2)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        textinfo="label+percent",
        textfont=dict(size=10),
        insidetextorientation="radial",
        texttemplate=[
            f"{row['segment']}<br>{row['pct_revenue']}%" if row["pct_revenue"] >= 5 else ""
            for _, row in df.iterrows()
        ],
    ))

    total_rev = df["total_revenue"].sum()
    if total_rev >= 1_000_000:
        rev_label = f"${total_rev/1_000_000:.1f}M"
    elif total_rev >= 1_000:
        rev_label = f"${total_rev/1_000:.1f}K"
    else:
        rev_label = f"${total_rev:,.0f}"

    fig.add_annotation(
        text=f"<b>{rev_label}</b><br><span style='font-size:10px'>Total Revenue</span>",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#1A1A2E"), align="center",
    )

    fig.update_layout(
        **_base_layout(),
        title=dict(text="Revenue Share by Segment", font=dict(size=15, color="#1A1A2E"), x=0),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10), itemsizing="constant"),
        margin=dict(l=10, r=120, t=50, b=20),
        height=380,
        showlegend=True,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#6B7E75"))
    fig.update_layout(**_base_layout(), height=300)
    return fig


def _base_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#1A1A2E"),
        hoverlabel=dict(bgcolor="#1A1A2E", font_size=12, font_color="#FFFFFF"),
    )
