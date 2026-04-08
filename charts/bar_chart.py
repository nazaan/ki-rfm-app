"""
charts/bar_chart.py
-------------------
Horizontal bar chart: customer count per segment.
"""

import plotly.graph_objects as go
import pandas as pd


def build_bar_chart(segment_summary: pd.DataFrame) -> go.Figure:
    df = segment_summary[segment_summary["count"] > 0].copy()
    if df.empty:
        return _empty_figure("No segment data to display.")

    df = df.iloc[::-1].reset_index(drop=True)

    hover = [
        f"<b>{row['segment']}</b><br>"
        f"Customers: {row['count']}<br>"
        f"Share: {row['pct_customers']}%<br>"
        f"Avg Spend: ${row['avg_spend']:,.2f}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["count"],
        y=df["segment"],
        orientation="h",
        marker_color=df["color"].tolist(),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"  {c}  ({p}%)" for c, p in zip(df["count"], df["pct_customers"])],
        textposition="outside",
        textfont=dict(size=11, color="#1A1A2E"),
    ))

    fig.update_layout(
        **_base_layout(),
        title=dict(text="Customers by Segment", font=dict(size=15, color="#1A1A2E"), x=0),
        xaxis=dict(title="Number of Customers", showgrid=True, gridcolor="#EEF2F0", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=50, b=40),
        height=380,
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
        showlegend=False,
        hoverlabel=dict(bgcolor="#1A1A2E", font_size=12, font_color="#FFFFFF"),
    )
