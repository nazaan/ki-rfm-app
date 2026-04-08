"""
charts/scatter_chart.py
-----------------------
Scatter chart: Avg Frequency vs Avg Spend per segment.
Bubble size = number of customers in segment.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def build_scatter_chart(segment_summary: pd.DataFrame) -> go.Figure:
    df = segment_summary[segment_summary["count"] > 0].copy()
    if df.empty:
        return _empty_figure("No segment data to display.")

    counts = df["count"].values.astype(float)
    min_size, max_size = 14, 55
    if counts.max() == counts.min():
        sizes = [30] * len(counts)
    else:
        sizes = ((counts - counts.min()) / (counts.max() - counts.min()) * (max_size - min_size) + min_size).tolist()

    fig = go.Figure()

    for i, (_, row) in enumerate(df.iterrows()):
        hover = (
            f"<b>{row['segment']}</b><br>"
            f"Avg Orders: {row['avg_f']:.1f}<br>"
            f"Avg Spend: ${row['avg_spend']:,.2f}<br>"
            f"Customers: {row['count']}<br>"
            f"Avg RFM: {row['avg_rfm']:.1f}"
        )
        fig.add_trace(go.Scatter(
            x=[row["avg_f"]],
            y=[row["avg_spend"]],
            mode="markers+text",
            name=row["segment"],
            marker=dict(size=sizes[i], color=row["color"], line=dict(color="#FFFFFF", width=1.5), opacity=0.9),
            text=[row["segment"]],
            textposition="top center" if row["avg_f"] <= df["avg_f"].median() else "bottom center",
            textfont=dict(size=9, color="#1A1A2E"),
            hovertemplate=f"{hover}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(),
        title=dict(text="Avg Frequency vs Avg Spend by Segment", font=dict(size=15, color="#1A1A2E"), x=0),
        xaxis=dict(title="Avg Number of Orders (Frequency)", showgrid=True, gridcolor="#EEF2F0", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Avg Total Spend ($)", showgrid=True, gridcolor="#EEF2F0", zeroline=False, tickfont=dict(size=10), tickprefix="$"),
        legend=dict(orientation="v", x=1.02, y=1.0, font=dict(size=9), itemsizing="constant"),
        margin=dict(l=60, r=140, t=50, b=50),
        height=420,
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
