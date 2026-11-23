"""
Visualization Charts
====================
Creates Plotly and Matplotlib charts for the ClauseAugment AI dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

from config.settings import CATEGORY_COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT


# ─────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Inter, Arial, sans-serif"

def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLOR_PRIMARY, family=FONT_FAMILY)),
        font=dict(family=FONT_FAMILY, size=12, color="#2C3E50"),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        margin=dict(l=40, r=40, t=60, b=40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Category Distribution Chart
# ─────────────────────────────────────────────────────────────────────────────

def category_distribution_chart(df: pd.DataFrame, title: str = "Clause Category Distribution") -> go.Figure:
    """Bar chart showing clause count per category."""
    if df is None or df.empty:
        return _empty_figure(title)

    cat_col = "category" if "category" in df.columns else df.columns[0]
    counts = df[cat_col].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    colors = [CATEGORY_COLORS.get(c, COLOR_ACCENT) for c in counts["Category"]]

    fig = go.Figure(go.Bar(
        x=counts["Category"],
        y=counts["Count"],
        marker_color=colors,
        text=counts["Count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title),
        xaxis_title="Category",
        yaxis_title="Number of Clauses",
        showlegend=False,
    )
    return fig


def category_pie_chart(df: pd.DataFrame, title: str = "Dataset Composition") -> go.Figure:
    """Donut pie chart for category distribution."""
    if df is None or df.empty:
        return _empty_figure(title)

    cat_col = "category" if "category" in df.columns else df.columns[0]
    counts = df[cat_col].value_counts()
    colors = [CATEGORY_COLORS.get(c, COLOR_ACCENT) for c in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker_colors=colors,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Original vs Augmented Comparison
# ─────────────────────────────────────────────────────────────────────────────

def original_vs_augmented_bar(original_count: int, augmented_count: int) -> go.Figure:
    """Side-by-side bar comparing original vs augmented clause counts."""
    fig = go.Figure(data=[
        go.Bar(name="Original", x=["Dataset Size"], y=[original_count],
               marker_color=COLOR_PRIMARY, text=[original_count], textposition="outside"),
        go.Bar(name="Augmented", x=["Dataset Size"], y=[augmented_count],
               marker_color=COLOR_ACCENT, text=[augmented_count], textposition="outside"),
    ])
    fig.update_layout(
        **_base_layout("Original vs Augmented Clauses"),
        barmode="group",
        yaxis_title="Number of Clauses",
    )
    return fig


def augmentation_technique_chart(augmented_df: pd.DataFrame) -> go.Figure:
    """Show distribution of augmentation techniques used."""
    if augmented_df is None or augmented_df.empty or "technique" not in augmented_df.columns:
        return _empty_figure("Augmentation Techniques Used")

    counts = augmented_df["technique"].value_counts().reset_index()
    counts.columns = ["Technique", "Count"]
    palette = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, "#1ABC9C", "#8E44AD"]

    fig = go.Figure(go.Bar(
        x=counts["Technique"],
        y=counts["Count"],
        marker_color=palette[:len(counts)],
        text=counts["Count"],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout("Augmentation Techniques Used"),
        xaxis_title="Technique",
        yaxis_title="Count",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Model Performance Charts
# ─────────────────────────────────────────────────────────────────────────────

def metrics_comparison_chart(before: dict, after: dict) -> go.Figure:
    """Grouped bar chart comparing metrics before and after augmentation."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    before_vals = [before.get(m, 0) * 100 for m in metrics]
    after_vals  = [after.get(m, 0) * 100 for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name="Before Augmentation", x=labels, y=before_vals,
               marker_color=COLOR_SECONDARY, text=[f"{v:.1f}%" for v in before_vals],
               textposition="outside"),
        go.Bar(name="After Augmentation", x=labels, y=after_vals,
               marker_color=COLOR_ACCENT, text=[f"{v:.1f}%" for v in after_vals],
               textposition="outside"),
    ])
    fig.update_layout(
        **_base_layout("Model Performance: Before vs After Augmentation"),
        barmode="group",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 115]),
    )
    return fig


def metrics_radar_chart(before: dict, after: dict) -> go.Figure:
    """Radar chart comparing metrics before and after augmentation."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    b_vals = [before.get(m, 0) for m in metrics]
    a_vals = [after.get(m, 0) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=b_vals + [b_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Before Augmentation",
        line_color=COLOR_SECONDARY,
        fillcolor=f"rgba(26,64,128,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=a_vals + [a_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="After Augmentation",
        line_color=COLOR_ACCENT,
        fillcolor="rgba(46,134,193,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=dict(text="Performance Radar", font=dict(size=16, color=COLOR_PRIMARY)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, classes: list, title: str = "Confusion Matrix") -> go.Figure:
    """Interactive confusion-matrix heatmap."""
    if cm is None or len(classes) == 0:
        return _empty_figure(title)

    # Shorten class names for display
    short = [c[:12] for c in classes]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=short,
        y=short,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickangle=-30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Distribution
# ─────────────────────────────────────────────────────────────────────────────

def similarity_histogram(scores: list) -> go.Figure:
    """Histogram of cosine similarity scores."""
    if not scores:
        return _empty_figure("Similarity Score Distribution")

    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color=COLOR_ACCENT,
        opacity=0.8,
        hovertemplate="Similarity: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Augmented Clause Similarity Distribution"),
        xaxis_title="Cosine Similarity Score",
        yaxis_title="Frequency",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Probability Bar Chart (single prediction)
# ─────────────────────────────────────────────────────────────────────────────

def probability_bar_chart(proba_dict: dict) -> go.Figure:
    """Horizontal bar chart of class probabilities for a single prediction."""
    if not proba_dict:
        return _empty_figure("Prediction Probabilities")

    sorted_items = sorted(proba_dict.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]
    colors = [CATEGORY_COLORS.get(l, COLOR_ACCENT) for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout("Classification Probability by Category"),
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 115]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_base_layout(title),
        annotations=[dict(
            text="No data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#999"),
        )],
    )
    return fig
