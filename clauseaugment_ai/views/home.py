"""
Home Dashboard Page
===================
Displays overview metrics, dataset distribution charts, and system status.
"""

import streamlit as st
import pandas as pd

from config.settings import (
    CLAUSE_CATEGORIES, AUGMENTATION_TECHNIQUES, APP_VERSION
)
from modules.augmentation.gemini_augmenter import get_mode_label
from modules.dataset.manager import load_sample_dataset, dataset_stats
from modules.visualization.charts import (
    category_distribution_chart, category_pie_chart,
    original_vs_augmented_bar, augmentation_technique_chart
)


def render():
    """Render the Home Dashboard page."""
    st.markdown('<h1 class="page-title">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Overview of your contract clause augmentation system</p>',
        unsafe_allow_html=True,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    original_df = st.session_state.get("original_df", load_sample_dataset())
    augmented_df = st.session_state.get("augmented_df", pd.DataFrame())

    # Persist original_df in session
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = original_df

    orig_stats = dataset_stats(original_df)
    aug_stats  = dataset_stats(augmented_df)

    # ── API / Mode Banner ────────────────────────────────────────────────────
    mode_label = get_mode_label()
    if "Demo" in mode_label:
        st.info(
            f"**{mode_label}** — Add your `GEMINI_API_KEY` to a `.env` file to enable live AI generation.",
            icon="🔵",
        )
    else:
        st.success(f"**{mode_label}** — Gemini API is active and ready.", icon="🟢")

    st.markdown("---")

    # ── Metric Cards ─────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        _metric_card("📄 Original Clauses", orig_stats["total"], "#0A2558")
    with col2:
        _metric_card("✨ Augmented Clauses", aug_stats["total"], "#2E86C1")
    with col3:
        _metric_card("🏷️ Categories", len(CLAUSE_CATEGORIES), "#1ABC9C")
    with col4:
        _metric_card("🔧 Techniques", len(AUGMENTATION_TECHNIQUES), "#8E44AD")
    with col5:
        total = orig_stats["total"] + aug_stats["total"]
        _metric_card("📚 Total Dataset", total, "#F39C12")

    st.markdown("---")

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📈 Category Distribution (Original)")
        fig = category_distribution_chart(original_df, "Original Clauses by Category")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🍩 Dataset Composition")
        fig2 = category_pie_chart(original_df, "Clause Category Share")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ⚖️ Original vs Augmented")
        fig3 = original_vs_augmented_bar(orig_stats["total"], aug_stats["total"])
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("#### 🔧 Augmentation Techniques Used")
        if not augmented_df.empty:
            fig4 = augmentation_technique_chart(augmented_df)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No augmented data yet. Go to **Clause Augmentation** to generate variations.")

    # ── Quick Stats Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Category Breakdown")
    if orig_stats["categories"]:
        breakdown = pd.DataFrame(
            [(cat, cnt) for cat, cnt in orig_stats["categories"].items()],
            columns=["Category", "Original Count"],
        )
        if aug_stats["categories"]:
            aug_map = aug_stats["categories"]
            breakdown["Augmented Count"] = breakdown["Category"].map(aug_map).fillna(0).astype(int)
        else:
            breakdown["Augmented Count"] = 0

        breakdown["Total"] = breakdown["Original Count"] + breakdown["Augmented Count"]
        st.dataframe(breakdown.set_index("Category"), use_container_width=True)

    # ── Footer / Version ─────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"ClauseAugment AI v{APP_VERSION} | Built with Streamlit + Google Gemini API")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value, color: str):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
