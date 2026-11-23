"""
Dataset Explorer Page
=====================
Allows users to view, filter, search, and download original
and augmented clause datasets.
"""

import streamlit as st
import pandas as pd

from config.settings import CLAUSE_CATEGORIES
from modules.dataset.manager import (
    load_sample_dataset, filter_by_category,
    search_clauses, df_to_csv_bytes, build_combined_dataset
)
from modules.visualization.charts import category_distribution_chart


def render():
    """Render the Dataset Explorer page."""
    st.markdown('<h1 class="page-title">📂 Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Browse, filter, search, and download clause datasets</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load Data ─────────────────────────────────────────────────────────────
    original_df = st.session_state.get("original_df", load_sample_dataset())
    augmented_df = st.session_state.get("augmented_df", pd.DataFrame())

    # ── Dataset Selector Tabs ─────────────────────────────────────────────────
    tab_orig, tab_aug, tab_combined = st.tabs([
        f"📄 Original ({len(original_df)})",
        f"✨ Augmented ({len(augmented_df)})",
        f"📚 Combined ({len(original_df) + len(augmented_df)})",
    ])

    with tab_orig:
        _explore_df(original_df, "original", show_text_col="clause_text")

    with tab_aug:
        if augmented_df.empty:
            st.info("No augmented data yet. Go to **Clause Augmentation** to generate variations.")
        else:
            _explore_df(augmented_df, "augmented", show_text_col="augmented_text")

    with tab_combined:
        combined_df = build_combined_dataset(original_df, augmented_df)
        if combined_df.empty:
            st.info("No combined data available.")
        else:
            _explore_df(combined_df, "combined", show_text_col="clause_text")


def _explore_df(df: pd.DataFrame, ds_name: str, show_text_col: str = "clause_text"):
    """Render explore controls and data table for a dataset."""
    if df is None or df.empty:
        st.info("Dataset is empty.")
        return

    # ── Determine text column ─────────────────────────────────────────────────
    text_col = show_text_col if show_text_col in df.columns else df.columns[0]

    # ── Filter Controls ───────────────────────────────────────────────────────
    col_cat, col_search = st.columns([1, 2])
    with col_cat:
        cat_options = ["All"] + sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"]
        selected_cat = st.selectbox(
            "Filter by Category",
            cat_options,
            key=f"filter_cat_{ds_name}",
        )
    with col_search:
        search_query = st.text_input(
            "🔍 Search clauses",
            placeholder="Type keywords to search...",
            key=f"search_{ds_name}",
        )

    # Apply filters
    filtered = filter_by_category(df, selected_cat)
    if search_query:
        filtered = search_clauses(filtered, search_query)

    # ── Stats Row ─────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Clauses", len(df))
    col_b.metric("Filtered Results", len(filtered))
    if "category" in df.columns:
        col_c.metric("Categories", df["category"].nunique())

    # ── Data Table ────────────────────────────────────────────────────────────
    display_cols = []
    for col in [text_col, "category", "technique", "similarity_score", "source", "is_demo"]:
        if col in filtered.columns:
            display_cols.append(col)
    if not display_cols:
        display_cols = list(filtered.columns)

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

    # ── Category Chart ────────────────────────────────────────────────────────
    if "category" in filtered.columns and not filtered.empty:
        with st.expander("📊 Category Distribution Chart", expanded=False):
            # Rename text column for consistent chart input
            chart_df = filtered.copy()
            if "category" in chart_df.columns:
                fig = category_distribution_chart(chart_df, f"{ds_name.capitalize()} Dataset – Category Distribution")
                st.plotly_chart(fig, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Download")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=f"Download {ds_name.capitalize()} Dataset",
            data=df_to_csv_bytes(df),
            file_name=f"{ds_name}_clauses.csv",
            mime="text/csv",
            key=f"dl_{ds_name}_full",
        )
    with col_dl2:
        if len(filtered) != len(df):
            st.download_button(
                label="Download Filtered Results",
                data=df_to_csv_bytes(filtered),
                file_name=f"{ds_name}_filtered.csv",
                mime="text/csv",
                key=f"dl_{ds_name}_filtered",
            )
