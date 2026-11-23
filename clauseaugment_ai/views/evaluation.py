"""
Model Evaluation Page
=====================
Compare ML model performance before and after augmentation
with accuracy, precision, recall, F1-score, and confusion matrix.
"""

import streamlit as st
import pandas as pd

from config.settings import CLASSIFIER_OPTIONS
from modules.classification.classifier import train_classifier
from modules.evaluation.metrics import compute_metrics, compare_metrics, metrics_to_df
from modules.dataset.manager import load_sample_dataset
from modules.visualization.charts import (
    metrics_comparison_chart, metrics_radar_chart, confusion_matrix_heatmap
)


def render():
    """Render the Model Evaluation page."""
    st.markdown('<h1 class="page-title">📈 Model Evaluation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Compare classifier performance before and after augmentation</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        """
        > 🎯 **Purpose**: This page demonstrates whether augmenting the training dataset
        > improves classification robustness. Both models are trained and evaluated on
        > real data — results are **not fabricated**.
        """
    )

    # ── Settings ──────────────────────────────────────────────────────────────
    col_set, _ = st.columns([2, 3])
    with col_set:
        classifier_name = st.selectbox(
            "Classifier Algorithm",
            CLASSIFIER_OPTIONS,
            key="eval_clf_name",
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    original_df = st.session_state.get("original_df", load_sample_dataset())
    augmented_df = st.session_state.get("augmented_df", pd.DataFrame())

    # Prepare text columns
    orig_train = _prepare_df(original_df, "clause_text")
    if augmented_df.empty:
        aug_train = None
    else:
        aug_train = _prepare_df(augmented_df, "augmented_text")

    has_aug = aug_train is not None and not aug_train.empty

    # ── Run Evaluation ────────────────────────────────────────────────────────
    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Training and evaluating models..."):
            # Train on original data
            before_result = train_classifier(
                df=orig_train,
                classifier_name=classifier_name,
                text_col="clause_text",
                label_col="category",
            )

            if before_result.get("error"):
                st.error(f"❌ Failed to train on original data: {before_result['error']}")
                return

            before_metrics = compute_metrics(
                before_result["y_test"],
                before_result["y_pred"],
                before_result["classes"],
            )

            # Train on augmented or combined data
            if has_aug:
                # Combined = original + augmented
                combined = pd.concat([orig_train, aug_train], ignore_index=True)
                after_result = train_classifier(
                    df=combined,
                    classifier_name=classifier_name,
                    text_col="clause_text",
                    label_col="category",
                )
                after_label = "Combined (Original + Augmented)"
            else:
                # If no augmented data, evaluate on original again with note
                after_result = before_result
                after_label = "Original (no augmented data)"
                st.warning("⚠️ No augmented data available. Showing results on original dataset only.")

            if after_result.get("error"):
                st.error(f"❌ Failed to train on augmented data: {after_result['error']}")
                return

            after_metrics = compute_metrics(
                after_result["y_test"],
                after_result["y_pred"],
                after_result["classes"],
            )

            # Store in session
            st.session_state["eval_before"] = before_metrics
            st.session_state["eval_after"] = after_metrics
            st.session_state["eval_before_result"] = before_result
            st.session_state["eval_after_result"] = after_result
            st.session_state["eval_clf_label"] = classifier_name
            st.session_state["eval_after_label"] = after_label

        st.success("✅ Evaluation complete!")

    # ── Display Results ───────────────────────────────────────────────────────
    if "eval_before" in st.session_state:
        _display_evaluation_results()


# ─────────────────────────────────────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _display_evaluation_results():
    before = st.session_state["eval_before"]
    after  = st.session_state["eval_after"]
    clf    = st.session_state.get("eval_clf_label", "Classifier")
    after_label = st.session_state.get("eval_after_label", "After Augmentation")

    st.markdown("---")
    st.markdown(f"### 📊 Results: **{clf}**")

    # ── Metric Cards ──────────────────────────────────────────────────────────
    comparison = compare_metrics(before, after)
    metrics_order = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    col1, col2, col3, col4 = st.columns(4)
    for col, m, label in zip([col1, col2, col3, col4], metrics_order, labels):
        c = comparison[m]
        delta_str = f"{c['delta']:+.2%}"
        with col:
            st.metric(
                label=label,
                value=f"{c['after']:.2%}",
                delta=delta_str,
                help=f"Before: {c['before']:.2%} → After: {c['after']:.2%}",
            )

    st.markdown("---")

    # ── Comparison Charts ─────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_bar = metrics_comparison_chart(before, after)
        st.plotly_chart(fig_bar, use_container_width=True)
    with chart_col2:
        fig_radar = metrics_radar_chart(before, after)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("### 🧮 Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**Before Augmentation (Original Data)**")
        fig_cm_b = confusion_matrix_heatmap(
            before["confusion_matrix"],
            before["classes"],
            "Before – Confusion Matrix",
        )
        st.plotly_chart(fig_cm_b, use_container_width=True)

    with cm_col2:
        st.markdown(f"**After Augmentation ({after_label})**")
        fig_cm_a = confusion_matrix_heatmap(
            after["confusion_matrix"],
            after["classes"],
            "After – Confusion Matrix",
        )
        st.plotly_chart(fig_cm_a, use_container_width=True)

    # ── Classification Reports ────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Classification Reports")
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        st.markdown("**Before Augmentation**")
        st.code(before.get("classification_report", "N/A"), language="text")
    with rep_col2:
        st.markdown(f"**After Augmentation ({after_label})**")
        st.code(after.get("classification_report", "N/A"), language="text")

    # ── Metric Table ──────────────────────────────────────────────────────────
    with st.expander("📊 Metrics Table", expanded=False):
        tbl_before = metrics_to_df(before).rename(columns={"Score": "Before"})
        tbl_after = metrics_to_df(after).rename(columns={"Score": "After"})
        merged = tbl_before.merge(tbl_after, on="Metric")
        st.dataframe(merged.set_index("Metric"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation Helper
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Return a clean df with 'clause_text' and 'category' columns."""
    df = df.copy()
    if text_col in df.columns:
        df = df.rename(columns={text_col: "clause_text"})
    if "clause_text" not in df.columns:
        # Try first text-like column
        for c in df.columns:
            if df[c].dtype == object and c != "category":
                df = df.rename(columns={c: "clause_text"})
                break
    return df[["clause_text", "category"]].dropna()
