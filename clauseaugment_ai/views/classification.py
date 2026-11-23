"""
Clause Classification Page
==========================
Train ML models on original and augmented datasets,
then classify new clauses and display prediction results.
"""

import streamlit as st
import pandas as pd

from config.settings import CLAUSE_CATEGORIES, CLASSIFIER_OPTIONS
from modules.classification.classifier import train_classifier, predict_clause
from modules.dataset.manager import load_sample_dataset
from modules.visualization.charts import probability_bar_chart
from modules.nlp.preprocessor import text_stats


def render():
    """Render the Classification page."""
    st.markdown('<h1 class="page-title">🤖 Clause Classification</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Train ML models and classify contract clauses by category</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Training Section ──────────────────────────────────────────────────────
    _training_section()

    st.markdown("---")

    # ── Prediction Section ────────────────────────────────────────────────────
    _prediction_section()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _training_section():
    st.markdown("### 🎓 Train Classifier")

    col_settings, col_info = st.columns([2, 3])

    with col_settings:
        st.markdown("#### Training Settings")
        classifier_name = st.selectbox(
            "Select Algorithm",
            CLASSIFIER_OPTIONS,
            key="clf_algorithm",
        )
        dataset_choice = st.radio(
            "Training Dataset",
            ["Original Dataset", "Augmented Dataset", "Combined Dataset"],
            key="clf_dataset_choice",
        )
        st.markdown(
            """
            > **TF-IDF + classifier pipeline**  
            > Features: unigrams + bigrams  
            > Max features: 5,000  
            > Split: 80% train / 20% test
            """
        )

    with col_info:
        st.markdown("#### Dataset Preview")
        original_df = st.session_state.get("original_df", load_sample_dataset())
        augmented_df = st.session_state.get("augmented_df", pd.DataFrame())

        if dataset_choice == "Original Dataset":
            train_df = original_df.copy()
            if "clause_text" not in train_df.columns:
                train_df = train_df.rename(columns={train_df.columns[1]: "clause_text"})
        elif dataset_choice == "Augmented Dataset":
            if augmented_df.empty:
                st.warning("⚠️ No augmented data yet. Please generate augmentations first.")
                train_df = original_df.copy()
            else:
                aug_copy = augmented_df.copy()
                aug_copy = aug_copy.rename(columns={"augmented_text": "clause_text"})
                train_df = aug_copy[["clause_text", "category"]]
        else:  # Combined
            orig_copy = original_df.copy()
            if "clause_text" not in orig_copy.columns:
                orig_copy = orig_copy.rename(columns={orig_copy.columns[1]: "clause_text"})
            orig_copy = orig_copy[["clause_text", "category"]]

            if not augmented_df.empty:
                aug_copy = augmented_df.copy()
                aug_copy = aug_copy.rename(columns={"augmented_text": "clause_text"})
                aug_copy = aug_copy[["clause_text", "category"]]
                train_df = pd.concat([orig_copy, aug_copy], ignore_index=True)
            else:
                train_df = orig_copy

        st.dataframe(
            train_df[["clause_text", "category"]].head(8) if "clause_text" in train_df.columns
            else train_df.head(8),
            use_container_width=True,
        )
        st.caption(f"Total rows available: **{len(train_df)}**")

    # Train button
    if st.button("🚀 Train Classifier", type="primary", use_container_width=True, key="btn_train"):
        with st.spinner(f"Training {classifier_name}..."):
            result = train_classifier(
                df=train_df,
                classifier_name=classifier_name,
                text_col="clause_text",
                label_col="category",
            )

        if result.get("error"):
            st.error(f"❌ Training failed: {result['error']}")
        else:
            # Store trained pipeline in session
            session_key = f"pipeline_{dataset_choice.replace(' ', '_').lower()}"
            st.session_state[session_key] = result["pipeline"]
            st.session_state["last_train_result"] = result
            st.session_state["last_train_dataset"] = dataset_choice
            st.session_state["last_clf_name"] = classifier_name

            st.success(
                f"✅ **{classifier_name}** trained on **{dataset_choice}**! "
                f"({result['train_size']} train / {result['test_size']} test)"
            )

            # Quick accuracy display
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(result["y_test"], result["y_pred"])
            st.metric("Test Accuracy", f"{acc:.2%}")

    # Show training result if available
    if "last_train_result" in st.session_state and st.session_state["last_train_result"].get("pipeline"):
        r = st.session_state["last_train_result"]
        with st.expander("📊 View Training Results", expanded=False):
            from sklearn.metrics import classification_report
            report = classification_report(
                r["y_test"], r["y_pred"],
                labels=r["classes"],
                zero_division=0,
            )
            st.text(report)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def _prediction_section():
    st.markdown("### 🔍 Classify a Clause")

    # Find available trained pipelines
    available_pipelines = {}
    for ds in ["Original_Dataset", "Augmented_Dataset", "Combined_Dataset"]:
        key = f"pipeline_{ds.lower()}"
        if key in st.session_state and st.session_state[key] is not None:
            available_pipelines[ds.replace("_", " ")] = st.session_state[key]

    if not available_pipelines:
        st.info("💡 Train a classifier above first to enable predictions.")
        # Still show the input area, but disable the button
        pipeline_to_use = None
        pipeline_label = "None"
    else:
        pipeline_label = st.selectbox(
            "Use pipeline trained on:",
            list(available_pipelines.keys()),
            key="pred_pipeline_choice",
        )
        pipeline_to_use = available_pipelines.get(pipeline_label)

    col_pred, col_result = st.columns([3, 2])

    with col_pred:
        clause_to_classify = st.text_area(
            "Enter clause to classify:",
            height=140,
            placeholder="Paste a contract clause here to predict its category...",
            key="pred_clause_input",
        )
        if clause_to_classify.strip():
            stats = text_stats(clause_to_classify)
            st.caption(f"📊 {stats['words']} words | {stats['sentences']} sentences")

    btn_disabled = pipeline_to_use is None
    if st.button("🔍 Classify Clause", type="primary", disabled=btn_disabled, key="btn_classify"):
        if not clause_to_classify.strip():
            st.error("Please enter a clause to classify.")
        elif pipeline_to_use is None:
            st.error("No trained pipeline available. Please train a classifier first.")
        else:
            pred_result = predict_clause(pipeline_to_use, clause_to_classify.strip())

            with col_result:
                _display_prediction(pred_result, pipeline_label)

            # Show probability chart below
            if pred_result.get("all_probabilities"):
                st.markdown("#### 📊 Classification Probabilities")
                fig = probability_bar_chart(pred_result["all_probabilities"])
                st.plotly_chart(fig, use_container_width=True)


def _display_prediction(pred: dict, pipeline_label: str):
    """Show prediction result as a styled card."""
    predicted = pred.get("predicted_class", "Unknown")
    confidence = pred.get("confidence", 0.0)

    conf_color = "#1ABC9C" if confidence > 0.7 else "#F39C12" if confidence > 0.4 else "#E74C3C"

    st.markdown(
        f"""
        <div style="background:#F4F6F9;border-radius:12px;padding:20px;border-left:5px solid #0A2558;">
            <p style="color:#666;margin:0;font-size:0.85em;">Pipeline: {pipeline_label}</p>
            <h3 style="color:#0A2558;margin:8px 0;">📌 {predicted}</h3>
            <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
                <span style="background:{conf_color};color:white;padding:4px 12px;
                             border-radius:12px;font-size:0.85em;font-weight:600;">
                    Confidence: {confidence:.1%}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
