"""
Clause Augmentation Page
========================
Allows users to input a clause (manually or via CSV upload),
select a category, and generate AI-powered augmented variations.
"""

import streamlit as st
import pandas as pd

from config.settings import CLAUSE_CATEGORIES, AUGMENTATION_TECHNIQUES
from modules.augmentation.gemini_augmenter import augment_clause, get_mode_label
from modules.dataset.manager import (
    load_uploaded_csv, save_augmented_dataset, df_to_csv_bytes
)
from modules.nlp.preprocessor import compute_cosine_similarity, text_stats
from modules.visualization.charts import similarity_histogram


def render():
    """Render the Clause Augmentation page."""
    st.markdown('<h1 class="page-title">✨ Clause Augmentation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Generate AI-powered variations of your contract clauses</p>',
        unsafe_allow_html=True,
    )

    # ── Mode Banner ───────────────────────────────────────────────────────────
    mode = get_mode_label()
    if "Demo" in mode:
        st.warning(
            f"**{mode}** – Results shown are pre-written demo examples. "
            "Set `GEMINI_API_KEY` in `.env` to enable live generation.",
            icon="⚠️",
        )

    st.markdown("---")

    # ── Input Section ─────────────────────────────────────────────────────────
    input_tab, csv_tab = st.tabs(["✍️ Manual Input", "📂 CSV Upload"])

    with input_tab:
        _manual_input_section()

    with csv_tab:
        _csv_upload_section()


# ─────────────────────────────────────────────────────────────────────────────
# Manual Input Section
# ─────────────────────────────────────────────────────────────────────────────

def _manual_input_section():
    col_input, col_settings = st.columns([3, 1])

    with col_settings:
        st.markdown("#### ⚙️ Settings")
        category = st.selectbox(
            "Clause Category",
            CLAUSE_CATEGORIES,
            key="aug_category",
        )
        num_samples = st.slider(
            "Number of Variations",
            min_value=1, max_value=10, value=5,
            key="aug_num_samples",
        )
        selected_techniques = st.multiselect(
            "Augmentation Techniques",
            AUGMENTATION_TECHNIQUES,
            default=AUGMENTATION_TECHNIQUES,
            key="aug_techniques",
        )
        if not selected_techniques:
            selected_techniques = AUGMENTATION_TECHNIQUES

    with col_input:
        st.markdown("#### 📝 Original Clause")
        # Provide example clauses to get started
        example_map = {
            "Confidentiality": "Each party agrees to keep confidential all information disclosed by the other party and shall not disclose such information to any third party without prior written consent.",
            "Termination": "Either party may terminate this Agreement by providing thirty (30) days written notice to the other party.",
            "Indemnification": "Each party shall indemnify, defend, and hold harmless the other party from any claims, damages, or liabilities arising out of its breach of this Agreement.",
            "Payment Terms": "Payment of all invoices shall be due within thirty (30) days of the invoice date unless otherwise agreed in writing by the parties.",
            "Liability": "In no event shall either party be liable for any indirect, incidental, special, or consequential damages arising out of or related to this Agreement.",
            "Intellectual Property": "All intellectual property rights in the work product created by the Service Provider under this Agreement shall vest exclusively in the Client upon full payment.",
            "Governing Law": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law provisions.",
            "Data Protection": "The parties shall comply with all applicable data protection laws, including the General Data Protection Regulation (GDPR) and any national implementing legislation.",
        }
        example_text = example_map.get(category, "")

        clause_text = st.text_area(
            "Enter your contract clause below:",
            value=st.session_state.get("manual_clause", example_text),
            height=150,
            placeholder="Paste or type your contract clause here...",
            key="aug_clause_text",
        )

        # Show text statistics
        if clause_text.strip():
            stats = text_stats(clause_text)
            st.caption(f"📊 {stats['words']} words | {stats['sentences']} sentences | {stats['chars']} characters")

    # ── Augment Button ────────────────────────────────────────────────────────
    if st.button("🚀 Generate Augmented Clauses", type="primary", use_container_width=True):
        if not clause_text.strip():
            st.error("Please enter a contract clause before generating variations.")
            return

        with st.spinner("Generating augmented clause variations..."):
            result = augment_clause(
                clause_text=clause_text.strip(),
                category=category,
                num_samples=num_samples,
                techniques=selected_techniques,
            )

        _display_augmentation_results(clause_text.strip(), category, result)

    # ── Show previous results if still in session ─────────────────────────────
    elif "last_augmentation" in st.session_state:
        r = st.session_state["last_augmentation"]
        _display_augmentation_results(r["original"], r["category"], r["result"])


def _display_augmentation_results(original_text: str, category: str, result: dict):
    """Display augmentation results with comparison view."""
    variations = result.get("variations", [])
    is_demo = result.get("is_demo", False)

    # Save to session for persistence
    st.session_state["last_augmentation"] = {
        "original": original_text,
        "category": category,
        "result": result,
    }

    if not variations:
        st.error("No variations were generated. Please try again.")
        return

    st.markdown("---")
    # Demo mode notice
    if is_demo:
        st.info(
            "🔵 **Demo Mode** – The variations below are pre-written examples, "
            "not generated by the live Gemini API.",
            icon="ℹ️",
        )

    st.markdown(f"### 📋 Results: {len(variations)} Variation(s) Generated")
    st.markdown(
        f"**Category:** `{category}` | **Human Review Required** ⚠️ — Always verify AI-generated clauses with a legal professional."
    )

    # ── Original Clause Box ───────────────────────────────────────────────────
    with st.expander("📄 Original Clause", expanded=True):
        st.markdown(
            f'<div class="clause-box original-clause">{original_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Side-by-side comparison ───────────────────────────────────────────────
    st.markdown("#### 🔄 Augmented Variations")

    similarity_scores = []
    rows_for_save = []

    for i, v in enumerate(variations):
        technique = v.get("technique", f"Variation {i+1}")
        aug_text = v.get("augmented_text", "")
        sim_score = compute_cosine_similarity(original_text, aug_text)
        similarity_scores.append(sim_score)

        with st.expander(f"**{i+1}. {technique}** | Similarity: {sim_score:.2%}", expanded=(i == 0)):
            left, right = st.columns(2)
            with left:
                st.markdown("**Original**")
                st.markdown(
                    f'<div class="clause-box original-clause">{original_text}</div>',
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown("**Augmented**")
                st.markdown(
                    f'<div class="clause-box augmented-clause">{aug_text}</div>',
                    unsafe_allow_html=True,
                )

            # Similarity badge
            sim_color = "#1ABC9C" if sim_score > 0.5 else "#F39C12" if sim_score > 0.3 else "#E74C3C"
            st.markdown(
                f'<span style="background:{sim_color};color:white;padding:4px 10px;border-radius:12px;font-size:0.8em;">'
                f'Similarity: {sim_score:.2%}</span>',
                unsafe_allow_html=True,
            )

        rows_for_save.append({
            "original_clause": original_text,
            "augmented_text": aug_text,
            "category": category,
            "technique": technique,
            "similarity_score": round(sim_score, 4),
            "is_demo": is_demo,
        })

    # ── Similarity Distribution ───────────────────────────────────────────────
    if len(similarity_scores) > 1:
        st.markdown("#### 📊 Similarity Distribution")
        fig = similarity_histogram(similarity_scores)
        st.plotly_chart(fig, use_container_width=True)

    # ── Save to session ───────────────────────────────────────────────────────
    _update_augmented_session(rows_for_save)

    # ── Download ──────────────────────────────────────────────────────────────
    aug_df = pd.DataFrame(rows_for_save)
    csv_bytes = df_to_csv_bytes(aug_df)
    st.download_button(
        label="⬇️ Download These Variations as CSV",
        data=csv_bytes,
        file_name=f"augmented_{category.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV Upload Section
# ─────────────────────────────────────────────────────────────────────────────

def _csv_upload_section():
    st.markdown("#### 📂 Upload CSV File")
    st.markdown(
        "Upload a CSV file with at least a `clause_text` column. "
        "Optionally include a `category` column."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")

    if uploaded:
        df, error = load_uploaded_csv(uploaded)
        if error:
            st.error(f"❌ {error}")
            return

        st.success(f"✅ Loaded {len(df)} rows from CSV.")
        st.dataframe(df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            default_cat = st.selectbox(
                "Default Category (if not in CSV)",
                CLAUSE_CATEGORIES,
                key="csv_default_cat",
            )
        with col_b:
            samples_per = st.slider(
                "Variations per Clause",
                min_value=1, max_value=5, value=3,
                key="csv_samples_per",
            )

        max_clauses = st.slider(
            "Max clauses to augment (from uploaded file)",
            min_value=1, max_value=min(len(df), 20), value=min(5, len(df)),
            key="csv_max_clauses",
        )

        if st.button("🚀 Augment Uploaded Clauses", type="primary"):
            rows_subset = df.head(max_clauses)
            all_aug_rows = []

            progress = st.progress(0, text="Augmenting clauses...")
            for idx, (_, row) in enumerate(rows_subset.iterrows()):
                clause_text = str(row.get("clause_text", row.iloc[0]))
                category = str(row.get("category", default_cat))
                if category not in CLAUSE_CATEGORIES:
                    category = default_cat

                result = augment_clause(clause_text, category, num_samples=samples_per)
                for v in result["variations"]:
                    all_aug_rows.append({
                        "original_clause": clause_text,
                        "augmented_text": v["augmented_text"],
                        "category": category,
                        "technique": v["technique"],
                        "similarity_score": round(
                            compute_cosine_similarity(clause_text, v["augmented_text"]), 4
                        ),
                        "is_demo": v.get("is_demo", result["is_demo"]),
                    })

                progress.progress((idx + 1) / max_clauses, text=f"Processed {idx+1}/{max_clauses} clauses...")

            progress.empty()
            st.success(f"✅ Generated {len(all_aug_rows)} augmented variations!")

            aug_df = pd.DataFrame(all_aug_rows)
            st.dataframe(aug_df, use_container_width=True)

            _update_augmented_session(all_aug_rows)

            csv_bytes = df_to_csv_bytes(aug_df)
            st.download_button(
                label="⬇️ Download All Augmented Clauses",
                data=csv_bytes,
                file_name="augmented_clauses_batch.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Session State Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_augmented_session(new_rows: list):
    """Append new augmented rows to the session-level augmented_df."""
    import pandas as pd
    new_df = pd.DataFrame(new_rows)
    existing = st.session_state.get("augmented_df", pd.DataFrame())
    if existing.empty:
        st.session_state["augmented_df"] = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["augmented_text"]).reset_index(drop=True)
        st.session_state["augmented_df"] = combined
