"""
About Page
==========
Project overview, architecture, team information, and usage instructions.
"""

import streamlit as st
from config.settings import APP_VERSION, CLAUSE_CATEGORIES, AUGMENTATION_TECHNIQUES


def render():
    """Render the About Project page."""
    st.markdown('<h1 class="page-title">ℹ️ About ClauseAugment AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">A Generative AI-powered contract clause text augmentation system</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Project Overview ──────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ## 🎯 Project Overview

            **ClauseAugment AI** is a Generative AI-powered system designed to improve
            the robustness of machine learning models that classify contract clauses.

            ### The Problem
            Contract clause classification models trained on limited datasets often
            fail to generalise across different legal phrasing, writing styles, and
            contract templates.

            ### Our Solution
            Use Google Gemini API to generate diverse, legally-faithful paraphrases
            of existing contract clauses. These augmented samples expand the training
            dataset, exposing classifiers to greater linguistic variety without
            altering legal meaning.

            ### Key Benefits
            - 📈 Improved model generalisation through data augmentation
            - ⚖️ Legal meaning preserved in all generated variations
            - 🔄 Five distinct augmentation techniques
            - 🤖 Three ML classifiers with real performance comparison
            - 📊 End-to-end evaluation pipeline
            """
        )

    with col2:
        st.markdown("### 🏷️ Supported Categories")
        for cat in CLAUSE_CATEGORIES:
            st.markdown(f"- {cat}")

        st.markdown("### 🔧 Augmentation Techniques")
        for tech in AUGMENTATION_TECHNIQUES:
            st.markdown(f"- {tech}")

    st.markdown("---")

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown("## 🏗️ System Architecture")

    st.markdown(
        """
        ```
        ClauseAugment AI
        ├── Frontend Layer (Streamlit)
        │   ├── Home Dashboard
        │   ├── Clause Augmentation
        │   ├── Clause Classification
        │   ├── Dataset Explorer
        │   └── Model Evaluation
        │
        ├── AI Layer (Google Gemini API)
        │   ├── Legal Paraphrasing
        │   ├── Sentence Restructuring
        │   ├── Vocabulary Variation
        │   ├── Formal Language Variation
        │   └── Structural Variation
        │
        ├── NLP Layer (NLTK)
        │   ├── Text Preprocessing
        │   ├── Tokenisation & Lemmatisation
        │   └── Cosine Similarity Scoring
        │
        ├── ML Layer (Scikit-learn)
        │   ├── TF-IDF Vectorisation
        │   ├── Logistic Regression
        │   ├── Naive Bayes
        │   └── Random Forest
        │
        └── Data Layer (Pandas / CSV)
            ├── Sample Dataset (50 clauses)
            ├── Augmented Dataset
            └── Combined Dataset
        ```
        """
    )

    st.markdown("---")

    # ── Technology Stack ──────────────────────────────────────────────────────
    st.markdown("## 🛠️ Technology Stack")

    tech_data = {
        "Component": [
            "Frontend", "Backend", "Generative AI",
            "Machine Learning", "NLP", "Data Processing", "Visualisation"
        ],
        "Technology": [
            "Streamlit", "Python 3.10+", "Google Gemini API (gemini-1.5-flash)",
            "Scikit-learn (LR, NB, RF)", "NLTK", "Pandas + NumPy", "Plotly + Matplotlib"
        ],
    }
    import pandas as pd
    st.table(pd.DataFrame(tech_data).set_index("Component"))

    st.markdown("---")

    # ── Usage Instructions ────────────────────────────────────────────────────
    st.markdown("## 🚀 Getting Started")

    with st.expander("📦 Installation", expanded=True):
        st.code(
            """
# 1. Clone or navigate to the project directory
cd clauseaugment_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key (optional — demo mode works without it)
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here

# 4. Run the application
streamlit run app.py
            """,
            language="bash",
        )

    with st.expander("🔑 Gemini API Key Setup"):
        st.markdown(
            """
            1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a new API key
            3. Copy the key to your `.env` file:
               ```
               GEMINI_API_KEY=AIza...your_key_here
               ```
            4. Restart the application

            > **Note**: Without an API key, the app runs in **Demo Mode** with
            > pre-written example variations clearly labelled as such.
            """
        )

    with st.expander("📋 Workflow Guide"):
        st.markdown(
            """
            **Step 1 – Augment Clauses**
            - Go to *Clause Augmentation*
            - Enter a clause manually or upload a CSV
            - Select a category and number of variations
            - Click *Generate Augmented Clauses*

            **Step 2 – Classify Clauses**
            - Go to *Clause Classification*
            - Train a model on original / augmented / combined data
            - Enter any clause to get its predicted category

            **Step 3 – Evaluate Performance**
            - Go to *Model Evaluation*
            - Click *Run Evaluation* to compare before vs after augmentation
            - View confusion matrices, charts, and classification reports

            **Step 4 – Explore Datasets**
            - Go to *Dataset Explorer*
            - Filter, search, and download datasets
            """
        )

    st.markdown("---")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.warning(
        """
        ⚖️ **Legal Disclaimer**

        All AI-generated contract clause variations are for **demonstration and research
        purposes only**. They do not constitute legal advice. Always have any contractual
        language reviewed by a qualified legal professional before use in actual contracts.
        """,
        icon="⚠️",
    )

    # ── Version ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"ClauseAugment AI v{APP_VERSION} | "
        "Built with ❤️ using Python, Streamlit, and Google Gemini API | "
        "For educational and portfolio purposes."
    )
