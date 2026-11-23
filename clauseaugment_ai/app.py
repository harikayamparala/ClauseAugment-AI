"""
ClauseAugment AI – Main Application
=====================================
Entry point for the Streamlit application.
Handles page routing, theme injection, and sidebar navigation.

Run with:
    streamlit run app.py
"""

import sys
import os

# ── Ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page Configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="ClauseAugment AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ClauseAugment AI – Contract Clause Text Augmenter v1.0.0",
    },
)

# ── Import page modules ────────────────────────────────────────────────────────
from views import home, augmentation, classification, dataset_explorer, evaluation, about
from config.settings import APP_TITLE, APP_SUBTITLE, IS_DEMO_MODE


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – Navy Blue LegalTech Theme
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        /* ── Google Font ─────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global ──────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Main background ─────────────────────── */
        .stApp {
            background-color: #F4F6F9;
        }

        /* ── Sidebar ─────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A2558 0%, #1A4080 60%, #0D3B8E 100%);
            border-right: none;
        }
        section[data-testid="stSidebar"] * {
            color: #E8EDF5 !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #E8EDF5 !important;
            font-weight: 500;
            padding: 4px 0;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }
        /* Radio button selected highlight */
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            background-color: #2E86C1 !important;
        }

        /* ── Page Titles ─────────────────────────── */
        .page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0A2558;
            margin-bottom: 4px;
        }
        .page-subtitle {
            font-size: 1rem;
            color: #5D6D7E;
            margin-top: 0;
            margin-bottom: 12px;
        }

        /* ── Metric Cards ────────────────────────── */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px 18px;
            box-shadow: 0 2px 12px rgba(10,37,88,0.08);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(10,37,88,0.14);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #5D6D7E;
            margin-top: 6px;
            font-weight: 500;
        }

        /* ── Clause Boxes ────────────────────────── */
        .clause-box {
            background: #FAFBFD;
            border-radius: 10px;
            padding: 16px;
            font-size: 0.9rem;
            line-height: 1.65;
            color: #2C3E50;
            border: 1.5px solid #DDE3EE;
            min-height: 80px;
        }
        .original-clause {
            border-left: 4px solid #0A2558;
        }
        .augmented-clause {
            border-left: 4px solid #2E86C1;
        }

        /* ── Buttons ─────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0A2558, #2E86C1);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 24px;
            transition: opacity 0.2s, transform 0.1s;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.92;
            transform: translateY(-1px);
        }

        /* ── Tabs ────────────────────────────────── */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            color: #0A2558;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #2E86C1;
        }

        /* ── Expander ────────────────────────────── */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #0A2558;
        }

        /* ── DataFrames ──────────────────────────── */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 1px 8px rgba(10,37,88,0.06);
        }

        /* ── Metric deltas ───────────────────────── */
        [data-testid="stMetricDelta"] > div {
            font-weight: 600;
        }

        /* ── Sidebar logo area ───────────────────── */
        .sidebar-logo {
            text-align: center;
            padding: 24px 0 12px 0;
        }
        .sidebar-logo h2 {
            font-size: 1.3rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
            letter-spacing: 0.5px;
        }
        .sidebar-logo p {
            font-size: 0.75rem;
            color: #A8BBDA;
            margin: 2px 0 0 0;
        }
        .sidebar-divider {
            border-top: 1px solid rgba(255,255,255,0.15);
            margin: 12px 0;
        }

        /* ── Info/warning boxes ──────────────────── */
        .stAlert {
            border-radius: 10px;
        }

        /* ── Progress bar ────────────────────────── */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #0A2558, #2E86C1);
        }

        /* ── Scrollbar ───────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #F4F6F9; }
        ::-webkit-scrollbar-thumb { background: #2E86C1; border-radius: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Home Dashboard":       home,
    "✨ Clause Augmentation":  augmentation,
    "🤖 Clause Classification": classification,
    "📂 Dataset Explorer":     dataset_explorer,
    "📈 Model Evaluation":     evaluation,
    "ℹ️ About Project":        about,
}


def render_sidebar():
    with st.sidebar:
        # Logo / Branding
        st.markdown(
            """
            <div class="sidebar-logo">
                <div style="font-size:2.5rem;">⚖️</div>
                <h2>ClauseAugment AI</h2>
                <p>Contract Clause Text Augmenter</p>
            </div>
            <div class="sidebar-divider"></div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown("#### Navigation")
        selected = st.radio(
            "Go to",
            list(PAGES.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Mode indicator
        if IS_DEMO_MODE:
            st.markdown(
                '<div style="background:rgba(255,193,7,0.2);border-radius:8px;padding:10px;'
                'border:1px solid rgba(255,193,7,0.4);font-size:0.8rem;color:#FFD700;">'
                '🔵 <b>Demo Mode</b><br>'
                '<span style="font-size:0.75rem;color:#A8BBDA;">Add GEMINI_API_KEY<br>to .env for live AI</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(26,188,156,0.2);border-radius:8px;padding:10px;'
                'border:1px solid rgba(26,188,156,0.4);font-size:0.8rem;color:#1ABC9C;">'
                '🟢 <b>Live Mode</b><br>'
                '<span style="font-size:0.75rem;color:#A8BBDA;">Gemini API Active</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Quick stats from session
        orig_df = st.session_state.get("original_df")
        aug_df  = st.session_state.get("augmented_df")
        orig_count = len(orig_df) if orig_df is not None else 0
        aug_count  = len(aug_df)  if aug_df  is not None and not aug_df.empty else 0

        st.markdown(
            f"""
            <div style="font-size:0.8rem;color:#A8BBDA;padding:4px 0;">
                📄 Original: <b style="color:#E8EDF5;">{orig_count}</b> clauses<br>
                ✨ Augmented: <b style="color:#E8EDF5;">{aug_count}</b> clauses
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Initialise session state defaults
    if "original_df" not in st.session_state:
        try:
            from modules.dataset.manager import load_sample_dataset
            st.session_state["original_df"] = load_sample_dataset()
        except Exception:
            import pandas as pd
            st.session_state["original_df"] = pd.DataFrame()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Render the selected page
    page_module = PAGES.get(selected_page, home)
    try:
        page_module.render()
    except Exception as e:
        st.error(f"❌ Error rendering page: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
