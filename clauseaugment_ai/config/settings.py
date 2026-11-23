"""
ClauseAugment AI – Application Configuration
=============================================
Central settings and constants used across the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
APP_MODE = os.getenv("APP_MODE", "production")  # "demo" or "production"

# Determine whether to use demo mode
IS_DEMO_MODE = (not GEMINI_API_KEY) or (GEMINI_API_KEY == "your_gemini_api_key_here") or (APP_MODE == "demo")

# Gemini model to use
GEMINI_MODEL = "gemini-1.5-flash"

# ─────────────────────────────────────────────
# Application Constants
# ─────────────────────────────────────────────
APP_TITLE = "ClauseAugment AI"
APP_SUBTITLE = "Contract Clause Text Augmenter"
APP_VERSION = "1.0.0"

# ─────────────────────────────────────────────
# Clause Categories
# ─────────────────────────────────────────────
CLAUSE_CATEGORIES = [
    "Confidentiality",
    "Termination",
    "Indemnification",
    "Payment Terms",
    "Liability",
    "Intellectual Property",
    "Governing Law",
    "Data Protection",
]

# ─────────────────────────────────────────────
# Augmentation Techniques
# ─────────────────────────────────────────────
AUGMENTATION_TECHNIQUES = [
    "Legal Paraphrasing",
    "Sentence Restructuring",
    "Vocabulary Variation",
    "Formal Language Variation",
    "Structural Variation",
]

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATASET_PATH = DATA_DIR / "sample_clauses.csv"

# ─────────────────────────────────────────────
# ML Configuration
# ─────────────────────────────────────────────
CLASSIFIER_OPTIONS = ["Logistic Regression", "Naive Bayes", "Random Forest"]
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─────────────────────────────────────────────
# UI Colors (Navy Blue LegalTech Theme)
# ─────────────────────────────────────────────
COLOR_PRIMARY = "#0A2558"       # Navy blue
COLOR_SECONDARY = "#1A4080"     # Medium blue
COLOR_ACCENT = "#2E86C1"        # Bright blue accent
COLOR_SUCCESS = "#1ABC9C"       # Teal green
COLOR_WARNING = "#F39C12"       # Amber
COLOR_DANGER = "#E74C3C"        # Red
COLOR_LIGHT = "#F4F6F9"         # Light grey background
COLOR_WHITE = "#FFFFFF"
COLOR_TEXT = "#2C3E50"          # Dark text
COLOR_MUTED = "#7F8C8D"         # Muted grey

CATEGORY_COLORS = {
    "Confidentiality":    "#2E86C1",
    "Termination":        "#E74C3C",
    "Indemnification":    "#8E44AD",
    "Payment Terms":      "#27AE60",
    "Liability":          "#F39C12",
    "Intellectual Property": "#16A085",
    "Governing Law":      "#2C3E50",
    "Data Protection":    "#D35400",
}
