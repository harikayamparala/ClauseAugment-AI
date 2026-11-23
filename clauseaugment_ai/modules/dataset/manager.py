"""
Dataset Manager
===============
Handles loading, saving, filtering, and downloading of clause datasets.
"""

import io
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import SAMPLE_DATASET_PATH, CLAUSE_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# Load / Save
# ─────────────────────────────────────────────────────────────────────────────

def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled sample clause dataset."""
    try:
        df = pd.read_csv(SAMPLE_DATASET_PATH)
        df = _normalise_columns(df)
        return df
    except Exception as e:
        # Return an empty skeleton if file missing
        return _empty_df()


def load_uploaded_csv(uploaded_file) -> tuple:
    """
    Parse a user-uploaded CSV file.

    Returns
    -------
    (df, error_message)
    df is None if parsing failed.
    """
    try:
        df = pd.read_csv(uploaded_file)
        df = _normalise_columns(df)
        # Validate required columns
        if "clause_text" not in df.columns:
            return None, "CSV must contain a 'clause_text' column."
        if "category" not in df.columns:
            df["category"] = "Confidentiality"  # default
        return df, None
    except Exception as e:
        return None, f"Failed to parse CSV: {e}"


def save_augmented_dataset(augmented_rows: list, original_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert a list of augmented row dicts into a DataFrame.
    Optionally merges with the original dataset.
    """
    if not augmented_rows:
        return pd.DataFrame()

    aug_df = pd.DataFrame(augmented_rows)

    # Ensure expected columns exist
    for col in ["original_clause", "augmented_text", "category", "technique", "is_demo"]:
        if col not in aug_df.columns:
            aug_df[col] = ""

    return aug_df


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Return rows matching the given category. 'All' returns all rows."""
    if category == "All" or not category:
        return df
    return df[df["category"] == category].reset_index(drop=True)


def search_clauses(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Full-text search across text columns."""
    if not query or not isinstance(query, str):
        return df
    query_lower = query.lower()
    mask = df.apply(
        lambda row: any(
            query_lower in str(val).lower()
            for val in row.values
        ),
        axis=1,
    )
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 encoded CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def build_combined_dataset(original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stack original and augmented data into one combined DataFrame.
    Adds a 'source' column to distinguish them.
    """
    orig = original_df.copy()
    orig["source"] = "original"
    if "clause_text" not in orig.columns and "clause_text" in orig.columns:
        orig = orig.rename(columns={"clause_text": "clause_text"})

    aug = augmented_df.copy()
    aug["source"] = "augmented"
    if "augmented_text" in aug.columns and "clause_text" not in aug.columns:
        aug = aug.rename(columns={"augmented_text": "clause_text"})

    combined = pd.concat([orig, aug], ignore_index=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def dataset_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for a dataset."""
    if df is None or df.empty:
        return {"total": 0, "categories": {}, "num_categories": 0}

    cat_col = "category" if "category" in df.columns else None
    if cat_col:
        cats = df[cat_col].value_counts().to_dict()
    else:
        cats = {}

    return {
        "total": len(df),
        "categories": cats,
        "num_categories": len(cats),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names (strip whitespace, lowercase)."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["clause_id", "clause_text", "category", "augmented_clause", "source"])
