"""
ML Classifier
=============
Implements TF-IDF vectorisation + multiple classifiers (Logistic Regression,
Naive Bayes, Random Forest) for contract clause classification.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from config.settings import (
    TFIDF_MAX_FEATURES, RANDOM_STATE, TEST_SIZE, CLASSIFIER_OPTIONS
)
from modules.nlp.preprocessor import preprocess_for_ml


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_classifier(name: str):
    """Return an sklearn classifier instance by name."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    elif name == "Naive Bayes":
        return MultinomialNB(alpha=0.5)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def _build_pipeline(classifier_name: str) -> Pipeline:
    """Build a TF-IDF + Classifier sklearn Pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", _build_classifier(classifier_name)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    classifier_name: str = "Logistic Regression",
    text_col: str = "clause_text",
    label_col: str = "category",
) -> dict:
    """
    Train a classifier on the given DataFrame.

    Returns
    -------
    dict with keys:
        - "pipeline": trained sklearn Pipeline
        - "X_test": test texts
        - "y_test": true labels
        - "y_pred": predicted labels
        - "classes": list of class names
        - "error": str or None
    """
    # Determine which column to use for text
    actual_text_col = text_col
    if text_col not in df.columns:
        for alt in ["augmented_text", "clause_text", "text"]:
            if alt in df.columns:
                actual_text_col = alt
                break

    if actual_text_col not in df.columns or label_col not in df.columns:
        return {"pipeline": None, "error": f"Required columns not found. Available: {list(df.columns)}"}

    # Drop rows with missing values
    df_clean = df[[actual_text_col, label_col]].dropna()
    df_clean = df_clean[df_clean[actual_text_col].str.strip() != ""]

    if len(df_clean) < 4:
        return {"pipeline": None, "error": "Not enough data to train (need at least 4 samples)."}

    X = df_clean[actual_text_col].apply(preprocess_for_ml).tolist()
    y = df_clean[label_col].tolist()

    # Stratified split (handle edge cases with small classes)
    unique_classes = list(set(y))
    min_class_count = min(pd.Series(y).value_counts())
    
    # Use stratify only if each class has >= 2 samples
    use_stratify = min_class_count >= 2
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y if use_stratify else None,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    if len(X_train) == 0:
        return {"pipeline": None, "error": "Training set is empty after split."}

    # Build and train pipeline
    try:
        pipeline = _build_pipeline(classifier_name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        return {"pipeline": None, "error": str(e)}

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "classes": sorted(unique_classes),
        "error": None,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_clause(pipeline, text: str) -> dict:
    """
    Predict the category of a single clause text.

    Returns
    -------
    dict with keys: predicted_class, confidence, all_probabilities
    """
    if pipeline is None:
        return {"predicted_class": "Unknown", "confidence": 0.0, "all_probabilities": {}}

    processed = preprocess_for_ml(text)
    try:
        pred = pipeline.predict([processed])[0]

        # Try to get probability scores
        try:
            proba = pipeline.predict_proba([processed])[0]
            classes = pipeline.classes_
            proba_dict = {cls: float(round(p, 4)) for cls, p in zip(classes, proba)}
            confidence = float(max(proba))
        except AttributeError:
            # Random Forest / others may not support predict_proba in all configs
            proba_dict = {pred: 1.0}
            confidence = 1.0

        return {
            "predicted_class": pred,
            "confidence": confidence,
            "all_probabilities": proba_dict,
        }
    except Exception as e:
        return {"predicted_class": "Error", "confidence": 0.0, "all_probabilities": {}, "error": str(e)}


def predict_batch(pipeline, texts: list) -> list:
    """Predict categories for a list of clause texts."""
    return [predict_clause(pipeline, t) for t in texts]
