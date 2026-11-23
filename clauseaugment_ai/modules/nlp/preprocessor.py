"""
NLP Preprocessor
================
Handles text cleaning, tokenisation, and similarity scoring using
standard NLTK and scikit-learn utilities (no heavy model downloads).
"""

import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Download NLTK resources on first use ────────────────────────────────────
def _download_nltk():
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

_download_nltk()

# Initialise NLP objects
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = set()


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Strip leading/trailing whitespace
    - Normalise unicode
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_for_ml(text: str, lemmatize: bool = True) -> str:
    """
    Full NLP preprocessing pipeline for ML feature extraction:
    1. Lowercase
    2. Remove punctuation
    3. Tokenise
    4. Remove stopwords
    5. Lemmatize (or stem)

    Returns a space-joined string of processed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation except hyphens (legal terms often hyphenated)
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Tokenise
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]

    # Lemmatize or stem
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF character
    n-gram representation.  Returns a float in [0.0, 1.0].
    """
    if not text_a or not text_b:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(round(score, 4))
    except Exception:
        return 0.0


def compute_similarity_batch(original: str, variants: list) -> list:
    """
    Compute cosine similarity between one original text and a list of variants.
    Returns a list of float scores.
    """
    return [compute_cosine_similarity(original, v) for v in variants]


# ─────────────────────────────────────────────────────────────────────────────
# Text Statistics
# ─────────────────────────────────────────────────────────────────────────────

def text_stats(text: str) -> dict:
    """
    Return basic statistics about a clause text.
    """
    if not isinstance(text, str) or not text.strip():
        return {"words": 0, "sentences": 0, "chars": 0}

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    words = text.split()
    return {
        "words": len(words),
        "sentences": len(sentences),
        "chars": len(text),
    }
