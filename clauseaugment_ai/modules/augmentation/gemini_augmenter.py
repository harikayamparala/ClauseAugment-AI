"""
Gemini Augmenter
================
Integrates Google Gemini API to generate augmented contract clause variations.
Uses the new google-genai SDK (google.genai).
Falls back to demo mode when no valid API key is configured.
"""

import time
import re
import warnings

from config.settings import (
    GEMINI_API_KEY, IS_DEMO_MODE, GEMINI_MODEL,
    AUGMENTATION_TECHNIQUES,
)
from modules.augmentation.augmentation_techniques import (
    TECHNIQUE_PROMPTS, get_demo_variations
)

# ── Attempt to import new Gemini SDK (google-genai) ──────────────────────────
_gemini_available = False
_genai_client = None

try:
    # Try new SDK first (google-genai package)
    from google import genai as _genai_new
    from google.genai import types as _genai_types

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        _genai_client = _genai_new.Client(api_key=GEMINI_API_KEY)
        _SDK = "new"
        _gemini_available = True
except ImportError:
    pass

# ── Fallback: try legacy google-generativeai SDK ──────────────────────────────
if not _gemini_available:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as _genai_legacy
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            _genai_legacy.configure(api_key=GEMINI_API_KEY)
            _SDK = "legacy"
            _gemini_available = True
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Core Augmentation Function
# ─────────────────────────────────────────────────────────────────────────────

def augment_clause(
    clause_text: str,
    category: str,
    num_samples: int = 5,
    techniques: list = None,
) -> dict:
    """
    Generate augmented variations of a contract clause.

    Returns
    -------
    dict with keys:
        - "variations": list of dicts {technique, augmented_text, is_demo}
        - "is_demo": bool
        - "error": str or None
    """
    num_samples = max(1, min(10, num_samples))
    if techniques is None:
        techniques = AUGMENTATION_TECHNIQUES

    # ── Demo mode ────────────────────────────────────────────────────────────
    if IS_DEMO_MODE or not _gemini_available:
        demo_vars = get_demo_variations(category, num_samples)
        return {"variations": demo_vars, "is_demo": True, "error": None}

    # ── Live Gemini mode ─────────────────────────────────────────────────────
    variations = []
    errors = []

    for i in range(num_samples):
        technique = techniques[i % len(techniques)]
        prompt_template = TECHNIQUE_PROMPTS.get(technique, TECHNIQUE_PROMPTS["Legal Paraphrasing"])
        prompt = prompt_template.format(clause=clause_text, category=category)

        try:
            generated_text = _call_gemini(prompt)

            if generated_text:
                variations.append({
                    "technique": technique,
                    "augmented_text": generated_text,
                    "is_demo": False,
                })
            else:
                errors.append(f"Empty response for technique '{technique}'")

            # Polite rate limiting
            time.sleep(0.5)

        except Exception as e:
            err_msg = str(e)
            errors.append(f"{technique}: {err_msg}")

            # If quota / auth error, fall back to demo for remaining
            if any(k in err_msg.lower() for k in ["quota", "api_key", "permission", "unauthorized", "429"]):
                remaining = num_samples - len(variations)
                demo_vars = get_demo_variations(category, remaining)
                for dv in demo_vars:
                    dv["is_demo"] = True
                    variations.append(dv)
                break

    # If nothing was generated, fall back to demo
    if not variations:
        demo_vars = get_demo_variations(category, num_samples)
        return {
            "variations": demo_vars,
            "is_demo": True,
            "error": "; ".join(errors) if errors else None,
        }

    return {
        "variations": variations,
        "is_demo": False,
        "error": "; ".join(errors) if errors else None,
    }


def _call_gemini(prompt: str) -> str:
    """Call the appropriate Gemini SDK and return generated text."""
    if _SDK == "new" and _genai_client is not None:
        # New SDK
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=512,
                top_p=0.9,
            ),
        )
        return _clean_response(response.text)

    elif _SDK == "legacy":
        # Legacy SDK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _genai_legacy.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
        return _clean_response(response.text)

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Single-technique generation
# ─────────────────────────────────────────────────────────────────────────────

def augment_single(clause_text: str, category: str, technique: str) -> dict:
    """Generate one variation using a specific technique."""
    result = augment_clause(clause_text, category, num_samples=1, techniques=[technique])
    if result["variations"]:
        v = result["variations"][0]
        return {
            "augmented_text": v["augmented_text"],
            "is_demo": v.get("is_demo", result["is_demo"]),
            "error": result["error"],
        }
    return {"augmented_text": "", "is_demo": True, "error": "No variation generated"}


# ─────────────────────────────────────────────────────────────────────────────
# Batch augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_batch(rows: list, num_samples_per_clause: int = 3) -> list:
    """Augment a batch of clause dicts. Each row needs 'clause_text' and 'category'."""
    augmented_records = []
    for row in rows:
        clause_text = row.get("clause_text", "")
        category = row.get("category", "Confidentiality")
        result = augment_clause(clause_text, category, num_samples=num_samples_per_clause)
        for v in result["variations"]:
            augmented_records.append({
                "original_clause": clause_text,
                "augmented_text": v["augmented_text"],
                "category": category,
                "technique": v["technique"],
                "is_demo": v.get("is_demo", result["is_demo"]),
            })
    return augmented_records


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_response(text: str) -> str:
    """Strip common LLM preamble / labels from generated text."""
    if not text:
        return ""
    prefixes_to_strip = [
        r"^(rewritten|restructured|revised|alternative|varied)\s+(clause|draft)\s*:\s*",
        r"^\*\*[^*]+\*\*\s*",
        r"^#+\s*",
        r'^"', r'"$',
    ]
    text = text.strip()
    for pattern in prefixes_to_strip:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def get_mode_label() -> str:
    """Return a human-readable string indicating the current mode."""
    if IS_DEMO_MODE or not _gemini_available:
        return "🔵 Demo Mode (No API Key)"
    sdk_label = "New SDK" if _SDK == "new" else "Legacy SDK"
    return f"🟢 Live Mode (Gemini API – {sdk_label})"
