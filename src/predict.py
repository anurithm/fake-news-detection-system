"""
predict.py
----------
Prediction system for the Fake News Detection System.

Loads the saved TF-IDF vectoriser and best model,
preprocesses the given text, and returns a structured result.

Usage (as a module):
    from src.predict import predict_text
    result = predict_text("Your news article here.")
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"


# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loading cache
# ──────────────────────────────────────────────────────────────────────────────

_cache: dict = {}


def _load_assets() -> tuple:
    """
    Load (and cache) the TF-IDF vectoriser, model, and model info.

    Returns
    -------
    (vectorizer, model, model_info_dict)

    Raises
    ------
    FileNotFoundError – if model files are missing (training not yet run).
    """
    if _cache:
        return _cache["vectorizer"], _cache["model"], _cache["info"]

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectoriser not found at: {VECTORIZER_PATH}\n"
            "Please run 'python src/train.py' first to train the models."
        )
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found at: {BEST_MODEL_PATH}\n"
            "Please run 'python src/train.py' first to train the models."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    model_info: dict = {}
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            model_info = json.load(f)

    _cache["vectorizer"] = vectorizer
    _cache["model"] = model
    _cache["info"] = model_info

    logger.info("Loaded model: %s", model_info.get("best_model_name", "unknown"))
    return vectorizer, model, model_info


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def predict_text(
    raw_text: str,
    *,
    use_lemmatization: bool = True,
) -> dict:
    """
    Preprocess, vectorise, and classify a news article.

    Parameters
    ----------
    raw_text : str
        The raw news article or headline submitted by the user.
    use_lemmatization : bool
        Should match the setting used during training (default True).

    Returns
    -------
    dict with keys:
        prediction  : 'REAL' | 'FAKE'
        confidence  : float  (0.0 – 1.0)
        model_name  : str
        label_probs : dict   {'REAL': float, 'FAKE': float}
    """
    from src.preprocessor import clean_text

    if not raw_text or not raw_text.strip():
        raise ValueError("Input text is empty. Please provide a news article or headline.")

    vectorizer, model, model_info = _load_assets()
    model_name: str = model_info.get("best_model_name", "Unknown Model")

    # Preprocess
    cleaned = clean_text(raw_text, use_lemmatization=use_lemmatization)
    if not cleaned.strip():
        raise ValueError(
            "The text became empty after preprocessing. Please provide a longer, meaningful article."
        )

    # Vectorise
    X = vectorizer.transform([cleaned])

    # Predict
    raw_pred = model.predict(X)[0]
    prediction: str = str(raw_pred)

    # Confidence — use predict_proba when available (all our models support it)
    label_probs: dict[str, float] = {}
    confidence: float = 0.5

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes: list[str] = list(model.classes_)
        label_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        # Confidence = probability of the predicted class
        confidence = label_probs.get(prediction, max(proba))
    else:
        # Fallback: decision_function scaled to [0,1] via sigmoid
        if hasattr(model, "decision_function"):
            score = model.decision_function(X)[0]
            confidence = float(1 / (1 + np.exp(-score)))
        label_probs = {
            prediction: confidence,
            ("REAL" if prediction == "FAKE" else "FAKE"): round(1 - confidence, 4),
        }

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "model_name": model_name,
        "label_probs": label_probs,
    }


def models_ready() -> bool:
    """Return True if trained models and vectoriser exist on disk."""
    return VECTORIZER_PATH.exists() and BEST_MODEL_PATH.exists()


def get_model_info() -> Optional[dict]:
    """Return the model_info dict or None if not trained yet."""
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    sample = (
        "Scientists have discovered a new species of dinosaur in Argentina "
        "that lived 90 million years ago. The fossil was found remarkably preserved."
    )
    result = predict_text(sample)
    print(f"\nPrediction  : {result['prediction']}")
    print(f"Confidence  : {result['confidence'] * 100:.2f}%")
    print(f"Model       : {result['model_name']}")
    print(f"Probabilities: {result['label_probs']}")
