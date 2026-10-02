"""
utils.py
--------
Shared utility functions for the Fake News Detection System.
Includes chart generators and shared display helpers used by the Streamlit app.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"


def load_eval_results() -> Optional[pd.DataFrame]:
    """Load the model comparison CSV generated during training, or None."""
    csv_path = REPORTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_top_tfidf_terms(
    text: str,
    vectorizer,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    Return the top-N TF-IDF terms for a given piece of text.

    Parameters
    ----------
    text       : cleaned text string
    vectorizer : fitted TfidfVectorizer
    top_n      : number of terms to return

    Returns
    -------
    List of (term, score) tuples sorted descending by score.
    """
    from src.preprocessor import clean_text as _clean

    cleaned = _clean(text)
    if not cleaned.strip():
        return []

    tfidf_matrix = vectorizer.transform([cleaned])
    arr = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    indices = arr.argsort()[::-1][:top_n]
    return [(feature_names[i], round(float(arr[i]), 4)) for i in indices if arr[i] > 0]


def plot_top_tfidf(terms: list[tuple[str, float]]) -> Optional[plt.Figure]:
    """Draw a horizontal bar chart of TF-IDF term scores."""
    if not terms:
        return None
    words, scores = zip(*terms)
    fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.4)))
    ax.barh(list(reversed(words)), list(reversed(scores)), color="#4C72B0")
    ax.set_xlabel("TF-IDF Score", fontsize=11)
    ax.set_title("Top TF-IDF Terms in Submitted Text", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confidence_gauge(confidence: float, prediction: str) -> plt.Figure:
    """
    Draw a simple confidence gauge bar.

    Parameters
    ----------
    confidence : float in [0, 1]
    prediction : 'REAL' | 'FAKE'
    """
    color = "#55A868" if prediction == "REAL" else "#C44E52"
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.barh(["Confidence"], [confidence], color=color, height=0.4)
    ax.barh(["Confidence"], [1 - confidence], left=[confidence], color="#e0e0e0", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 11)])
    ax.set_title(f"Model Confidence: {confidence * 100:.1f}% ({prediction})", fontsize=11)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def format_prediction_result(result: dict) -> str:
    """Return a human-readable string summary of a prediction result."""
    return (
        f"Prediction : {result['prediction']}\n"
        f"Confidence : {result['confidence'] * 100:.2f}%\n"
        f"Model      : {result['model_name']}"
    )
