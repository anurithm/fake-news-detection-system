"""
train.py
--------
End-to-end training pipeline for the Fake News Detection System.

Usage:
    python src/train.py [--dataset PATH] [--no-lemmatize]

Steps:
    1. Load and validate CSV dataset
    2. Preprocess text
    3. Train/test split (stratified)
    4. Fit TF-IDF on training data only (no leakage)
    5. Train Logistic Regression, Naive Bayes, Linear SVM, Random Forest
    6. Evaluate all models
    7. Select best model (by F1)
    8. Save models, vectoriser, and evaluation results
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DEFAULT_DATASET = DATA_DIR / "dataset.csv"
SAMPLE_DATASET = DATA_DIR / "sample_data.csv"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
EVAL_RESULTS_PATH = REPORTS_DIR / "model_comparison.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ensure directories ─────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a dict mapping model display-name to estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42, C=1.0)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained model and return a metrics dict."""
    y_pred = model.predict(X_test)
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="FAKE", average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="FAKE", average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label="FAKE", average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"]).tolist(),
    }


def save_comparison_chart(results: list[dict]) -> None:
    """Save a bar chart comparing model metrics."""
    df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1_score"]]
    df = df.set_index("model")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df.index))
    width = 0.2
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize(), color=color)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved model comparison chart.")


def save_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    """Save a confusion matrix heatmap for the given model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "best_model_confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix for %s.", model_name)


def save_class_distribution(df: pd.DataFrame) -> None:
    """Save a bar chart of the class distribution in the dataset."""
    dist = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#55A868", "#C44E52"]
    dist.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Class Distribution in Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xticklabels(dist.index, rotation=0)
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved class distribution chart.")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_training(dataset_path: Path, use_lemmatization: bool = True) -> None:
    """Execute the full training pipeline."""

    # Import here to avoid circular-import issues
    from src.dataset_loader import load_dataset, dataset_summary
    from src.preprocessor import clean_text

    # ── 1. Load & validate ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading dataset from %s", dataset_path)
    df = load_dataset(dataset_path)

    summary = dataset_summary(df)
    logger.info(
        "\nDataset Summary:\n"
        "  Total samples  : %d\n"
        "  REAL           : %d\n"
        "  FAKE           : %d\n"
        "  Avg text length: %d words",
        summary["total_samples"],
        summary["real_count"],
        summary["fake_count"],
        summary["avg_text_length"],
    )

    # ── 2. Save class distribution chart ────────────────────────────────────
    save_class_distribution(df)

    # ── 3. Preprocess text ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing text (lemmatization=%s) …", use_lemmatization)
    df["clean_text"] = df["combined_text"].apply(
        lambda t: clean_text(t, use_lemmatization=use_lemmatization)
    )
    logger.info("Text preprocessing complete.")

    # ── 4. Train / test split (stratified) ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Splitting dataset (80/20 stratified) …")
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── 5. TF-IDF vectorisation (fit on train ONLY) ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Fitting TF-IDF vectoriser on training data …")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    logger.info("Vocabulary size: %d", len(vectorizer.vocabulary_))

    # ── 6. Train & evaluate all models ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Training and evaluating models …")
    models = build_models()
    results: list[dict] = []
    trained_models: dict = {}

    for name, model in models.items():
        logger.info("  Training: %s …", name)
        model.fit(X_train_tfidf, y_train)
        metrics = evaluate_model(model, X_test_tfidf, y_test, name)
        results.append(metrics)
        trained_models[name] = model
        logger.info(
            "    Accuracy: %.4f  |  F1: %.4f", metrics["accuracy"], metrics["f1_score"]
        )

    # ── 7. Select best model by F1 ───────────────────────────────────────────
    best = max(results, key=lambda r: r["f1_score"])
    best_name = best["model"]
    best_model = trained_models[best_name]
    logger.info("=" * 60)
    logger.info("Best model: %s  (F1=%.4f)", best_name, best["f1_score"])

    # ── 8. Save artefacts ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Saving artefacts …")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, BEST_MODEL_PATH)

    # Persist all individual models too
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")

    model_info = {
        "best_model_name": best_name,
        "best_model_f1": best["f1_score"],
        # Store only the basename so no personal filesystem paths are baked in
        "dataset_name": Path(dataset_path).name or Path(dataset_path).parent.name,
        "total_samples": summary["total_samples"],
        "real_count": summary["real_count"],
        "fake_count": summary["fake_count"],
        "vocab_size": len(vectorizer.vocabulary_),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    # Save evaluation CSV
    eval_df = pd.DataFrame(results).drop(columns=["confusion_matrix"])
    eval_df.to_csv(EVAL_RESULTS_PATH, index=False)

    # Save charts
    save_comparison_chart(results)
    save_confusion_matrix(best_model, X_test_tfidf, y_test, best_name)

    # ── 9. Print summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — Summary:")
    logger.info("")
    header = f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        marker = " ◀ BEST" if r["model"] == best_name else ""
        logger.info(
            "%-30s %10.4f %10.4f %10.4f %10.4f%s",
            r["model"], r["accuracy"], r["precision"], r["recall"], r["f1_score"], marker
        )
    logger.info("")
    logger.info("Models saved to  : %s", MODELS_DIR)
    logger.info("Reports saved to : %s", REPORTS_DIR)
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fake News Detection models")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the training CSV file. Defaults to data/dataset.csv (or data/sample_data.csv if available).",
    )
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Disable lemmatization during text preprocessing.",
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_path = Path(args.dataset)
    elif DEFAULT_DATASET.exists():
        dataset_path = DEFAULT_DATASET
    elif SAMPLE_DATASET.exists():
        logger.warning(
            "Main dataset not found at %s. Using sample dataset for demonstration.\n"
            "NOTE: For real evaluation, place your dataset CSV at: %s",
            DEFAULT_DATASET,
            DEFAULT_DATASET,
        )
        dataset_path = SAMPLE_DATASET
    else:
        logger.error(
            "\n"
            "ERROR: No dataset found.\n"
            "Please place your CSV dataset at: %s\n"
            "Expected columns: 'text' (required), 'title' (optional), 'label' (required).\n"
            "The 'label' column should contain REAL/FAKE (or 0/1) values.\n"
            "\n"
            "Alternatively, pass a custom path with: python src/train.py --dataset /path/to/file.csv\n",
            DEFAULT_DATASET,
        )
        sys.exit(1)

    run_training(dataset_path, use_lemmatization=not args.no_lemmatize)


if __name__ == "__main__":
    # Allow running as  python src/train.py  from the project root
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
