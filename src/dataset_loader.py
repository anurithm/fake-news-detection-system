"""
dataset_loader.py
-----------------
Loads, validates, and normalises a fake-news dataset.

Supported input modes
─────────────────────
1. Single CSV file  (path ends in .csv)
   Expected columns: 'text' (required), 'title' (optional), 'label' (required).
   Labels are normalised to REAL / FAKE automatically.

2. Directory containing Fake.csv + True.csv  (Kaggle format)
   Pass the directory path (e.g. "data/") and the loader will find both files,
   assign FAKE to every row in Fake.csv and REAL to every row in True.csv,
   combine them, and return a clean, shuffled DataFrame.
   Expected columns in each Kaggle CSV: 'title', 'text', 'subject', 'date'
   (only 'title' and 'text' are required; 'subject' and 'date' are ignored).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Label normalisation map
# ──────────────────────────────────────────────
_LABEL_MAP: dict[str, str] = {
    "1": "FAKE",
    "0": "REAL",
    "fake": "FAKE",
    "false": "FAKE",
    "pants-fire": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "REAL",
    "real": "REAL",
    "true": "REAL",
    "mostly true": "REAL",
    "fake news": "FAKE",
    "real news": "REAL",
}


def normalize_label(raw: str) -> Optional[str]:
    """Return canonical REAL / FAKE label or None if unrecognised."""
    key = str(raw).strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    if key == "real":
        return "REAL"
    if key == "fake":
        return "FAKE"
    return None


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply common cleaning steps to a DataFrame that already has
    'combined_text' and 'label' columns.

    Returns a cleaned, reset DataFrame or raises ValueError if nothing is left.
    """
    # Remove rows with empty combined_text
    empty_mask = df["combined_text"].str.strip() == ""
    n_empty = int(empty_mask.sum())
    if n_empty:
        logger.warning("Removing %d rows with empty text.", n_empty)
        df = df[~empty_mask]

    # Drop rows with missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values detected:\n%s\nDropping affected rows.", missing)
        df = df.dropna(subset=["combined_text", "label"])

    # Remove duplicates
    n_dups = int(df.duplicated(subset=["combined_text"]).sum())
    if n_dups:
        logger.warning("Removing %d duplicate articles.", n_dups)
        df = df.drop_duplicates(subset=["combined_text"])

    # Normalise labels
    df = df.copy()
    df["label"] = df["label"].apply(normalize_label)
    n_bad = int(df["label"].isnull().sum())
    if n_bad:
        logger.warning("%d rows had unrecognised labels and will be dropped.", n_bad)
        df = df.dropna(subset=["label"])

    if df.empty:
        raise ValueError(
            "No usable rows remain after cleaning. Please check your dataset."
        )

    dist = df["label"].value_counts()
    logger.info("Class distribution after cleaning:\n%s", dist.to_string())

    return df.reset_index(drop=True)


def _build_combined_text(df: pd.DataFrame, title_col: Optional[str], text_col: Optional[str]) -> pd.Series:
    """Merge title and text columns into a single combined_text Series."""
    if title_col and text_col:
        return (
            df[title_col].fillna("").astype(str)
            + " "
            + df[text_col].fillna("").astype(str)
        ).str.strip()
    elif text_col:
        return df[text_col].fillna("").astype(str).str.strip()
    else:
        return df[title_col].fillna("").astype(str).str.strip()  # type: ignore[index]


# ──────────────────────────────────────────────
# Kaggle directory loader (Fake.csv + True.csv)
# ──────────────────────────────────────────────

def _load_kaggle_directory(directory: Path) -> pd.DataFrame:
    """
    Load the standard Kaggle fake-news dataset from a directory that contains
    Fake.csv and True.csv.

    Fake.csv  → label FAKE
    True.csv  → label REAL

    The two files are combined and shuffled with random_state=42.
    """
    fake_path = directory / "Fake.csv"
    true_path = directory / "True.csv"

    missing = [str(p) for p in (fake_path, true_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle dataset files not found: {missing}\n"
            f"Expected both Fake.csv and True.csv inside: {directory}\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    logger.info("Loading Kaggle dataset from directory: %s", directory)
    fake_df = pd.read_csv(fake_path, low_memory=False)
    true_df = pd.read_csv(true_path, low_memory=False)
    logger.info("Fake.csv: %d rows | True.csv: %d rows", len(fake_df), len(true_df))

    # Assign canonical labels — ignore any existing label column
    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    combined = pd.concat([fake_df, true_df], ignore_index=True)

    # Detect title / text columns (case-insensitive)
    col_lower = {c.lower(): c for c in combined.columns}
    title_col = col_lower.get("title")
    text_col  = col_lower.get("text")

    if text_col is None and title_col is None:
        raise ValueError(
            "Neither 'title' nor 'text' column found in Fake.csv / True.csv. "
            f"Found columns: {list(combined.columns)}"
        )

    combined["combined_text"] = _build_combined_text(combined, title_col, text_col)
    result = combined[["combined_text", "label"]].copy()

    # Clean
    result = _clean_dataframe(result)

    # Shuffle reproducibly
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Kaggle dataset loaded and shuffled. Total rows: %d", len(result))
    return result


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a dataset and return a validated, clean DataFrame with columns:
    ['combined_text', 'label'].

    Accepts:
      • Path to a single CSV file  (title/text/label columns)
      • Path to a directory        (must contain Fake.csv + True.csv)

    Raises
    ------
    FileNotFoundError  – path does not exist / required files missing.
    ValueError         – missing required columns or no usable rows after cleaning.
    """
    path = Path(csv_path)

    # ── Directory mode: Kaggle Fake.csv + True.csv ────────────────────
    if path.is_dir():
        return _load_kaggle_directory(path)

    # ── Single CSV mode ───────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Options:\n"
            "  • Place a CSV file with columns: title (optional), text, label\n"
            "  • Place Fake.csv and True.csv in a directory and pass the directory path\n"
            "    e.g.  python src/train.py --dataset data"
        )

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # ── Column detection ──────────────────────────────────────────────
    col_lower = {c.lower(): c for c in df.columns}

    if "label" not in col_lower:
        raise ValueError(
            f"Dataset must contain a 'label' column. Found: {list(df.columns)}"
        )
    label_col = col_lower["label"]

    text_col  = col_lower.get("text")
    title_col = col_lower.get("title")

    if text_col is None and title_col is None:
        raise ValueError(
            "Dataset must contain a 'text' column (and optionally a 'title' column). "
            f"Found: {list(df.columns)}"
        )

    # ── Build combined_text ───────────────────────────────────────────
    df = df.copy()
    df["combined_text"] = _build_combined_text(df, title_col, text_col)
    df["label"] = df[label_col]
    df = df[["combined_text", "label"]].copy()

    # ── Clean ─────────────────────────────────────────────────────────
    return _clean_dataframe(df)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict describing the loaded dataset."""
    dist = df["label"].value_counts().to_dict()
    return {
        "total_samples": len(df),
        "real_count": dist.get("REAL", 0),
        "fake_count": dist.get("FAKE", 0),
        "class_distribution": dist,
        "avg_text_length": int(df["combined_text"].str.split().str.len().mean()),
    }
