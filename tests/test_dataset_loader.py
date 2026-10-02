"""
test_dataset_loader.py
-----------------------
Tests for the dataset loading and validation module.
Covers both single-CSV mode and Kaggle directory mode (Fake.csv + True.csv).
"""

import csv
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_loader import load_dataset, dataset_summary, normalize_label

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _write_csv(content: str, suffix: str = ".csv") -> Path:
    """Write CSV text to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


def _make_kaggle_dir(
    fake_rows: list[dict] | None = None,
    real_rows: list[dict] | None = None,
    missing: str | None = None,        # 'fake' | 'real' | None
) -> Path:
    """
    Create a temp directory with Fake.csv and/or True.csv for testing.
    Rows must be dicts with keys: title, text, subject, date.
    """
    tmpdir = Path(tempfile.mkdtemp())

    default_fake = [
        {"title": "Fake headline one", "text": "Fake article body one here.",   "subject": "News", "date": "January 1 2019"},
        {"title": "Fake headline two", "text": "Fake article body two here.",   "subject": "News", "date": "January 2 2019"},
        {"title": "Fake headline three","text": "Fake article body three here.","subject": "News", "date": "January 3 2019"},
    ]
    default_real = [
        {"title": "Real headline one", "text": "Real article body one here.",   "subject": "World", "date": "January 1 2019"},
        {"title": "Real headline two", "text": "Real article body two here.",   "subject": "World", "date": "January 2 2019"},
        {"title": "Real headline three","text": "Real article body three here.","subject": "World", "date": "January 3 2019"},
    ]

    fake_rows = fake_rows or default_fake
    real_rows = real_rows or default_real

    fields = ["title", "text", "subject", "date"]

    if missing != "fake":
        with open(tmpdir / "Fake.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(fake_rows)

    if missing != "real":
        with open(tmpdir / "True.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(real_rows)

    return tmpdir


# ──────────────────────────────────────────────────────────────
# Label normalisation
# ──────────────────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_fake_variants(self):
        for raw in ["fake", "FAKE", "1", "false"]:
            assert normalize_label(raw) == "FAKE", f"Failed for {raw!r}"

    def test_real_variants(self):
        for raw in ["real", "REAL", "0", "true", "mostly-true"]:
            assert normalize_label(raw) == "REAL", f"Failed for {raw!r}"

    def test_unknown_returns_none(self):
        assert normalize_label("xyz_unknown") is None


# ──────────────────────────────────────────────────────────────
# Single CSV mode
# ──────────────────────────────────────────────────────────────

class TestLoadDatasetSingleCSV:
    def test_basic_valid_csv(self):
        csv = "text,label\nSome real news article text,REAL\nSome fake news article,FAKE\n"
        path = _write_csv(csv)
        df = load_dataset(path)
        assert len(df) >= 1
        assert set(df["label"].unique()).issubset({"REAL", "FAKE"})

    def test_with_title_column(self):
        csv = "title,text,label\nHeadline,Body text real,REAL\nFake headline,Fake body,FAKE\n"
        path = _write_csv(csv)
        df = load_dataset(path)
        assert "combined_text" in df.columns
        assert "Headline" in df["combined_text"].iloc[0]

    def test_title_merged_with_text(self):
        csv = "title,text,label\nMyTitle,MyBody,REAL\n"
        path = _write_csv(csv)
        df = load_dataset(path)
        assert "MyTitle" in df["combined_text"].iloc[0]
        assert "MyBody" in df["combined_text"].iloc[0]

    def test_missing_label_raises(self):
        csv = "text\nSome text without label\n"
        path = _write_csv(csv)
        with pytest.raises(ValueError, match="label"):
            load_dataset(path)

    def test_missing_text_and_title_raises(self):
        csv = "headline,label\nHeadline text,REAL\n"
        path = _write_csv(csv)
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/to/file.csv")

    def test_empty_text_rows_dropped(self):
        csv = "text,label\n,REAL\nValid article content here,FAKE\n"
        path = _write_csv(csv)
        df = load_dataset(path)
        assert all(df["combined_text"].str.strip() != "")

    def test_duplicates_dropped(self):
        csv = "text,label\nDuplicate article text here,REAL\nDuplicate article text here,REAL\n"
        path = _write_csv(csv)
        df = load_dataset(path)
        assert len(df) == 1

    def test_sample_data_csv_is_valid(self):
        """Ensure the project sample_data.csv loads correctly."""
        sample = Path(__file__).parent.parent / "data" / "sample_data.csv"
        if not sample.exists():
            pytest.skip("sample_data.csv not found")
        df = load_dataset(sample)
        assert len(df) > 0
        assert set(df["label"].unique()).issubset({"REAL", "FAKE"})
        assert df["label"].nunique() == 2, "sample_data.csv should have both REAL and FAKE rows"


# ──────────────────────────────────────────────────────────────
# Directory mode: Fake.csv + True.csv (Kaggle format)
# ──────────────────────────────────────────────────────────────

class TestLoadDatasetKaggleDirectory:
    def test_loads_both_files(self):
        tmpdir = _make_kaggle_dir()
        df = load_dataset(tmpdir)
        assert len(df) == 6  # 3 fake + 3 real
        assert set(df["label"].unique()) == {"REAL", "FAKE"}

    def test_fake_csv_gets_fake_label(self):
        tmpdir = _make_kaggle_dir()
        df = load_dataset(tmpdir)
        # All rows from Fake.csv should be FAKE
        fake_rows = df[df["combined_text"].str.contains("Fake article body")]
        assert (fake_rows["label"] == "FAKE").all(), "Rows from Fake.csv should all be labelled FAKE"

    def test_true_csv_gets_real_label(self):
        tmpdir = _make_kaggle_dir()
        df = load_dataset(tmpdir)
        real_rows = df[df["combined_text"].str.contains("Real article body")]
        assert (real_rows["label"] == "REAL").all(), "Rows from True.csv should all be labelled REAL"

    def test_missing_fake_csv_raises(self):
        tmpdir = _make_kaggle_dir(missing="fake")
        with pytest.raises(FileNotFoundError, match="Fake.csv"):
            load_dataset(tmpdir)

    def test_missing_true_csv_raises(self):
        tmpdir = _make_kaggle_dir(missing="real")
        with pytest.raises(FileNotFoundError, match="True.csv"):
            load_dataset(tmpdir)

    def test_title_and_text_combined(self):
        tmpdir = _make_kaggle_dir()
        df = load_dataset(tmpdir)
        # combined_text should contain both title and text words
        first = df["combined_text"].iloc[0]
        assert len(first.split()) > 3

    def test_empty_text_rows_dropped(self):
        fake_rows = [
            {"title": "Title A", "text": "",              "subject": "News", "date": "Jan 1"},
            {"title": "Title B", "text": "Valid body.",   "subject": "News", "date": "Jan 1"},
        ]
        real_rows = [
            {"title": "Real title", "text": "Real body.", "subject": "World","date": "Jan 1"},
        ]
        tmpdir = _make_kaggle_dir(fake_rows=fake_rows, real_rows=real_rows)
        df = load_dataset(tmpdir)
        # Row with empty text ('Title A') should be removed
        assert not any(df["combined_text"].str.strip() == "")

    def test_duplicate_rows_dropped(self):
        dup_text = "Exactly the same article body text for duplicate test."
        fake_rows = [
            {"title": "Dup", "text": dup_text, "subject": "News", "date": "Jan 1"},
            {"title": "Dup", "text": dup_text, "subject": "News", "date": "Jan 1"},
        ]
        real_rows = [
            {"title": "Unique", "text": "Completely different content here.", "subject": "World", "date": "Jan 1"},
        ]
        tmpdir = _make_kaggle_dir(fake_rows=fake_rows, real_rows=real_rows)
        df = load_dataset(tmpdir)
        # duplicate fake row should be removed
        assert len(df[df["label"] == "FAKE"]) == 1

    def test_result_has_combined_text_and_label_only(self):
        tmpdir = _make_kaggle_dir()
        df = load_dataset(tmpdir)
        assert list(df.columns) == ["combined_text", "label"]

    def test_shuffled_deterministically(self):
        """Two calls with same data should produce same order (random_state=42)."""
        tmpdir = _make_kaggle_dir()
        df1 = load_dataset(tmpdir)
        df2 = load_dataset(tmpdir)
        assert df1["combined_text"].tolist() == df2["combined_text"].tolist()


# ──────────────────────────────────────────────────────────────
# dataset_summary
# ──────────────────────────────────────────────────────────────

class TestDatasetSummary:
    def test_summary_keys(self):
        df = pd.DataFrame({
            "combined_text": ["a b c", "d e f", "g h i"],
            "label": ["REAL", "FAKE", "REAL"],
        })
        summary = dataset_summary(df)
        assert summary["total_samples"] == 3
        assert summary["real_count"] == 2
        assert summary["fake_count"] == 1
        assert "avg_text_length" in summary
