# AI-Powered Fake News Detection System Using NLP and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **NLP + Machine Learning** system that classifies news articles as *REAL* or *FAKE* based on statistical patterns learned from a labelled training dataset — with a polished multi-page Streamlit dashboard.

> ⚠️ **Important:** This is a **text classification** system, not an independent fact-checking engine. The model predicts the class label that most closely matches patterns in its training data. A high confidence score does **not** prove factual truth or falsehood of a news claim.

---

## Problem Statement

The rapid spread of online misinformation makes it increasingly difficult to distinguish credible journalism from fabricated content. Manual fact-checking does not scale to the daily volume of published articles. This project explores whether statistical NLP methods can reliably detect linguistic patterns associated with real versus fake news at scale.

---

## Objectives

- Build a fully offline, reproducible NLP + ML classification pipeline.
- Compare four ML algorithms and auto-select the best by weighted F1-score.
- Provide a polished web dashboard for interactive article analysis.
- Store prediction history locally in SQLite.
- Keep the system completely API-free — no external services required.

---

## Key Features

| Feature | Details |
|---------|---------|
| **4 ML Models** | Logistic Regression, Naïve Bayes, Linear SVM, Random Forest |
| **TF-IDF** | Unigrams + bigrams, up to 50,000 features |
| **NLP Pipeline** | Lowercasing, URL/HTML removal, stop-word removal, lemmatisation |
| **Auto Model Selection** | Best model by weighted F1-score |
| **Calibrated Confidence** | `predict_proba` for all models — no raw decision scores |
| **Streamlit Dashboard** | 6-page interactive web app |
| **Prediction History** | SQLite persistence |
| **Charts** | Model comparison, confusion matrix, class distribution, TF-IDF terms |
| **Tests** | 53-test pytest suite |
| **Offline** | Runs entirely locally after setup — no paid APIs |

---

## System Architecture

```text
                 USER NEWS
                     |
                     v
             Text Preprocessing
                     |
          +----------+----------+
          |                     |
          v                     v
    ML Classifier        Claim Extraction
          |                     |
          |                     v
          |              Web Evidence Search
          |                     |
          |                     v
          |              Source Collection
          |                     |
          |                     v
          |            Evidence Comparison
          |                     |
          +----------+----------+
                     |
                     v
          Evidence-Based Result
                     |
          +----------+----------+
          |                     |
          v                     v
     ML Prediction       Verification Status
          |                     |
          +----------+----------+
                     |
                     v
             Explanation + Sources
```

---

## Methodology

### 1 · Text Preprocessing

Raw article text is cleaned through a reproducible pipeline:

1. Lowercase conversion
2. URL and HTML removal (regex)
3. Punctuation removal
4. Whitespace normalisation
5. Tokenisation (NLTK punkt)
6. English stop-word removal
7. WordNet lemmatisation

### 2 · Feature Extraction (TF-IDF)

**Term Frequency–Inverse Document Frequency** converts cleaned text to a numerical matrix. The vectoriser is **fit only on training data** to prevent data leakage. Configuration: unigrams + bigrams, max 50,000 features, sublinear TF scaling.

### 3 · Train / Test Split

80 % training / 20 % test, **stratified** by class, `random_state=42`.

### 4 · Models Compared

| Model | Notes |
|-------|-------|
| **Logistic Regression** | `C=1.0`, `max_iter=1000`, fast and interpretable |
| **Multinomial Naive Bayes** | `alpha=0.1`, strong baseline for text |
| **Linear SVM** | Wrapped in `CalibratedClassifierCV` for proper probabilities |
| **Random Forest** | 100 estimators, captures non-linear feature interactions |

### 5 · Model Selection

The model with the highest **weighted F1-score** on the held-out test set is automatically saved as `best_model.joblib`.

---

## Results

Trained on the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — **39,103 articles** (21,196 REAL · 17,907 FAKE) after removing 5,795 duplicate articles.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9916 | 0.9916 | 0.9916 | 0.9916 |
| Multinomial Naive Bayes | 0.9614 | 0.9615 | 0.9614 | 0.9614 |
| **Linear SVM** ⭐ | **0.9968** | **0.9968** | **0.9968** | **0.9968** |
| Random Forest | 0.9951 | 0.9951 | 0.9951 | 0.9951 |

> Metrics are from actual model evaluation on a held-out 20% test set (7,821 articles). These results are specific to this dataset and training configuration and should not be generalised to other domains.

---

## Screenshots

> *Screenshots will be added after the first public deployment.*

| Page | Preview |
|------|---------|
| 🏠 Home | *(screenshot placeholder)* |
| 🔎 Detect News | *(screenshot placeholder)* |
| 📊 Model Performance | *(screenshot placeholder)* |
| 📝 Text Analysis | *(screenshot placeholder)* |
| 📋 Prediction History | *(screenshot placeholder)* |

---

## Technology Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.12 |
| Data | Pandas · NumPy |
| ML | Scikit-learn |
| NLP | NLTK |
| Feature Extraction | TF-IDF (sklearn) |
| Charts | Matplotlib · Seaborn |
| Web App | Streamlit |
| Serialisation | Joblib |
| Database | SQLite (stdlib) |
| Tests | pytest |

---

## Dataset

### Kaggle — Fake and Real News Dataset *(primary training dataset)*

[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset contains two separate CSV files. **Labels are assigned automatically by the loader** — do not add a label column to these files.

| File | Content | Label |
|------|---------|-------|
| `Fake.csv` | ~23,000 fabricated news articles | `FAKE` |
| `True.csv` | ~21,000 Reuters news articles | `REAL` |

Each file has columns: `title`, `text`, `subject`, `date`.

**These files are NOT committed to this repository.** Download from Kaggle and pass the directory path at training time.

### Demo Dataset — `data/sample_data.csv`

A **20-row synthetic dataset** (10 REAL · 10 FAKE) for testing the pipeline end-to-end without downloading the full Kaggle data.

> ⚠️ Metrics from `sample_data.csv` are **meaningless**. Only the Kaggle dataset results above should be cited.

---

## Installation

### Prerequisites

- Python 3.12
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-news-detection-system.git
cd fake-news-detection-system

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training

### Option A — Kaggle dataset *(recommended for real results)*

Download `Fake.csv` and `True.csv` from Kaggle and place them in any local directory, then:

```bash
python src/train.py --dataset /path/to/fakeandrealnewsdataset
```

The loader automatically detects both files, assigns labels, deduplicates, and shuffles.

### Option B — Demo dataset *(pipeline test only)*

```bash
python src/train.py --dataset data/sample_data.csv
```

### Training Options

```
--dataset PATH       Path to CSV file OR directory containing Fake.csv + True.csv
--no-lemmatize       Disable lemmatisation (faster, slightly lower accuracy)
```

---

## Running the Application

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The app requires trained models. Run training first if the `models/` directory is empty.

---

## Running Tests

```bash
python -m pytest tests/ -v
# Expected: 53 passed
```

---

## Project Structure

```
fake-news-detection-system/
├── app.py                          # Streamlit dashboard (6 pages)
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── preprocessor.py             # NLP cleaning pipeline
│   ├── dataset_loader.py           # Flexible CSV + Kaggle directory loader
│   ├── train.py                    # Training pipeline entry point
│   ├── predict.py                  # Prediction + calibrated confidence
│   ├── database.py                 # SQLite prediction history
│   └── utils.py                    # Charts & TF-IDF helpers
│
├── data/
│   └── sample_data.csv             # 20-row demo dataset (committed)
│   # Kaggle CSVs: NOT committed — pass directory via --dataset
│
├── models/                         # NOT committed — generated by training
│   ├── best_model.joblib           # Linear SVM (~1.9 MB)
│   ├── tfidf_vectorizer.joblib     # TF-IDF vectoriser (~1.9 MB)
│   ├── logistic_regression.joblib  # (~392 KB)
│   ├── multinomial_naive_bayes.joblib  # (~1.5 MB)
│   ├── linear_svm.joblib           # (~1.9 MB)
│   ├── random_forest.joblib        # (~34 MB — too large for GitHub)
│   └── model_info.json             # Training metadata
│
├── reports/                        # Generated by training
│   ├── model_comparison.csv        # Metrics table
│   ├── model_comparison.png        # Bar chart
│   ├── best_model_confusion_matrix.png
│   └── class_distribution.png
│
└── tests/
    ├── test_preprocessor.py        # 22 NLP tests
    ├── test_dataset_loader.py      # 23 loader tests (CSV + directory mode)
    └── test_predict.py             # 8 prediction pipeline tests
```

---

## Model Files & GitHub

The `models/` directory is **excluded from Git** (see `.gitignore`).

| Model file | Size | Recommendation |
|------------|------|----------------|
| `best_model.joblib` | ~1.9 MB | Commit directly or via Git LFS |
| `tfidf_vectorizer.joblib` | ~1.9 MB | Commit directly or via Git LFS |
| `logistic_regression.joblib` | ~392 KB | Commit directly |
| `multinomial_naive_bayes.joblib` | ~1.5 MB | Commit directly |
| `linear_svm.joblib` | ~1.9 MB | Commit directly |
| `random_forest.joblib` | **~34 MB** | Git LFS or regenerate from training |

To commit models (except Random Forest):

```bash
# Option 1 — Git LFS (recommended for large files)
git lfs install
git lfs track "*.joblib"
git add .gitattributes models/

# Option 2 — Remove models/ line from .gitignore and commit small models only
# Add random_forest.joblib back to .gitignore manually
```

---

## Limitations

- **Not a fact-checker.** This classifier learns patterns from its training corpus. It cannot verify whether a real-world claim is true or false.
- **Dataset-dependent.** Performance reflects the Kaggle dataset's specific writing styles. Results will differ on other domains (scientific, regional, or non-English news).
- **Static model.** The model does not update automatically as misinformation tactics evolve.
- **Adversarial vulnerability.** Well-crafted fake news designed to mimic real-news style may evade detection.
- **High confidence ≠ factual truth.** A 99% confidence score means the article closely resembles patterns in REAL training examples — it is not independent verification.

---

## Future Enhancements

- 🤖 Transformer-based models (BERT, RoBERTa) for richer semantic understanding
- 🔍 Source credibility analysis alongside article text
- 📰 Claim extraction + retrieval-augmented verification
- 🌐 Multilingual detection (Indian-language news)
- 🧠 Explainable AI (LIME / SHAP) for per-prediction explanations
- ⚡ Real-time news feed monitoring
- 📡 Integration with public fact-checking APIs

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI-Powered Fake News Detection System"
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detection-system.git
git push -u origin main
```

> The `.gitignore` excludes `models/`, Kaggle CSV files (`Fake.csv`, `True.csv`), `data/predictions.db`, and `.venv/`. Only source code, tests, and `data/sample_data.csv` are committed.

---

## Deactivate Virtual Environment

```bash
deactivate
```

---

## License

MIT License — free for educational and research use.
