# AI-Powered Fake News Detection System

An advanced multi-layered application that evaluates whether news claims are reliable, leveraging live web-evidence verification as the primary truth signal and utilizing local Machine Learning statistical models as secondary signals.

## LIVE DEMO 📎
https://fake-news-detection-system-3tu5r8wmh7omda43nbtmjv.streamlit.app/

## Project Overview
This project targets the rapid spread of online misinformation by applying an **Evidence-First** evaluation pipeline. Instead of blindly trusting a static machine learning model (which can inherit domain biases), the system:
1. Extracts the primary factual claim dynamically using a local LLM.
2. Performs live internet keyword searches (via DuckDuckGo and Google News RSS) to gather current independent news sources.
3. Automatically ranks and evaluates the retrieved evidence against the claim.
4. Outputs a definitive, evidence-backed verdict (`REAL NEWS`, `FAKE NEWS`, or `UNVERIFIED`).

A secondary classical ML pipeline (Linear SVM + TF-IDF) is also evaluated statically in the background to provide a heuristic pattern match.

## Tech Stack
* **Language:** Python 3.12
* **Web Dashboard:** Streamlit
* **Live Evidence Search:** `duckduckgo_search` (ddgs), `feedparser`
* **Local LLM Engine:** Ollama (`llama3:latest`) for factual extraction and assessment without inventing data.
* **Classical ML Engine:** `scikit-learn` (Linear SVM, Logistic Regression, Random Forest, Naive Bayes)
* **NLP Processing:** `nltk` (WordNet Lemmatization)
* **Data Handling:** `pandas`, `numpy`
* **Local Storage:** SQLite (for persisting prediction history natively)

## Installed Models & Tools
* **LLM:** `llama3:latest` running locally via Ollama.
* **ML Classifier:** Optimized Linear SVM with TF-IDF Vectoriser handling up to 50,000 features.
* **Dataset:** Trained on the 70,000+ article `WELFake` HuggingFace dataset, locally mixed with specific cross-domain samples (`sample_data.csv`).

## How to Run

### 1. Prerequisites
Ensure you have Python 3.12+ and [Ollama](https://ollama.ai) installed.
You must pull the required local LLM model beforehand:
```bash
ollama run llama3:latest
```

### 2. Setup
Clone the repository and install the required dependencies:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
Launch the Streamlit web dashboard directly:
```bash
streamlit run app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`. Navigate to the "Detect News" page to submit queries.*

---
*License: MIT*
