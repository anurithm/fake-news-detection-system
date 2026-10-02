"""
app.py
------
Streamlit multi-page dashboard for the AI-Powered Fake News Detection System.

Pages:
  1. Home
  2. Detect News
  3. Model Performance
  4. Text Analysis
  5. Prediction History
  6. About
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1b1632 0%, #29214b 50%, #151128 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.3rem 0; color: white; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin: 0; color: white; }

    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-real {
        background: linear-gradient(135deg, #112d1c, #163e26);
        border: 2px solid #28a745;
        color: #d4edda;
    }
    .result-fake {
        background: linear-gradient(135deg, #3d1418, #5a1e24);
        border: 2px solid #dc3545;
        color: #f8d7da;
    }
    .result-card h2 { font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .result-card p  { font-size: 1rem; margin: 0.3rem 0; }

    .disclaimer-box {
        background: #2a2010;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: #ffda6a;
    }
    .info-card {
        background: #1b1632;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #6c4aff;
        margin: 0.5rem 0;
        color: white;
    }
    .metric-card {
        background: #1b1632;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: white;
    }
    .metric-card h3 { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0; }
    .metric-card p  { font-size: 0.85rem; color: #a49fc4; margin: 0.2rem 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: check if models exist ─────────────────────────────────────────────

def models_trained() -> bool:
    return (MODELS_DIR / "tfidf_vectorizer.joblib").exists() and \
           (MODELS_DIR / "best_model.joblib").exists()


def load_model_info() -> dict:
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("*NLP & Machine Learning*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔎 Detect News", "📊 Model Performance",
         "📝 Text Analysis", "📋 Prediction History", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    if models_trained():
        info = load_model_info()
        st.success("✅ Models trained")
        st.caption(f"Best model: **{info.get('best_model_name', 'N/A')}**")
        st.caption(f"Best F1: **{info.get('best_model_f1', 0):.4f}**")
        n_total = info.get('total_samples')
        if n_total:
            st.caption(f"Dataset: **{n_total:,}** articles")
            st.caption(f"REAL: {info.get('real_count', 0):,} · FAKE: {info.get('fake_count', 0):,}")
    else:
        st.warning("⚠️ Models not trained yet")
        st.caption("Run: `python src/train.py --dataset <path>`")
    st.divider()
    st.caption("AI Fake News Detector · MIT License")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown(
        """
        <div class="main-header">
          <h1>🔍 AI-Powered Fake News Detection System</h1>
          <p>NLP and Machine Learning Based News Classification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>4</h3><p>ML Models Compared</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>TF-IDF</h3><p>Feature Extraction</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>NLP</h3><p>Text Preprocessing</p></div>', unsafe_allow_html=True)

    st.markdown("")

    with st.expander("📌 About this Project", expanded=True):
        st.markdown(
            """
            This system uses **Natural Language Processing (NLP)** and **Machine Learning** to classify
            news articles as *REAL* or *FAKE* based on patterns learned from a labelled training dataset.

            It compares four ML classifiers — **Logistic Regression**, **Multinomial Naive Bayes**,
            **Linear SVM**, and **Random Forest** — and automatically selects the best-performing model
            based on the **weighted F1-score**.
            """
        )

    with st.expander("🛠️ Technology Stack"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "- **Python 3.12**\n"
                "- **Scikit-learn** — ML models\n"
                "- **NLTK** — text preprocessing\n"
                "- **TF-IDF** — feature extraction"
            )
        with col_b:
            st.markdown(
                "- **Streamlit** — web dashboard\n"
                "- **Pandas / NumPy** — data handling\n"
                "- **Matplotlib / Seaborn** — charts\n"
                "- **SQLite** — prediction history"
            )

    with st.expander("⚙️ How the System Works"):
        st.markdown(
            """
            1. **Preprocessing** — text is cleaned (lowercased, URLs removed, HTML stripped, stop-words removed, lemmatised)
            2. **TF-IDF Vectorisation** — words are converted to numerical feature vectors
            3. **Classification** — a trained ML model predicts REAL or FAKE
            4. **Confidence** — calibrated probability of the predicted class is reported
            """
        )

    st.markdown(
        """
        <div class="disclaimer-box">
        ⚠️ <strong>Important Limitation:</strong>
        This system predicts the class label learned from its training data.
        It does <em>not</em> independently verify real-world facts.
        Predictions should not be used as the sole basis for accepting or rejecting any news claim.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DETECT NEWS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Detect News":
    st.markdown(
        '<div class="main-header"><h1>🔎 Detect News</h1>'
        '<p>Submit a news article or headline for classification</p></div>',
        unsafe_allow_html=True,
    )

    if not models_trained():
        st.error(
            "⚠️ **Models not trained yet.**\n\n"
            "Run the training script first:\n\n```\npython src/train.py\n```"
        )
        st.stop()

    headline = st.text_input("📰 Headline (optional)", placeholder="Enter the article headline here…")
    article_text = st.text_area(
        "📄 Article Text",
        placeholder="Paste the full news article text here…",
        height=220,
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 5])
    analyze_clicked = col_btn1.button("🔍 Analyze", type="primary", use_container_width=True)
    clear_clicked   = col_btn2.button("🗑 Clear",   use_container_width=True)

    if clear_clicked:
        st.rerun()

    if analyze_clicked:
        combined = (headline + " " + article_text).strip() if headline else article_text.strip()

        if not combined:
            st.warning("Please enter some text before clicking Analyze.")
        else:
            with st.spinner("Analysing…"):
                try:
                    from src.predict import predict_text
                    from src.preprocessor import clean_text, get_text_stats
                    from src.database import save_prediction

                    result = predict_text(combined)
                    prediction  = result["prediction"]
                    confidence  = result["confidence"]
                    model_name  = result["model_name"]
                    label_probs = result["label_probs"]

                    cleaned = clean_text(combined)
                    stats   = get_text_stats(combined, cleaned)

                    # ── Result card ────────────────────────────────────────
                    card_cls = "result-real" if prediction == "REAL" else "result-fake"
                    emoji    = "✅" if prediction == "REAL" else "❌"
                    label    = "REAL NEWS" if prediction == "REAL" else "FAKE NEWS"
                    st.markdown(
                        f'<div class="result-card {card_cls}">'
                        f'<h2>{emoji} Model Prediction: {label}</h2>'
                        f'<p>Confidence: <strong>{confidence * 100:.2f}%</strong></p>'
                        f'<p>Model used: <strong>{model_name}</strong></p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Probabilities
                    p_col1, p_col2 = st.columns(2)
                    p_col1.metric("REAL probability", f"{label_probs.get('REAL', 0)*100:.2f}%")
                    p_col2.metric("FAKE probability", f"{label_probs.get('FAKE', 0)*100:.2f}%")

                    # Text statistics
                    st.markdown("#### 📊 Text Statistics")
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("Word Count",     stats["word_count"])
                    s_col2.metric("Sentence Count", stats["sentence_count"])
                    s_col3.metric("Character Count", stats["char_count"])

                    # Disclaimer
                    st.markdown(
                        '<div class="disclaimer-box">'
                        "ℹ️ This prediction reflects patterns learned from the training dataset "
                        "and is <strong>not</strong> independent factual verification."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # Persist to history
                    save_prediction(combined, prediction, confidence, model_name)

                except FileNotFoundError as e:
                    st.error(f"Model file missing: {e}")
                except ValueError as e:
                    st.warning(f"Input error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

            # ── Evidence Verification Section ─────────────────────────────
            st.divider()
            st.markdown("## 🔎 Evidence-Based Verification")
            
            # Use columns to align the button nicely
            ev_btn_col1, ev_btn_col2, _ = st.columns([2, 1, 3])
            if ev_btn_col1.button("Verify with Live Evidence", use_container_width=True, type="secondary"):
                with st.spinner("Searching for relevant evidence..."):
                    try:
                        from src.evidence_verifier import run_verification_pipeline
                        
                        # Cache the backend call at the Streamlit layer to prevent duplicate requests
                        @st.cache_data(show_spinner=False, ttl=300)
                        def get_evidence(text: str):
                            return run_verification_pipeline(text)
                            
                        ev_result = get_evidence(combined)
                        
                        st.markdown(f"**Verification Status:** `{ev_result['status']}`")
                        st.markdown(f"**Evidence Strength:** `{ev_result['strength']}`")
                        st.markdown(f"**Main Claim:** {ev_result['claim']}")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("### Supporting Evidence")
                            if ev_result['supporting']:
                                for e in ev_result['supporting']:
                                    st.markdown(f"**Source:** {e['source']}")
                                    st.markdown(f"**Title:** {e['title']}")
                                    st.markdown(f"**Date:** {e.get('date', 'Unknown')}")
                                    st.markdown(f"*Relevant excerpt:* {e['snippet'][:150]}...")
                                    st.markdown(f"[Open source]({e['url']})")
                                    st.divider()
                            else:
                                st.write("No significant supporting evidence found.")
                                
                        with c2:
                            st.markdown("### Conflicting Evidence")
                            if ev_result['conflicting']:
                                for e in ev_result['conflicting']:
                                    st.markdown(f"**Source:** {e['source']}")
                                    st.markdown(f"**Title:** {e['title']}")
                                    st.markdown(f"**Date:** {e.get('date', 'Unknown')}")
                                    st.markdown(f"*Relevant excerpt:* {e['snippet'][:150]}...")
                                    st.markdown(f"[Open source]({e['url']})")
                                    st.divider()
                            else:
                                st.write("No significant conflicting evidence found.")
                                
                        st.markdown("### Verification Explanation")
                        st.info(ev_result['explanation'])
                        
                    except Exception as e:
                        st.error(f"Live evidence verification is temporarily unavailable. Error: {e}")
                        st.info("The ML classification is still available.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.markdown(
        '<div class="main-header"><h1>📊 Model Performance</h1>'
        '<p>Actual evaluation metrics from the training pipeline</p></div>',
        unsafe_allow_html=True,
    )

    if not models_trained():
        st.error("⚠️ Models not trained yet. Run `python src/train.py` first.")
        st.stop()

    from src.utils import load_eval_results

    eval_df = load_eval_results()

    if eval_df is None:
        st.warning("Evaluation results CSV not found. Please retrain the models.")
        st.stop()

    # Metrics table
    st.subheader("Model Comparison Table")
    display_df = eval_df[["model", "accuracy", "precision", "recall", "f1_score"]].copy()
    display_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    st.dataframe(
        display_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score"],
            color="#d4edda",
        ).format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
        use_container_width=True,
    )

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        cmp_img = REPORTS_DIR / "model_comparison.png"
        if cmp_img.exists():
            st.subheader("Performance Bar Chart")
            st.image(str(cmp_img), use_container_width=True)

    with chart_col2:
        cm_img = REPORTS_DIR / "best_model_confusion_matrix.png"
        if cm_img.exists():
            info = load_model_info()
            st.subheader(f"Confusion Matrix — {info.get('best_model_name', 'Best Model')}")
            st.image(str(cm_img), use_container_width=True)

    dist_img = REPORTS_DIR / "class_distribution.png"
    if dist_img.exists():
        st.subheader("Dataset Class Distribution")
        _, dist_mid, _ = st.columns([1, 3, 1])
        with dist_mid:
            st.image(str(dist_img), use_container_width=True)

    # Model info summary
    info = load_model_info()
    if info:
        st.divider()
        st.subheader("Training Run Summary")
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        i_col1.metric("Total Samples",   info.get("total_samples",  "N/A"))
        i_col2.metric("Training Samples", info.get("train_size",     "N/A"))
        i_col3.metric("Test Samples",     info.get("test_size",      "N/A"))
        i_col4.metric("Vocabulary Size",  info.get("vocab_size",     "N/A"))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TEXT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📝 Text Analysis":
    st.markdown(
        '<div class="main-header"><h1>📝 Text Analysis</h1>'
        '<p>Inspect how the NLP pipeline processes your text</p></div>',
        unsafe_allow_html=True,
    )

    if not models_trained():
        st.warning("⚠️ Please train the models first to enable TF-IDF analysis.")

    raw_input = st.text_area("Paste text to analyse:", height=200)

    if st.button("Analyse Text", type="primary"):
        if not raw_input.strip():
            st.warning("Please enter some text.")
        else:
            from src.preprocessor import clean_text, get_text_stats

            cleaned = clean_text(raw_input)
            stats   = get_text_stats(raw_input, cleaned)

            st.subheader("📊 Text Statistics")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Words (original)",    stats["word_count"])
            sc2.metric("Sentences",            stats["sentence_count"])
            sc3.metric("Characters",           stats["char_count"])
            sc4.metric("Words (after cleaning)", stats["cleaned_word_count"])

            col_raw, col_cln = st.columns(2)
            with col_raw:
                st.subheader("Original Text")
                st.text_area("", value=raw_input, height=200, disabled=True, label_visibility="collapsed")
            with col_cln:
                st.subheader("Cleaned Text")
                st.text_area("", value=cleaned, height=200, disabled=True, label_visibility="collapsed")

            # TF-IDF top terms (only if models are trained)
            if models_trained():
                try:
                    import joblib
                    from src.utils import get_top_tfidf_terms, plot_top_tfidf

                    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
                    terms = get_top_tfidf_terms(raw_input, vectorizer, top_n=20)

                    if terms:
                        st.subheader("🔑 Top TF-IDF Terms")
                        fig = plot_top_tfidf(terms)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)

                        terms_df = pd.DataFrame(terms, columns=["Term", "TF-IDF Score"])
                        st.dataframe(terms_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No significant TF-IDF terms found after preprocessing.")
                except Exception as e:
                    st.warning(f"Could not compute TF-IDF terms: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📋 Prediction History":
    st.markdown(
        '<div class="main-header"><h1>📋 Prediction History</h1>'
        '<p>Recent predictions stored in the local SQLite database</p></div>',
        unsafe_allow_html=True,
    )

    try:
        from src.database import get_predictions, clear_history, init_db

        init_db()

        col_h, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("🗑 Clear History", type="secondary"):
                n = clear_history()
                st.success(f"Cleared {n} record(s).")
                st.rerun()

        rows = get_predictions(limit=100)

        if not rows:
            st.info("No predictions yet. Go to **Detect News** to analyse an article.")
        else:
            hist_df = pd.DataFrame(rows)
            hist_df = hist_df.rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "article": "Article (preview)",
                "prediction": "Prediction",
                "confidence": "Confidence",
                "model_name": "Model",
            })
            hist_df["Article (preview)"] = hist_df["Article (preview)"].str[:80] + "…"
            hist_df["Confidence"] = hist_df["Confidence"].apply(
                lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
            )
            # Colour-hint prediction
            def color_pred(val):
                color = "#d4edda" if val == "REAL" else "#f8d7da"
                return f"background-color: {color}"

            st.dataframe(
                hist_df[["Timestamp", "Prediction", "Confidence", "Model", "Article (preview)"]]
                .style.map(color_pred, subset=["Prediction"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing {len(rows)} most recent predictions.")

    except Exception as e:
        st.error(f"Database error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown(
        '<div class="main-header"><h1>ℹ️ About This Project</h1>'
        '<p>Methodology, limitations, and future enhancements</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("📌 Problem Statement", expanded=True):
        st.markdown(
            """
            The proliferation of misinformation online has made it increasingly difficult for readers
            to distinguish credible news from fabricated content. Manual fact-checking is slow and
            does not scale to the volume of content published daily.
            """
        )

    with st.expander("🎯 Objective"):
        st.markdown(
            """
            Build an automated text classification system that learns patterns from a labelled dataset
            of real and fake news articles and uses those patterns to classify new, unseen articles.
            """
        )

    with st.expander("🔬 Methodology"):
        st.markdown(
            """
            1. **Dataset** – CSV with `text`, optional `title`, and `label` columns.
            2. **Preprocessing** – lowercase, URL/HTML removal, punctuation handling, stop-word removal, lemmatisation.
            3. **Feature Extraction** – TF-IDF (Term Frequency – Inverse Document Frequency) with bigrams.
            4. **Model Training** – Four sklearn classifiers trained on 80 % of the data.
            5. **Evaluation** – Accuracy, Precision, Recall, F1 on the held-out 20 % test set.
            6. **Model Selection** – Best model by weighted F1-score is saved and used for inference.
            """
        )

    with st.expander("📐 Machine Learning Algorithms"):
        st.markdown(
            """
            | Model | Key Characteristics |
            |-------|-------------------|
            | **Logistic Regression** | Linear, interpretable, fast, supports probabilities |
            | **Multinomial Naive Bayes** | Probabilistic, assumes feature independence, great for text |
            | **Linear SVM** | High-margin boundary, robust to high-dimensional space |
            | **Random Forest** | Ensemble of decision trees, captures non-linear patterns |
            """
        )

    with st.expander("📏 Evaluation Metrics"):
        st.markdown(
            """
            - **Accuracy** – fraction of correctly classified articles
            - **Precision** – of all predicted FAKE, how many were actually FAKE
            - **Recall** – of all actual FAKE, how many were correctly identified
            - **F1-Score** – harmonic mean of Precision and Recall (primary selection criterion)
            - **Confusion Matrix** – breakdown of TP, TN, FP, FN
            """
        )

    with st.expander("⚠️ Limitations"):
        st.markdown(
            """
            - **Not a fact-checker** – this system classifies based on patterns learned from training data,
              and cannot verify real-world factual claims.
            - **Dataset-dependent** – performance depends entirely on the quality and diversity of the training corpus.
            - **Domain sensitivity** – a model trained on English news may fail on other languages or domains.
            - **Adversarial robustness** – sufficiently well-written fake news may fool the classifier.
            - **Static model** – the model does not update automatically as new misinformation styles emerge.
            """
        )

    with st.expander("🚀 Conceptual Architecture & Verification"):
        st.markdown(
            """
            Traditional fake-news classifiers identify linguistic patterns learned from historical data. 
            This system has been **upgraded** to add an independent evidence-verification layer that retrieves 
            current public sources dynamically and compares the extracted claims against live available evidence.
            
            **Important Distinctions:**
            - **ML confidence is not factual certainty.** It simply measures closeness to the training distribution.
            - **Evidence verification depends on the availability and quality of retrieved sources.** It leverages Google News and DuckDuckGo to find related reporting dynamically.
            """
        )

    with st.expander("🛠️ Future Enhancements (Not Implemented)"):
        st.markdown(
            """
            - 🌐 Multilingual fake-news detection (Indian-language news)
            - 🤖 Deep integration with local LLMs (Ollama) natively
            - 📡 Real-time news feed monitoring
            """
        )

    with st.expander("⚖️ Ethical Considerations"):
        st.markdown(
            """
            - Predictions should supplement, not replace, human editorial judgement.
            - Mis-labelling legitimate news as fake could have serious consequences.
            - Training data bias can amplify existing societal biases.
            - This tool must be used responsibly and with clear disclosure of its limitations.
            """
        )

    st.divider()
    st.markdown(
        "**Project:** AI-Powered Fake News Detection System | "
        "**Stack:** Python · Scikit-learn · NLTK · Streamlit | "
        "**License:** MIT"
    )
