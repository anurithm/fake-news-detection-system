"""
preprocessor.py
---------------
Reusable NLP text preprocessing pipeline for the Fake News Detection System.
Applies: lowercasing, URL removal, HTML removal, punctuation handling,
         whitespace normalization, tokenization, stop-word removal, and lemmatization.
"""

import re
import string
import logging
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# NLTK resource bootstrap
# ──────────────────────────────────────────────
_REQUIRED_NLTK = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_resources() -> None:
    """
    Download any missing NLTK resources.
    Applies a macOS SSL workaround automatically when needed
    (Python 3.12 on macOS often has unverified SSL certs).
    """
    import ssl

    def _download(pkg_name: str) -> None:
        # First try normal download
        try:
            nltk.download(pkg_name, quiet=True)
            return
        except Exception:
            pass
        # Fallback: bypass SSL verification (macOS common issue)
        try:
            orig = ssl._create_default_https_context  # type: ignore[attr-defined]
            ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
            nltk.download(pkg_name, quiet=True)
            ssl._create_default_https_context = orig  # type: ignore[attr-defined]
        except Exception:
            pass  # Will surface as LookupError later if resource truly absent

    for path, name in _REQUIRED_NLTK:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            _download(name)


ensure_nltk_resources()

# Instantiate shared objects once
_STOP_WORDS: set[str] = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ──────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<[^>]+>")
_EXTRA_SPACE = re.compile(r"\s+")


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Replace HTTP/HTTPS and www URLs with a space."""
    return _URL_PATTERN.sub(" ", text)


def remove_html(text: str) -> str:
    """Strip HTML / XML tags from text."""
    return _HTML_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters, keeping spaces."""
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return _EXTRA_SPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Word-tokenise text using NLTK's punkt tokeniser."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop English stop-words and single-character tokens."""
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens using WordNetLemmatizer."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def clean_text(
    text: str,
    *,
    use_lemmatization: bool = True,
    remove_stops: bool = True,
) -> str:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw input text.
    use_lemmatization : bool
        Whether to apply lemmatization (default True).
    remove_stops : bool
        Whether to remove stop-words (default True).

    Returns
    -------
    str
        Cleaned, normalised text ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs
    text = remove_urls(text)

    # Step 3 – remove HTML tags
    text = remove_html(text)

    # Step 4 – remove punctuation
    text = remove_punctuation(text)

    # Step 5 – normalise whitespace
    text = normalize_whitespace(text)

    # Step 6 – tokenise
    tokens = tokenize(text)

    # Step 7 – remove stop-words (optional)
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # Step 8 – lemmatize (optional)
    if use_lemmatization:
        tokens = lemmatize(tokens)

    return " ".join(tokens)


# ──────────────────────────────────────────────
# Text statistics helpers
# ──────────────────────────────────────────────

def count_sentences(text: str) -> int:
    """Return the sentence count for a piece of raw text."""
    try:
        return len(sent_tokenize(text))
    except Exception:
        return text.count(".") + 1


def get_text_stats(raw_text: str, cleaned_text: Optional[str] = None) -> dict:
    """
    Compute basic statistics for raw (and optionally cleaned) text.

    Returns a dict with keys: word_count, sentence_count, char_count,
    cleaned_word_count (if cleaned_text is provided).
    """
    stats = {
        "word_count": len(raw_text.split()),
        "sentence_count": count_sentences(raw_text),
        "char_count": len(raw_text),
    }
    if cleaned_text is not None:
        stats["cleaned_word_count"] = len(cleaned_text.split())
    return stats
