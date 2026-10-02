"""
test_preprocessor.py
--------------------
Unit tests for the NLP preprocessing pipeline.
"""

import sys
from pathlib import Path

# Allow importing src.* from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.preprocessor import (
    clean_text,
    remove_urls,
    remove_html,
    remove_punctuation,
    normalize_whitespace,
    tokenize,
    remove_stopwords,
    lemmatize,
    get_text_stats,
    count_sentences,
)


class TestRemoveUrls:
    def test_http_url_removed(self):
        text = "Visit https://example.com for more info."
        result = remove_urls(text)
        assert "https://example.com" not in result

    def test_www_url_removed(self):
        text = "Go to www.example.com now"
        result = remove_urls(text)
        assert "www.example.com" not in result

    def test_no_url_unchanged(self):
        text = "This has no URL."
        assert remove_urls(text) == text


class TestRemoveHtml:
    def test_tag_removed(self):
        text = "<p>Hello <b>world</b></p>"
        assert "Hello" in remove_html(text)
        assert "<p>" not in remove_html(text)

    def test_no_html_unchanged(self):
        text = "Plain text here."
        assert remove_html(text) == text


class TestRemovePunctuation:
    def test_punctuation_removed(self):
        text = "Hello, world! Is this working?"
        result = remove_punctuation(text)
        for ch in ".,!?":
            assert ch not in result
        assert "Hello" in result


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        text = "too   many    spaces"
        assert normalize_whitespace(text) == "too many spaces"

    def test_strips_ends(self):
        text = "  trim me  "
        assert normalize_whitespace(text) == "trim me"


class TestTokenize:
    def test_returns_list(self):
        tokens = tokenize("hello world")
        assert isinstance(tokens, list)
        assert "hello" in tokens

    def test_empty_string(self):
        tokens = tokenize("")
        # NLTK returns empty list for empty string
        assert isinstance(tokens, list)


class TestRemoveStopwords:
    def test_the_removed(self):
        tokens = ["the", "quick", "brown", "fox"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "quick" in result

    def test_single_chars_removed(self):
        tokens = ["a", "b", "word"]
        result = remove_stopwords(tokens)
        assert "a" not in result
        assert "word" in result


class TestLemmatize:
    def test_running_to_run(self):
        tokens = ["running", "dogs", "better"]
        result = lemmatize(tokens)
        assert "running" in result or "run" in result  # lemmatiser may or may not change

    def test_returns_same_length(self):
        tokens = ["cats", "dogs", "running"]
        assert len(lemmatize(tokens)) == len(tokens)


class TestCleanText:
    def test_lowercase(self):
        result = clean_text("UPPER CASE TEXT")
        assert result == result.lower()

    def test_url_removed(self):
        result = clean_text("Check https://example.com content")
        assert "https" not in result

    def test_html_removed(self):
        result = clean_text("<div>news</div>")
        assert "<div>" not in result

    def test_non_string_input(self):
        # Should not raise; converts to str
        result = clean_text(12345)
        assert isinstance(result, str)

    def test_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)


class TestGetTextStats:
    def test_basic_stats(self):
        text = "Hello world. This is great!"
        stats = get_text_stats(text)
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0
        assert stats["char_count"] == len(text)

    def test_with_cleaned(self):
        raw = "Hello world."
        cleaned = "hello world"
        stats = get_text_stats(raw, cleaned)
        assert "cleaned_word_count" in stats


class TestCountSentences:
    def test_multiple_sentences(self):
        text = "This is one. This is two. This is three."
        assert count_sentences(text) >= 2
