"""
Test data cleaning function.
"""

from src.data.clean_transform import clean_text


def test_clean_text_stopwords_and_not():
    """Verify if not stopword is kept."""
    raw_text = "This is not a good example"
    expected = "not good example"
    assert clean_text(raw_text) == expected


def test_clean_text_only_stopwords():
    """Test stopwords cleaning"""
    raw_text = "this is a very common sentence"
    expected = "common sentence"
    assert clean_text(raw_text) == expected


def test_clean_text_lemmatization():
    """Test lemmatization."""
    raw_text = "Dogs are running faster"
    expected = "dog run faster"
    assert clean_text(raw_text) == expected


def test_clean_text_basic():
    """Test whether text cleaning works on a simple case."""
    raw_text = "Hello World! This is a test."
    expected = "hello world test"
    assert clean_text(raw_text).strip() == expected


def test_clean_text_empty_string():
    """Test empty string cleaning."""
    raw_text = ""
    expected = ""
    assert clean_text(raw_text) == expected


def test_clean_text_special_chars():
    """Test with special characters."""
    raw_text = "!!!Wow??? @Amazon"
    expected = "wow amazon"
    assert clean_text(raw_text).strip() == expected


def test_clean_text_numbers():
    """Test with numbers."""
    raw_text = "Version 2.0 is better than version 1"
    expected = "version 2 0 good version 1"
    assert clean_text(raw_text) == expected


def test_clean_text_idempotent():
    """Test the idempotence of the cleaning function."""
    raw_text = "This is NOT a Test!"
    once = clean_text(raw_text)
    twice = clean_text(once)
    assert once == twice


def test_clean_text_mixed_case_and_punctuation():
    """Test complex punctuation"""
    raw_text = "Well... THIS, is a—strange!!! sentence???"
    expected = "well strange sentence"
    assert clean_text(raw_text) == expected
