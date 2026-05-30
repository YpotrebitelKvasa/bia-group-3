"""Юнит-тесты для retrieval модулей."""
import pytest
from src.utils.metrics import normalize_answer, token_f1

def test_normalize_answer():
    """Проверка нормализации ответов."""
    assert normalize_answer("The 16th President") == "16th president"
    assert normalize_answer("Abraham Lincoln!") == "abraham lincoln"
    assert normalize_answer("  extra   spaces  ") == "extra spaces"

def test_token_f1_perfect_match():
    """F1 для идеального совпадения."""
    f1 = token_f1("abraham lincoln", ["abraham lincoln"])
    assert f1 == 1.0

def test_token_f1_partial_match():
    """F1 для частичного совпадения."""
    f1 = token_f1("abraham lincoln was president", ["abraham lincoln"])
    assert 0.0 < f1 < 1.0

def test_token_f1_no_match():
    """F1 для несовпадения."""
    f1 = token_f1("george washington", ["abraham lincoln"])
    assert f1 == 0.0