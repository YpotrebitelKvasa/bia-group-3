from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# новые тесты

def test_has_constant_columns():
    #Проверяет эвристику has_constant_columns.
    #Тестирует DataFrame с константной колонкой (все одинаковые значения).
    
    # DataFrame с одной константной колонкой
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "status": ["active", "active", "active", "active", "active"],  # все одинаковые
            "value": [10.5, 20.3, 15.1, 30.2, 25.4],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    # Проверка что флаг выставлен
    assert flags["has_constant_columns"] is True

    # Проверка что константный столбец в списке
    assert "status" in flags["constant_columns_list"]
    assert len(flags["constant_columns_list"]) == 1

    # Проверка что качество понизилось из-за константы
    assert flags["quality_score"] < 1.0


def test_has_high_cardinality_categoricals():
    #Проверка эвристику has_high_cardinality_categoricals.
    #Тестирует DataFrame с категориальной колонкой, где много уникальных значений (90%+).
    
    # DataFrame с категориальной колонкой с высокой cardinality
    # 100 строк, 92 уникальных значения в category → 92% уникальности
    df = pd.DataFrame(
        {
            "id": range(100),
            "category": [f"cat_{i}" for i in range(92)] + ["cat_91", "cat_91", "cat_91", "cat_91", "cat_91", "cat_91", "cat_91", "cat_91"],  # 92 уникальные из 100
            "value": [i * 1.5 for i in range(100)],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Проверка что флаг выставлен
    assert flags["has_high_cardinality_categoricals"] is True

    # Проверка что проблемная категория в списке
    assert "category" in flags["high_cardinality_categoricals"]

    # Проверка что качество понизилось
    assert flags["quality_score"] < 1.0