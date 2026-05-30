"""Smoke-тесты для API."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Проверка health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "indices_loaded" in data

@patch("src.api.main.dense")
@patch("src.api.main.bm25")
def test_search_endpoint(mock_bm25, mock_dense):
    """Проверка поиска (с моками индексов)."""
    # Настраиваем моки: притворяемся, что индексы загружены
    mock_bm25.is_built = True
    mock_dense.is_built = True
    
    # Мокаем результат поиска для FAISS
    mock_dense.search.return_value = [
        {
            "id": 123,
            "title": "Test Article",
            "text": "This is a test chunk about the query topic.",
            "score": 0.95,
            "url": "https://example.com"
        }
    ]
    
    response = client.post(
        "/search",
        json={"query": "test query", "method": "faiss", "top_k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert "results" in data
    assert "latency_ms" in data
    assert len(data["results"]) <= 2
    assert data["results"][0]["title"] == "Test Article"

@patch("src.api.main.dense")
@patch("src.api.main.bm25")
def test_search_bm25(mock_bm25, mock_dense):
    """Проверка BM25 метода (с моками)."""
    # Настраиваем моки
    mock_bm25.is_built = True
    mock_dense.is_built = True
    
    # Мокаем результат поиска для BM25
    mock_bm25.search.return_value = [
        {
            "id": 456,
            "title": "Telephone History",
            "text": "Alexander Graham Bell invented the telephone...",
            "score": 0.88,
            "url": "https://en.wikipedia.org/wiki/Telephone"
        }
    ]
    
    response = client.post(
        "/search",
        json={"query": "who invented telephone", "method": "bm25", "top_k": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "bm25"
    assert len(data["results"]) == 1
    assert "Bell" in data["results"][0]["text"]