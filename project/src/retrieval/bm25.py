"""
BM25 retriever на основе rank-bm25.
Классический вероятностный ранжировщик для сравнения с dense-подходом.
"""
import json
import pickle
import re
from pathlib import Path
from typing import Optional, List, Dict
from rank_bm25 import BM25Okapi
from .base import Retriever


def simple_tokenize(text: str) -> List[str]:
    """
    Простая токенизация: lower + split by non-alphanumeric.
    Можно заменить на более сложную (nltk, spacy) при необходимости.
    """
    text = text.lower()
    # Оставляем только буквы, цифры и пробелы
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever(Retriever):
    """
    Ретривер на основе алгоритма BM25 (Best Matching 25).
    
    Параметры:
    - k1: контролирует насыщение частоты термина (обычно 1.2-2.0)
    - b: контролирует нормализацию по длине документа (0.0-1.0)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Параметр насыщения частоты термина
            b: Параметр нормализации длины документа
        """
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []
        self._is_built = False
    
    def build_index(
        self, 
        chunks: List[Dict], 
        tokenizer=None,
        **kwargs
    ) -> None:
        """
        Построение BM25 индекса.
        
        Args:
            chunks: Список чанков с ключом "text"
            tokenizer: Функция токенизации (по умолчанию simple_tokenize)
        """
        if not chunks:
            raise ValueError("Пустой список чанков")
        
        self.chunks = chunks
        tokenizer = tokenizer or simple_tokenize
        
        # Токенизация корпуса
        self._tokenized_corpus = [tokenizer(chunk["text"]) for chunk in chunks]
        
        # Создание BM25 индекса
        self.bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        self._is_built = True
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        tokenizer=None,
        **kwargs
    ) -> List[Dict]:
        """
        Поиск топ-k релевантных чанков.
        
        Returns:
            Список чанков с добавленным полем "score"
        """
        if not self._is_built or self.bm25 is None:
            raise RuntimeError("Индекс не построен. Вызовите build_index()")
        
        tokenizer = tokenizer or simple_tokenize
        query_tokens = tokenizer(query)
        
        # Получение скоров для всех документов
        scores = self.bm25.get_scores(query_tokens)
        
        # Топ-k по убыванию скора
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result["score"] = float(scores[idx])
            result["retrieval_method"] = "bm25"
            results.append(result)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Сохранение индекса через pickle.
        
        Args:
            path: Путь к файлу (добавится .pkl)
        """
        if not self._is_built:
            raise RuntimeError("Нечего сохранять: индекс не построен")
        
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self.k1,
            "b": self.b
        }
        
        with open(output.with_suffix(".pkl"), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, path: str) -> None:
        """
        Загрузка индекса из pickle.
        
        Args:
            path: Путь к файлу .pkl
        """
        input_path = Path(path)
        if not input_path.suffix:
            input_path = input_path.with_suffix(".pkl")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {input_path}")
        
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self._tokenized_corpus = data.get("tokenized_corpus", [])
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self._is_built = True
    
    @property
    def is_built(self) -> bool:
        return self._is_built