"""
Абстрактные интерфейсы для компонентов retrieval-системы.
Паттерн из семинара S14: абстракция бэкендов для лёгкого переключения.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class EmbeddingBackend(ABC):
    """
    Абстрактный интерфейс для генерации эмбеддингов.
    Позволяет легко менять модель (sentence-transformers, OpenAI, etc.)
    """
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Генерирует эмбеддинги для списка текстов.
        
        Args:
            texts: Список строк для эмбеддинга
            **kwargs: Дополнительные параметры (batch_size, device, etc.)
            
        Returns:
            numpy.array shape (n_texts, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность эмбеддинга."""
        pass


class Retriever(ABC):
    """
    Абстрактный интерфейс для ретривера (поискового модуля).
    """
    
    @abstractmethod
    def build_index(self, chunks: List[Dict], **kwargs) -> None:
        """
        Построение индекса из списка чанков.
        
        Args:
            chunks: Список словарей с ключом "text"
            **kwargs: Дополнительные параметры индексации
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск топ-k релевантных чанков по запросу.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список словарей с чанками и скором релевантности
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Сохранение индекса на диск."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Загрузка индекса с диска."""
        pass