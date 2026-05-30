"""
Dense retriever на основе FAISS и sentence-transformers.
Реализация соответствует паттернам семинара S14.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from .base import Retriever, EmbeddingBackend


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Бэкенд для эмбеддингов через sentence-transformers.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        device: str = "cpu"
    ):
        """
        Args:
            model_name: Название модели из HuggingFace
            device: "cpu" или "cuda"
        """
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self.device = device
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Генерирует эмбеддинги с опцией нормализации для косинусного сходства.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype("float32")
    
    @property
    def dimension(self) -> int:
        return self._dimension


class DenseRetriever(Retriever):
    """
    Ретривер на основе векторного поиска через FAISS.
    Использует косинусное сходство (IndexFlatIP) по умолчанию.
    """
    
    def __init__(
        self, 
        backend: Optional[EmbeddingBackend] = None,
        index_type: str = "FlatIP"  # "FlatIP" для косинуса, "FlatL2" для евклида
    ):
        """
        Args:
            backend: Эмбеддинг-бэкенд (по умолчанию SentenceTransformerBackend)
            index_type: Тип FAISS индекса
        """
        self.backend = backend or SentenceTransformerBackend()
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self._is_built = False
    
    def build_index(
        self, 
        chunks: List[Dict], 
        batch_size: int = 64,
        **kwargs
    ) -> None:
        """
        Построение FAISS индекса из чанков.
        
        Args:
            chunks: Список чанков с ключом "text"
            batch_size: Размер батча для эмбеддинга
        """
        if not chunks:
            raise ValueError("Пустой список чанков")
        
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Генерация эмбеддингов
        vectors = self.backend.encode(texts, batch_size=batch_size)
        
        # Создание FAISS индекса
        dimension = vectors.shape[1]
        
        if self.index_type == "FlatIP":
            # Косинусное сходство (требует нормализации векторов)
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(vectors)
        elif self.index_type == "FlatL2":
            # Евклидово расстояние
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Неизвестный тип индекса: {self.index_type}")
        
        # Добавление векторов в индекс
        self.index.add(vectors)
        self._is_built = True
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        **kwargs
    ) -> List[Dict]:
        """
        Поиск топ-k релевантных чанков.
        
        Returns:
            Список чанков с добавленным полем "score"
        """
        if not self._is_built or self.index is None:
            raise RuntimeError("Индекс не построен. Вызовите build_index()")
        
        # Эмбеддинг запроса
        query_vec = self.backend.encode([query], normalize=(self.index_type == "FlatIP"))
        
        # Поиск в индексе
        scores, indices = self.index.search(query_vec, top_k)
        
        # Формирование результатов
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):  # FAISS возвращает -1 для пустых слотов
                continue
            
            result = self.chunks[idx].copy()
            result["score"] = float(score)
            result["retrieval_method"] = "dense"
            results.append(result)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Сохранение индекса и метаданных.
        
        Args:
            path: Путь без расширения (добавится .faiss и .meta.json)
        """
        if not self._is_built:
            raise RuntimeError("Нечего сохранять: индекс не построен")
        
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение FAISS индекса
        faiss.write_index(self.index, str(output.with_suffix(".faiss")))
        
        # Сохранение метаданных (чанки)
        meta = {
            "chunks": self.chunks,
            "index_type": self.index_type,
            "backend_model": getattr(self.backend, "model", None).__class__.__name__ if hasattr(self.backend, "model") else None,
            "dimension": self.backend.dimension
        }
        with open(output.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """
        Загрузка индекса и метаданных.
        
        Args:
            path: Путь без расширения
        """
        index_path = Path(path).with_suffix(".faiss")
        meta_path = Path(path).with_suffix(".meta.json")
        
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Файлы индекса не найдены: {path}")
        
        # Загрузка FAISS индекса
        self.index = faiss.read_index(str(index_path))
        
        # Загрузка метаданных
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        self.chunks = meta["chunks"]
        self.index_type = meta["index_type"]
        self._is_built = True
    
    @property
    def is_built(self) -> bool:
        return self._is_built