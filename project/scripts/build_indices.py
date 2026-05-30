#!/usr/bin/env python3
"""
Скрипт для построения индексов BM25 и FAISS из подготовленных чанков.
Запуск: python scripts/build_indices.py
"""
import json
import logging
import time
from pathlib import Path
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_chunks(path: str) -> list:
    """Загружает чанки из JSONL файла."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunks.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга строки {line_num}: {e}")
                continue
    return chunks


def main():
    # Пути
    chunks_path = Path("data/processed/chunks.jsonl")
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    if not chunks_path.exists():
        logger.error(f"Файл чанков не найден: {chunks_path}")
        logger.info("Сначала запустите: python -m src.data.loader")
        return
    
    # Загрузка чанков
    logger.info(f"Загрузка чанков из {chunks_path}...")
    start_load = time.time()
    chunks = load_chunks(str(chunks_path))
    logger.info(f"Загружено {len(chunks)} чанков за {time.time() - start_load:.1f}с")
    
    # === BM25 индекс ===
    logger.info("Построение BM25 индекса...")
    start_bm25 = time.time()
    
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(chunks)
    bm25_path = indices_dir / "bm25"
    bm25.save(str(bm25_path))
    
    logger.info(f"BM25 индекс сохранён: {bm25_path}.pkl")
    logger.info(f"Время: {time.time() - start_bm25:.1f}с")
    
    # === FAISS индекс ===
    logger.info("Построение FAISS индекса (это займёт 2-5 минут)...")
    start_faiss = time.time()
    
    dense = DenseRetriever(index_type="FlatIP")  # косинусное сходство
    dense.build_index(chunks, batch_size=64)
    faiss_path = indices_dir / "faiss"
    dense.save(str(faiss_path))
    
    logger.info(f"FAISS индекс сохранён: {faiss_path}.faiss + .meta.json")
    logger.info(f"Время: {time.time() - start_faiss:.1f}с")
    
    # === Итог ===
    total_time = time.time() - start_load
    logger.info(f"Готово! Все индексы в {indices_dir}/")
    logger.info(f" Общее время: {total_time:.1f}с")
    
    # Информация для отчёта
    logger.info("\nСтатистика для report.md:")
    logger.info(f"   - Чанков: {len(chunks)}")
    logger.info(f"   - Размер BM25: {bm25_path.with_suffix('.pkl').stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"   - Размер FAISS: {faiss_path.with_suffix('.faiss').stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()