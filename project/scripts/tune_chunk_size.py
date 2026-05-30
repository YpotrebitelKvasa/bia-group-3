#!/usr/bin/env python3
"""
Скрипт для подбора оптимального размера чанков.
Запуск: python scripts/tune_chunk_size.py
"""
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import sys
sys.path.insert(0, str(Path.cwd()))

from src.data.chunker import clean_html, chunk_text
from src.retrieval.dense import DenseRetriever
from datasets import load_dataset

def evaluate_chunk_size_with_answers(dataset_sample, chunk_size: int, overlap: int, top_k: int = 5):
    """Оценивает качество по вхождению short_answer в retrieved chunks."""
    from src.utils.metrics import normalize_answer
    
    all_chunks = []
    chunk_id = 0
    
    for item in dataset_sample:
        html = item.get("document", {}).get("html", "")
        title = item.get("document", {}).get("title", "")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        
        clean = clean_html(html)
        if not clean or len(clean) < 50:
            continue
        
        text_chunks = chunk_text(clean, chunk_size=chunk_size, overlap=overlap, min_chunk_chars=50)
        
        for text_chunk in text_chunks:
            all_chunks.append({
                "id": chunk_id,
                "title": title,
                "text": text_chunk,
                "url": url
            })
            chunk_id += 1
    
    # Построение индекса
    retriever = DenseRetriever()
    retriever.build_index(all_chunks, batch_size=64)
    
    # Бенчмарк с короткими ответами
    benchmark = []
    for item in dataset_sample:
        raw_q = item.get("question", "")
        query_str = raw_q.get("text", "") if isinstance(raw_q, dict) else str(raw_q)
        if not query_str.strip():
            continue
        
        # Извлечение short_answers
        annotations = item.get("annotations", {})
        short_answers_list = annotations.get("short_answers", [])
        answers = []
        for sa in short_answers_list:
            texts = sa.get("text", [])
            if texts:
                answers.extend([t for t in texts if isinstance(t, str) and t.strip()])
        
        if answers:
            benchmark.append({"query": query_str.strip(), "gold_answers": answers})
    
    if not benchmark:
        return {"chunk_size": chunk_size, "recall@5": 0.0, "num_chunks": len(all_chunks)}
    
    # Оценка
    recall_scores = []
    latencies = []
    
    for q in benchmark[:50]:  # 50 для скорости
        query = q["query"]
        gold = q["gold_answers"]
        
        start = time.perf_counter()
        results = retriever.search(query, top_k=top_k)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Проверка вхождения ответа в текст
        combined_text = " ".join([r["text"] for r in results])
        found = any(normalize_answer(a) in normalize_answer(combined_text) for a in gold)
        recall_scores.append(1.0 if found else 0.0)
    
    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(all_chunks),
        "recall@5": float(np.mean(recall_scores)),
        "latency_p50_ms": float(np.median(latencies))
    }

def main():
    print("Tuning размера чанков для FAISS")
    print("=" * 60)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True
    )
    sample = list(dataset.take(200))  # 200 статей для теста
    print(f"Загружено {len(sample)} статей")
    
    # Сетка параметров
    chunk_sizes = [200, 300, 400, 500]
    overlap = 30  # Фиксированный
    
    print(f"\nТестирование размеров чанков: {chunk_sizes}")
    
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"\n  Testing chunk_size={chunk_size}...", end=" ", flush=True)
        start = time.time()
        
        config_result = evaluate_chunk_size_with_answers(sample, chunk_size, overlap)
        elapsed = time.time() - start
        
        results.append(config_result)
        print(f"Recall@5={config_result['recall@5']:.3f}, chunks={config_result['num_chunks']} ({elapsed:.1f}s)")
    
    # Сортировка
    results.sort(key=lambda x: x["recall@5"], reverse=True)
    
    # Сохранение
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = experiments_dir / f"chunk_size_tuning_{timestamp}.json"
    
    experiment_log = {
        "experiment_type": "chunk_size_tuning",
        "timestamp": timestamp,
        "num_articles": len(sample),
        "results": results,
        "best_config": results[0]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Лучший размер чанка: {results[0]['chunk_size']} токенов")
    print(f"Recall@5 = {results[0]['recall@5']:.3f}")
    print(f"Количество чанков: {results[0]['num_chunks']}")
    
    print("\nВсе конфигурации:")
    print(f"{'Chunk Size':<12} {'Recall@5':<12} {'Num Chunks':<12} {'Latency p50':<12}")
    print("-" * 50)
    for res in results:
        print(f"{res['chunk_size']:<12} {res['recall@5']:<12.3f} {res['num_chunks']:<12} {res['latency_p50_ms']:<12.1f}")
    
    print(f"\nРезультаты сохранены в {output_file}")

if __name__ == "__main__":
    main()