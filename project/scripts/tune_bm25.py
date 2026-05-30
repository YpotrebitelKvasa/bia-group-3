#!/usr/bin/env python3
"""
Скрипт для подбора оптимальных гиперпараметров BM25.
Запуск: python scripts/tune_bm25.py
"""
import json
import time
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np

# Добавляем корень проекта в path
import sys
sys.path.insert(0, str(Path.cwd()))

from src.retrieval.bm25 import BM25Retriever
from src.utils.metrics import normalize_answer
from datasets import load_dataset

def load_benchmark(num_queries: int = 200):
    """Загружает бенчмарк вопросов с золотыми ответами."""
    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True
    )
    
    benchmark = []
    for item in list(dataset.take(num_queries)):
        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            continue
            
        annotations = item.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        
        answers = []
        for sa in short_answers:
            texts = sa.get("text", [])
            if texts:
                answers.extend([t for t in texts if isinstance(t, str) and t.strip()])
        
        if answers:
            benchmark.append({"query": q, "gold_answers": answers})
    
    return benchmark

def evaluate_bm25_config(chunks, benchmark, k1: float, b: float, top_k: int = 5):
    """Оценивает BM25 с заданными гиперпараметрами."""
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.build_index(chunks)
    
    recall_scores = []
    latencies = []
    
    for item in benchmark:
        query = item["query"]
        gold = item["gold_answers"]
        
        start = time.perf_counter()
        results = retriever.search(query, top_k=top_k)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Проверка вхождения ответа
        combined_text = " ".join([r["text"] for r in results])
        found = any(normalize_answer(a) in normalize_answer(combined_text) for a in gold)
        recall_scores.append(1.0 if found else 0.0)
    
    return {
        "k1": k1,
        "b": b,
        "recall@5": float(np.mean(recall_scores)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95))
    }

def main():
    print("Tuning гиперпараметров BM25")
    print("=" * 60)
    
    # Загрузка данных
    print("\nЗагрузка чанков...")
    chunks_path = Path("data/processed/chunks.jsonl")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Загружено {len(chunks)} чанков")
    
    print("\nЗагрузка бенчмарка...")
    benchmark = load_benchmark(num_queries=100)  # 100 для быстрого tuning
    print(f"Загружено {len(benchmark)} вопросов")
    
    # Сетка гиперпараметров
    k1_values = [1.0, 1.2, 1.5, 1.8, 2.0]
    b_values = [0.5, 0.65, 0.75, 0.9]
    
    print(f"\n🔍 Перебор конфигураций: {len(k1_values)} x {len(b_values)} = {len(k1_values)*len(b_values)}")
    
    results = []
    best_config = None
    best_recall = 0.0
    
    for k1, b in itertools.product(k1_values, b_values):
        print(f"\n  Testing k1={k1}, b={b}...", end=" ", flush=True)
        start = time.time()
        
        config_result = evaluate_bm25_config(chunks, benchmark, k1, b)
        elapsed = time.time() - start
        
        results.append(config_result)
        print(f"Recall@5={config_result['recall@5']:.3f} ({elapsed:.1f}s)")
        
        if config_result["recall@5"] > best_recall:
            best_recall = config_result["recall@5"]
            best_config = config_result
    
    # Сортировка по Recall@5
    results.sort(key=lambda x: x["recall@5"], reverse=True)
    
    # Сохранение результатов
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = experiments_dir / f"bm25_tuning_{timestamp}.json"
    
    experiment_log = {
        "experiment_type": "bm25_hyperparameter_tuning",
        "timestamp": timestamp,
        "num_chunks": len(chunks),
        "num_queries": len(benchmark),
        "grid_search": {
            "k1_values": k1_values,
            "b_values": b_values
        },
        "results": results,
        "best_config": best_config
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Лучшая конфигурация:")
    print(f"k1 = {best_config['k1']}")
    print(f"b  = {best_config['b']}")
    print(f"Recall@5 = {best_config['recall@5']:.3f}")
    print(f"Latency p50 = {best_config['latency_p50_ms']:.1f}ms")
    print(f"\nРезультаты сохранены в {output_file}")
    
    # Вывод топ-5 конфигураций
    print("\n Топ-5 конфигураций:")
    print(f"{'Rank':<5} {'k1':<8} {'b':<8} {'Recall@5':<12} {'Latency p50':<12}")
    print("-" * 50)
    for i, res in enumerate(results[:5], 1):
        print(f"{i:<5} {res['k1']:<8} {res['b']:<8} {res['recall@5']:<12.3f} {res['latency_p50_ms']:<12.1f}")

if __name__ == "__main__":
    main()