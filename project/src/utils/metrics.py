"""
Метрики оценки качества retrieval-систем.
Соответствует методологии семинара S14.
"""
import re
import string
from typing import List, Dict, Optional
import numpy as np
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    NQ-style нормализация ответа: lower, remove punctuation/articles, fix whitespace.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Вычисляет максимальный F1 по токенам среди всех золотых ответов.
    
    Args:
        prediction: Предсказанный ответ (или текст чанка)
        ground_truths: Список золотых ответов
        
    Returns:
        F1-score в диапазоне [0, 1]
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            continue
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    
    return best_f1


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> int:
    """
    Hit@K: 1 если хотя бы один релевантный документ в топ-k.
    
    Args:
        retrieved_ids: Список ID найденных документов (в порядке ранжирования)
        relevant_ids: Список ID релевантных документов
        k: Порог топ-k
        
    Returns:
        1 или 0
    """
    return int(any(doc_id in retrieved_ids[:k] for doc_id in relevant_ids))


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@K: доля релевантных документов, найденных в топ-k.
    
    Args:
        retrieved_ids: Список ID найденных документов
        relevant_ids: Список ID релевантных документов
        k: Порог топ-k
        
    Returns:
        Recall в диапазоне [0, 1]
    """
    if not relevant_ids:
        return 1.0  # Нет релевантных — тривиальный случай
    
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    MRR@K (Mean Reciprocal Rank): обратная позиция первого релевантного.
    
    Args:
        retrieved_ids: Список ID найденных документов (в порядке ранжирования)
        relevant_ids: Список ID релевантных документов
        k: Порог топ-k
        
    Returns:
        MRR в диапазоне [0, 1]
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retriever,
    queries: List[Dict],
    k_values: List[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Полная оценка ретривера на наборе вопросов.
    
    Args:
        retriever: Объект, реализующий интерфейс Retriever
        queries: Список вопросов с полями:
            - "query": текст запроса
            - "relevant_ids": список ID релевантных документов
        k_values: Список значений k для метрик (по умолчанию [1, 3, 5, 10])
        
    Returns:
        Словарь метрик: {k: {"hit@k": ..., "recall@k": ..., "mrr@k": ...}}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    # Инициализация результатов
    results = {k: {"hit": [], "recall": [], "mrr": []} for k in k_values}
    
    for q in queries:
        query_text = q["query"]
        relevant_ids = q.get("relevant_ids", [])
        
        # Поиск
        retrieved = retriever.search(query_text, top_k=max(k_values))
        retrieved_ids = [r.get("id") for r in retrieved if r.get("id") is not None]
        
        # Расчёт метрик для каждого k
        for k in k_values:
            results[k]["hit"].append(hit_at_k(retrieved_ids, relevant_ids, k))
            results[k]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            results[k]["mrr"].append(mrr_at_k(retrieved_ids, relevant_ids, k))
    
    # Агрегация: среднее по всем вопросам
    aggregated = {}
    for k in k_values:
        aggregated[f"hit@{k}"] = float(np.mean(results[k]["hit"]))
        aggregated[f"recall@{k}"] = float(np.mean(results[k]["recall"]))
        aggregated[f"mrr@{k}"] = float(np.mean(results[k]["mrr"]))
    
    return aggregated