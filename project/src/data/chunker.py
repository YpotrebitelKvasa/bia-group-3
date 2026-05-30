"""
Модуль для предобработки текста: очистка HTML и разбиение на чанки
"""
import re
from typing import List
from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """
    Удаляет HTML-теги, скрипты, стили, навигацию.
    Возвращает чистый текст, готовый к чанкингу.
    
    Args:
        html: Исходный HTML-документ
        
    Returns:
        Очищенный текст
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Удаляем шумные элементы
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "iframe", "aside"]):
        tag.decompose()
    
    # Извлекаем текст
    text = soup.get_text(separator=" ", strip=True)
    
    # Нормализация: убираем лишние пробелы, переносы
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def chunk_text(
    text: str, 
    chunk_size: int = 300, 
    overlap: int = 30,
    min_chunk_chars: int = 50
) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием (overlap).
    
    - chunk_size: количество токенов (слов) в чанке
    - overlap: количество общих токенов между соседними чанками
    - Это предотвращает разрыв смысла на границах
    
    Args:
        text: Очищенный текст
        chunk_size: Размер чанка в токенах (словах)
        overlap: Перекрытие между чанками
        min_chunk_chars: Минимальная длина чанка в символах для фильтрации
        
    Returns:
        Список чанков (строк)
    """
    if chunk_size <= 0 or overlap >= chunk_size:
        raise ValueError(f"Некорректные параметры: chunk_size={chunk_size}, overlap={overlap}")
    
    # Токенизация: простое разбиение по пробелам
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = chunk_size - overlap
    
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        
        if not chunk_words:
            continue
            
        chunk = " ".join(chunk_words)
        
        # Фильтруем слишком короткие чанки
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)
        
        # Если дошли до конца текста — выходим
        if end >= len(words):
            break
    
    return chunks


def chunk_document(
    title: str,
    html: str,
    url: str,
    chunk_size: int = 300,
    overlap: int = 30,
    min_chunk_chars: int = 50
) -> List[dict]:
    """
    Полный пайплайн: HTML → чистый текст → чанки с метаданными.
    
    Args:
        title: Заголовок статьи
        html: Исходный HTML
        url: URL статьи
        chunk_size: Размер чанка
        overlap: Перекрытие
        min_chunk_chars: Мин. длина чанка
        
    Returns:
        Список словарей с чанками и метаданными
    """
    clean = clean_html(html)
    if not clean or len(clean) < min_chunk_chars:
        return []
    
    text_chunks = chunk_text(clean, chunk_size, overlap, min_chunk_chars)
    
    results = []
    for idx, chunk_text in enumerate(text_chunks):
        results.append({
            "text": chunk_text,
            "title": title,
            "url": url,
            "chunk_idx": idx,
            "total_chunks": len(text_chunks),
            "metadata": {
                "source": "natural_questions",
                "split": "validation"
            }
        })
    
    return results