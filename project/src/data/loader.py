import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from .chunker import clean_html, chunk_text
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_nq_chunks(sample_size: int = 5000, output_dir: str = "./data/processed") -> Path:
    """
    Загружает NQ в режиме STREAMING.
    """
    output_path = Path(output_dir) / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Загружаю natural_questions (Streaming mode, dev split)...")
    
    # 1. Включаем streaming=True — данные качаются потоком
    dataset = load_dataset(
        "google-research-datasets/natural_questions", 
        split="validation", 
        streaming=True
    )  
    
    # 2. Берём только первые N элементов
    dataset = dataset.take(sample_size)
    
    chunks = []
    chunk_id = 0
    
    # 3. Итерируемся (данные подгружаются по мере чтения)
    logger.info(f" Обработка {sample_size} статей...")
    for item in tqdm(dataset, total=sample_size, desc="Парсинг"):
        title = item["document"]["title"]
        html = item["document"]["html"]
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        
        try:
            clean_text = clean_html(html)
            if not clean_text or len(clean_text) < 100:
                continue
                
            text_chunks = chunk_text(clean_text)
            for chunk in text_chunks:
                chunks.append({
                    "id": chunk_id,
                    "title": title,
                    "text": chunk,
                    "url": url,
                    "metadata": {"source": "natural_questions", "split": "dev"}
                })
                chunk_id += 1
        except Exception as e:
            logger.warning(f" Пропуск {title}: {e}")
            
    # Сохраняем результат
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logger.info(f"Готово! Сохранено {len(chunks)} чанков в {output_path}")
    return output_path

if __name__ == "__main__":
    load_nq_chunks(sample_size=5000)