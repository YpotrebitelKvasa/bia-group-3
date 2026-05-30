import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    SearchRequest, SearchResponse, SearchResult,
    AnswerRequest, AnswerResponse, HealthResponse
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="NQ RAG API", version="0.1.0")

# CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные ретриверы (загружаются при старте)
bm25: BM25Retriever = None
dense: DenseRetriever = None

@app.on_event("startup")
async def startup_event():
    """Загрузка индексов при старте сервиса."""
    global bm25, dense
    
    indices_dir = Path("data/indices")
    if not indices_dir.exists():
        logger.warning("Индексы не найдены. Запустите scripts/build_indices.py")
        return
    
    try:
        bm25 = BM25Retriever()
        bm25.load(str(indices_dir / "bm25"))
        logger.info("BM25 индекс загружен")
    except Exception as e:
        logger.error(f"Ошибка загрузки BM25: {e}")
    
    try:
        dense = DenseRetriever()
        dense.load(str(indices_dir / "faiss"))
        logger.info("FAISS индекс загружен")
    except Exception as e:
        logger.error(f"Ошибка загрузки FAISS: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    indices_ok = (bm25 is not None and bm25.is_built) or (dense is not None and dense.is_built)
    return HealthResponse(indices_loaded=indices_ok)

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Семантический поиск по корпусу документов."""
    start = time.perf_counter()
    
    # Выбор ретривера
    if req.method == "bm25":
        retriever = bm25
    elif req.method == "faiss":
        retriever = dense
    else:  # hybrid — пока используем FAISS как дефолт
        retriever = dense
    
    if not retriever or not retriever.is_built:
        raise HTTPException(status_code=503, detail="Индекс не загружен")
    
    # Поиск
    results = retriever.search(req.query, top_k=req.top_k)
    
    # Форматирование ответа
    formatted = [
        SearchResult(
            chunk_id=r.get("id", 0),
            title=r.get("title", ""),
            text=r.get("text", "")[:500] + "..." if len(r.get("text", "")) > 500 else r.get("text", ""),
            score=r.get("score", 0.0),
            url=r.get("url")
        )
        for r in results
    ]
    
    latency = (time.perf_counter() - start) * 1000
    logger.info(f"🔍 Search: '{req.query[:50]}...' → {len(formatted)} results in {latency:.1f}ms")
    
    return SearchResponse(
        query=req.query,
        results=formatted,
        method=req.method,
        latency_ms=round(latency, 2)
    )

@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    """RAG endpoint: поиск + (опционально) генерация ответа."""
    start = time.perf_counter()
    
    # Поиск контекста
    retriever = dense if dense and dense.is_built else bm25
    if not retriever:
        raise HTTPException(status_code=503, detail="Индекс не загружен")
    
    results = retriever.search(req.query, top_k=req.top_k)
    sources = [
        SearchResult(
            chunk_id=r.get("id", 0),
            title=r.get("title", ""),
            text=r.get("text", "")[:300] + "...",
            score=r.get("score", 0.0),
            url=r.get("url")
        )
        for r in results
    ]
    
    # Генерация ответа (заглушка — можно подключить LLM позже)
    if req.generate and results:
        # Простая эвристика: возвращаем первый абзац найденного чанка
        answer_text = results[0].get("text", "").split(". ")[0] + "."
    else:
        answer_text = "Найдите релевантный контекст в поле 'sources'."
    
    latency = (time.perf_counter() - start) * 1000
    logger.info(f"💬 Answer: '{req.query[:50]}...' in {latency:.1f}ms")
    
    return AnswerResponse(
        answer=answer_text,
        sources=sources,
        latency_ms=round(latency, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)