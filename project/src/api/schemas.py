from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Поисковый запрос")
    method: Literal["bm25", "faiss", "hybrid"] = Field(default="faiss")
    top_k: int = Field(default=5, ge=1, le=20)

class SearchResult(BaseModel):
    chunk_id: int
    title: str
    text: str
    score: float
    url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    method: str
    latency_ms: float

class AnswerRequest(BaseModel):
    query: str
    generate: bool = Field(default=False)
    top_k: int = Field(default=5, ge=1, le=10)

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    indices_loaded: bool