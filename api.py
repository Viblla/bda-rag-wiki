# api.py - FastAPI backend for Wiki Whatiz RAG
"""
Run with: uvicorn api:app --reload --port 8000

This serves both the API and the static frontend.
Open http://localhost:8000 in your browser.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import time

from src.retrieval import (
    bm25_retrieve,
    vector_retrieve,
    hybrid_retrieve,
    rerank,
    load_bm25,
    load_faiss,
    load_embedder,
    load_reranker,
    get_device_info,
)
from src.rag_answer import rag_answer_timed, rag_answer_iterative_timed

# Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "web" / "public"

app = FastAPI(
    title="Wiki Whatiz RAG API",
    description="RAG-powered Wikipedia Q&A API",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    retriever_mode: str = "Hybrid"
    use_rerank: bool = True
    use_iterative: bool = True
    bm25_k: int = 60
    vec_k: int = 60
    merge_k: int = 30
    rerank_k: int = 5
    refine_n: int = 3
    sarcastic_mode: bool = False


class Source(BaseModel):
    text: str
    chunk_id: Any
    doc_id: str
    source: str
    score: float
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    refined_queries: Optional[List[str]] = None
    timings: Dict[str, float]
    total_time: float


@app.get("/api")
def api_root():
    return {"message": "Wiki Whatiz RAG API", "status": "running", "docs": "/docs"}


@app.get("/status")
def status():
    """Get system status including GPU info."""
    device_info = get_device_info()
    return {
        "status": "ok",
        "cuda_available": device_info["cuda_available"],
        "device": device_info["device"],
        "gpu_name": device_info["gpu_name"],
        "gpu_memory_gb": device_info["gpu_memory_gb"],
    }


@app.post("/warmup")
def warmup():
    """Pre-load all models and indexes into memory."""
    start = time.time()
    try:
        load_bm25()
        load_faiss()
        load_embedder()
        load_reranker()
        elapsed = round(time.time() - start, 2)
        return {"status": "ok", "message": "Cache warmed up", "time_seconds": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Main RAG endpoint - ask a question and get an answer with sources."""
    start = time.time()
    
    try:
        if req.use_iterative and req.retriever_mode == "Hybrid":
            answer, sources, refined, timings = rag_answer_iterative_timed(
                query=req.question,
                bm25_k=req.bm25_k,
                vec_k=req.vec_k,
                merge_k=req.merge_k,
                rerank_k=req.rerank_k,
                refine_n=req.refine_n,
                use_rerank=req.use_rerank,
                sarcastic_mode=req.sarcastic_mode,
            )
        else:
            answer, sources, timings = rag_answer_timed(
                query=req.question,
                retriever_mode=req.retriever_mode,
                bm25_k=req.bm25_k,
                vec_k=req.vec_k,
                merge_k=req.merge_k,
                rerank_k=req.rerank_k,
                use_rerank=req.use_rerank,
                sarcastic_mode=req.sarcastic_mode,
            )
            refined = None
        
        total_time = round(time.time() - start, 3)
        
        return AskResponse(
            answer=answer,
            sources=sources,
            refined_queries=refined,
            timings=timings,
            total_time=total_time,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve")
def retrieve(question: str, mode: str = "hybrid", k: int = 10):
    """Debug endpoint - just retrieve documents without LLM."""
    if mode == "bm25":
        results = bm25_retrieve(question, k=k)
    elif mode == "vector":
        results = vector_retrieve(question, k=k)
    else:
        results = hybrid_retrieve(question, bm25_k=k, vec_k=k, final_k=k)
    
    return {"mode": mode, "results": results}


# ========================================
# Static File Serving (Frontend)
# ========================================

# Serve static files (CSS, JS)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the main frontend page."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Frontend not found. API is running at /docs"}

    @app.get("/{filename:path}")
    async def serve_static(filename: str):
        """Serve static files."""
        file_path = STATIC_DIR / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Wiki Whatiz...")
    print(f"📁 Serving frontend from: {STATIC_DIR}")
    print(f"🌐 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
