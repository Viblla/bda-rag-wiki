# src/retrieval.py
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


# ----------------------------
# Absolute base dir (project root)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # .../bda-rag-wiki

# ✅ Your indexes are in processed folder
STORE_DIR = BASE_DIR / "data" / "processed"

BM25_INDEX_PATH = STORE_DIR / "bm25_index.pkl"
BM25_DOCS_PATH  = STORE_DIR / "bm25_docs.pkl"

FAISS_PATH      = STORE_DIR / "faiss.index"
FAISS_META_PATH = STORE_DIR / "faiss_meta.pkl"


def _ensure_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {label} file: {p}")


# ----------------------------
# Models
# ----------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ----------------------------
# Tokenizer (BM25)
# ----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ----------------------------
# Cached loaders
# Uses functools.lru_cache for FastAPI compatibility
# Falls back to st.cache_resource for Streamlit
# ----------------------------
from functools import lru_cache

@lru_cache(maxsize=1)
def load_bm25():
    _ensure_exists(BM25_INDEX_PATH, "BM25 index")
    _ensure_exists(BM25_DOCS_PATH, "BM25 docs")

    with BM25_INDEX_PATH.open("rb") as f:
        bm25 = pickle.load(f)
    with BM25_DOCS_PATH.open("rb") as f:
        docs = pickle.load(f)

    return bm25, docs


@lru_cache(maxsize=1)
def load_faiss():
    _ensure_exists(FAISS_PATH, "FAISS index")
    _ensure_exists(FAISS_META_PATH, "FAISS meta/store")

    index = faiss.read_index(str(FAISS_PATH))
    with FAISS_META_PATH.open("rb") as f:
        store = pickle.load(f)

    return index, store


@lru_cache(maxsize=1)
def load_embedder():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    # Warm up the model with a dummy encode to compile CUDA kernels
    if device == "cuda":
        _ = model.encode(["warmup"], convert_to_numpy=True)
    return model


@lru_cache(maxsize=1)
def load_reranker():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(RERANK_MODEL, device=device)
    # Warm up reranker
    if device == "cuda":
        _ = model.predict([("warmup query", "warmup passage")])
    return model


def get_device_info() -> dict:
    """Return GPU/device info for diagnostics."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None,
    }


# ----------------------------
# Retrieval functions
# ----------------------------
def bm25_retrieve(query: str, k: int = 20) -> List[Dict]:
    bm25, docs = load_bm25()
    q_tokens = _tokenize(query)

    scores = np.asarray(bm25.get_scores(q_tokens))
    if scores.size == 0:
        return []

    top_idx = np.argsort(-scores)[:k]

    out = []
    for idx in top_idx:
        d = docs[int(idx)]
        out.append({
            "text": d.get("text", ""),
            "chunk_id": d.get("chunk_id", int(idx)),
            "doc_id": d.get("doc_id", d.get("source", "unknown")),
            "source": d.get("source", "bm25"),
            "score": float(scores[int(idx)]),
        })
    return out


def vector_retrieve(query: str, k: int = 20) -> List[Dict]:
    index, store = load_faiss()
    embedder = load_embedder()

    # Use convert_to_numpy=True for faster GPU->CPU transfer
    q_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    D, I = index.search(q_emb, k)
    I = I[0]
    D = D[0]

    out = []
    for score, idx in zip(D, I):
        if idx < 0:
            continue

        idx_int = int(idx)

        # ✅ store might be list OR dict
        meta = None
        if isinstance(store, dict):
            meta = store.get(idx_int)
        else:
            # list/sequence
            if 0 <= idx_int < len(store):
                meta = store[idx_int]

        # If meta missing -> skip safely (prevents KeyError)
        if meta is None:
            continue

        out.append({
            "text": meta.get("text", ""),
            "chunk_id": meta.get("chunk_id", idx_int),
            "doc_id": meta.get("doc_id", meta.get("source", "unknown")),
            "source": meta.get("source", "vector"),
            "score": float(score),
        })

    return out



def hybrid_retrieve(query: str, bm25_k: int = 60, vec_k: int = 60, final_k: int = 30) -> List[Dict]:
    bm25_hits = bm25_retrieve(query, k=bm25_k)
    vec_hits = vector_retrieve(query, k=vec_k)

    merged = {}
    for h in bm25_hits:
        merged[h["chunk_id"]] = dict(h)

    for h in vec_hits:
        cid = h["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(h)
        else:
            if h["score"] > merged[cid]["score"]:
                merged[cid] = dict(h)

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:final_k]


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    if not candidates:
        return []

    reranker = load_reranker()
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k]
