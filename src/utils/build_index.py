"""
Unified build script for corpus preprocessing and index creation.
Consolidates:
  - build_corpus.py: Wikipedia dataset cleaning and chunking
  - bm25_retriever.py: BM25 index creation
  - vector_index.py: FAISS vector index creation
"""

import json
import re
import pickle
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer


# ===========================
# Paths
# ===========================
CHUNKS_PATH = Path("data/processed/wiki_chunks.jsonl")
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")
BM25_DOCS_PATH = Path("data/processed/bm25_docs.pkl")
FAISS_PATH = Path("data/processed/faiss.index")
FAISS_META_PATH = Path("data/processed/faiss_meta.pkl")

# ===========================
# Text Processing
# ===========================

def clean_text(t: str) -> str:
    """Clean and normalize text."""
    t = t.replace("\t", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def is_good_article(text: str) -> bool:
    """Filter out short/empty articles."""
    return text and len(text) >= 400


# ===========================
# Step 1: Build Corpus from Wikipedia
# ===========================

def build_corpus():
    """Load Wikipedia dataset, clean, chunk, and save to JSONL."""
    print("=" * 60)
    print("STEP 1: Building corpus from Wikipedia")
    print("=" * 60)
    
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Wikitext-103 dataset...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    texts = ds["train"]["text"]

    articles = []
    buff = []
    for line in texts:
        line = line.strip()
        if not line:
            if buff:
                art = clean_text(" ".join(buff))
                if is_good_article(art):
                    articles.append(art)
                buff = []
            continue

        # skip heading lines like: "== History =="
        if re.fullmatch(r"=+.*=+", line):
            continue

        buff.append(line)

    if buff:
        art = clean_text(" ".join(buff))
        if is_good_article(art):
            articles.append(art)

    print(f"Collected articles: {len(articles)}")

    chunk_id = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for doc_id, art in enumerate(tqdm(articles, desc="Chunking")):
            for c in chunk_text(art):
                rec = {"chunk_id": chunk_id, "doc_id": doc_id, "source": "wikitext-103", "text": c}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"Saved chunks: {chunk_id}")
    print(f"Output: {CHUNKS_PATH.resolve()}\n")


# ===========================
# Step 2: Build BM25 Index
# ===========================

def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def build_bm25_index():
    """Build BM25 index from chunks."""
    print("=" * 60)
    print("STEP 2: Building BM25 index")
    print("=" * 60)
    
    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found. Run build_corpus() first.")
        return

    texts = []
    meta = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            rec = json.loads(line)
            texts.append(tokenize(rec["text"]))
            meta.append(rec)

    print(f"Building BM25 index for {len(texts)} chunks...")
    bm25 = BM25Okapi(texts)

    with BM25_INDEX_PATH.open("wb") as f:
        pickle.dump(bm25, f)

    with BM25_DOCS_PATH.open("wb") as f:
        pickle.dump(meta, f)

    print(f"BM25 index built")
    print(f"Chunks indexed: {len(meta)}")
    print(f"Index saved to: {BM25_INDEX_PATH}")
    print(f"Metadata saved to: {BM25_DOCS_PATH}\n")


# ===========================
# Step 3: Build FAISS Vector Index
# ===========================

def build_faiss_index(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Build FAISS vector index from chunks."""
    print("=" * 60)
    print("STEP 3: Building FAISS vector index")
    print("=" * 60)
    
    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found. Run build_corpus() first.")
        return

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = []
    meta = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            rec = json.loads(line)
            meta.append({"chunk_id": rec["chunk_id"], "doc_id": rec["doc_id"], "source": rec["source"]})
            texts.append(rec["text"])

    print(f"Total chunks: {len(texts)}")

    # Encode in batches (avoid RAM spikes)
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}")

    # Cosine similarity via inner product on normalized vectors
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    with FAISS_META_PATH.open("wb") as f:
        pickle.dump({"meta": meta, "texts": texts}, f)

    print(f"FAISS index saved: {FAISS_PATH}")
    print(f"Metadata saved: {FAISS_META_PATH}\n")


# ===========================
# Main: Run All Steps
# ===========================

def main():
    """Run complete indexing pipeline."""
    print("\n" + "=" * 60)
    print("RAG Index Building Pipeline")
    print("=" * 60 + "\n")

    try:
        build_corpus()
        build_bm25_index()
        build_faiss_index()
        print("=" * 60)
        print("✓ All indexes built successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error during indexing: {e}")
        raise


if __name__ == "__main__":
    main()
