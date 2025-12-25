"""
Quick guide for the simplified project structure.

=== Building/Rebuilding Indexes ===

To build the complete index pipeline (corpus → BM25 → FAISS):

    python -m src.utils.build_index

This will:
1. Download and process Wikitext-103 dataset
2. Create BM25 index
3. Create FAISS vector index

Output files:
  - data/processed/wiki_chunks.jsonl     (source chunks)
  - data/processed/bm25_index.pkl        (BM25 index)
  - data/processed/bm25_docs.pkl         (BM25 metadata)
  - data/processed/faiss.index           (FAISS index)
  - data/processed/faiss_meta.pkl        (FAISS metadata)

=== Running the Application ===

    streamlit run app.py

=== Project Structure ===

src/
  ├── app dependencies (llm.py, retrieval.py, rag_answer.py, etc.)
  └── utils/
      └── build_index.py              (consolidated build pipeline)
        (formerly: build_corpus.py, bm25_retriever.py, vector_index.py)

Removed files:
  - test_bm25.py, test_env.py, test_vector.py, test_hybrid_rerank.py (test files)
  - eval.py (outdated, replaced by eval_compare.py)
  - build_corpus.py, bm25_retriever.py, vector_index.py (merged into utils/build_index.py)
"""
