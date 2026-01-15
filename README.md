# Wiki Whatiz

> *سب کو سب نہیں ملتا*

A RAG (Retrieval-Augmented Generation) powered Wikipedia knowledge assistant with a beautifully dark "Cabinet of Wonders" inspired UI.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch)
![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-purple)

---

## About

**Wiki Whatiz** is a Big Data semester project that implements a complete RAG pipeline to answer questions using Wikipedia as a knowledge base. The system combines:

- **Hybrid Retrieval**: BM25 + Vector search with FAISS
- **Query Refinement**: LLM-powered query expansion
- **Re-ranking**: Cross-encoder based relevance scoring
- **Intelligent Responses**: Powered by local LLM with personality

## Features

- **Beautiful Dark UI** - Cabinet of Wonders inspired theme with purple accents
- **Cursor Glow Effect** - Smooth motion-blur following cursor
- **Latency Visualization** - Animated bar charts showing pipeline timing
- **Smart Responses** - LLM with contextual understanding
- **GPU Accelerated** - CUDA support for faster embeddings
- **Hybrid Search** - Combines keyword (BM25) and semantic (vector) search
- **Source Citations** - Shows Wikipedia sources with relevance scores

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| LLM | Ollama (llama3.1:8b) |
| Embeddings | sentence-transformers |
| Vector Store | FAISS |
| Keyword Search | rank-bm25 |
| Re-ranking | Cross-encoder |

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with `llama3.1:8b` model
- NVIDIA GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Viblla/bda-rag-wiki.git
   cd bda-rag-wiki
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull Ollama model**
   ```bash
   ollama pull llama3.1:8b
   ```

5. **Run the server**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

6. **Open in browser**
   ```
   http://localhost:8000
   ```

## Project Structure

```
bda-rag-wiki/
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit app (legacy)
├── requirements.txt       # Python dependencies
├── data/
│   ├── processed/
│   │   ├── faiss.index    # Vector index
│   │   └── wiki_chunks.jsonl
│   └── eval/
│       └── questions.jsonl
├── src/
│   ├── rag_answer.py      # RAG pipeline & LLM
│   ├── retrieval.py       # Hybrid retrieval
│   ├── llm.py             # Ollama integration
│   ├── query_refine.py    # Query expansion
│   ├── confidence.py      # Answer confidence
│   └── utils/
│       └── build_index.py # Index builder
└── web/
    └── public/
        ├── index.html     # Frontend UI
        ├── styles.css     # Dark theme styles
        └── app.js         # Frontend logic
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend UI |
| `/status` | GET | Check backend & GPU status |
| `/ask` | POST | Ask a question (full RAG) |
| `/retrieve` | POST | Retrieve sources only |
| `/warmup` | POST | Warm up model cache |

## RAG Pipeline

```
Question → Query Refinement → Hybrid Retrieval → Re-ranking → LLM Generation → Answer
              |                    |              |              |
         Expand query         BM25 + Vector    Cross-encoder    Contextual
         with LLM             similarity       scoring           response
```

## UI Features

- **Staggered Animations** - Elements fade in sequentially
- **Scroll-triggered Effects** - Source cards animate on scroll
- **Bar Fill Animation** - Latency bars fill smoothly
- **Motion Blur Cursor** - Smooth trailing glow effect
- **Color Transitions** - Key answers highlighted with animated color

## Author

**Ahmed Bilal Nazim**  
Registration No: **2022064**  
GIKI - 7th Semester  
Big Data Analytics - Semester Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>"Because you couldn't just read Wikipedia yourself, could you?"</em>
</p>
