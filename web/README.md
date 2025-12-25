# Wiki Whatiz - RAG-Powered Wikipedia Q&A

A beautiful, modern web application for question-answering using Retrieval-Augmented Generation (RAG) with Wikipedia knowledge.

![Wiki Whatiz](https://img.shields.io/badge/RAG-Powered-blue) ![Node.js](https://img.shields.io/badge/Node.js-18+-green) ![Python](https://img.shields.io/badge/Python-3.11+-yellow)

## Features

- 🔍 **Hybrid Retrieval**: Combines BM25 and vector search for optimal results
- 🎯 **Re-ranking**: Uses cross-encoder models to refine search results
- 🔄 **Iterative RAG**: Query refinement for better answer quality
- 🚀 **GPU Accelerated**: Full CUDA support for fast inference
- 🎨 **Beautiful UI**: Modern, dark theme inspired by Cabinet of Wonders
- ⚡ **Real-time**: Fast responses with detailed timing breakdown

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│  Node.js API    │────▶│  Python RAG     │
│   (HTML/CSS/JS) │     │  (Express)      │     │  (FastAPI)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                ┌───────────────────────┼───────────────────────┐
                                │                       │                       │
                                ▼                       ▼                       ▼
                        ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
                        │   FAISS       │       │   BM25        │       │   Ollama      │
                        │   (Vectors)   │       │   (Keywords)  │       │   (LLM)       │
                        └───────────────┘       └───────────────┘       └───────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama with `llama3.1:8b` model
- NVIDIA GPU with CUDA (optional, but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wiki-whatiz.git
   cd wiki-whatiz
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   
   pip install -r requirements.txt
   pip install fastapi uvicorn
   ```

3. **Install Node.js dependencies**
   ```bash
   cd web
   npm install
   cd ..
   ```

4. **Start Ollama**
   ```bash
   ollama pull llama3.1:8b
   ollama serve
   ```

### Running the Application

1. **Start the Python API backend** (Terminal 1)
   ```bash
   .venv\Scripts\activate
   uvicorn api:app --reload --port 8000
   ```

2. **Start the Node.js frontend** (Terminal 2)
   ```bash
   cd web
   npm start
   ```

3. **Open in browser**
   ```
   http://localhost:3000
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Check backend status and GPU info |
| `/api/warmup` | POST | Pre-load models into memory |
| `/api/ask` | POST | Ask a question and get RAG answer |
| `/api/retrieve` | POST | Debug: retrieve documents only |

### Example Request

```bash
curl -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why is the sky blue?",
    "use_iterative": true,
    "use_rerank": true,
    "rerank_k": 5
  }'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | Node.js server port |
| `PYTHON_API_URL` | http://localhost:8000 | Python backend URL |
| `OLLAMA_MODEL` | llama3.1:8b | LLM model to use |
| `EMBED_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `RERANK_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-ranking model |

## Project Structure

```
bda-rag-wiki/
├── api.py                 # FastAPI backend
├── app.py                 # Original Streamlit app
├── requirements.txt       # Python dependencies
├── src/
│   ├── retrieval.py      # BM25, FAISS, reranking
│   ├── rag_answer.py     # RAG pipeline
│   ├── llm.py            # Ollama integration
│   └── ...
├── data/
│   └── processed/        # FAISS index and BM25 pickles
└── web/
    ├── package.json
    ├── server.js         # Express server
    └── public/
        ├── index.html    # Frontend HTML
        ├── styles.css    # CSS styles
        └── app.js        # Frontend JavaScript
```

## Deployment

### GitHub Pages (Frontend Only)

The static frontend can be deployed to GitHub Pages:

```bash
# Build and deploy
cd web/public
git init
git add .
git commit -m "Deploy"
git push -f git@github.com:yourusername/wiki-whatiz.git main:gh-pages
```

### Full Stack Deployment

For full deployment, you'll need:
1. A server with GPU (for optimal performance)
2. Docker or direct deployment
3. Reverse proxy (nginx) to combine frontend and API

## License

MIT License

## Acknowledgments

- [Cabinet of Wonders](https://cabinetofwonders.app/) - Design inspiration
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Ollama](https://ollama.ai/) - Local LLM inference
