# RAG v3 - Production-Grade Vietnam Labor Law QA System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LlamaIndex-0.10.x-green.svg" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-red.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Chainlit-1.0+-purple.svg" alt="Chainlit">
</p>

## 🏛️ Overview

RAG v3 is a **production-grade** Retrieval-Augmented Generation system for querying Vietnam Labor Law 2019. It features an **agentic architecture** with semantic routing, conversational memory, and hybrid search capabilities.

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Semantic Router** | LLM-based intent classification (CHAT vs LAW) |
| 💬 **Conversational Memory** | CondensePlusContextChatEngine with query rewriting |
| 🔍 **Hybrid Search** | Vector (Qdrant) + BM25 + Reciprocal Rank Fusion |
| 🎯 **BGE Reranker** | BAAI/bge-reranker-v2-m3 for result refinement |
| 📊 **Observability** | Arize Phoenix tracing integration |
| ⚡ **Streaming** | Full async streaming from backend to frontend |

## 📁 Project Structure

```
RAG_v3/
├── .env.example                # Environment config template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── VIETNAM_LABOR_LAW.pdf   # Source document
│
├── scripts/
│   └── ingest.py               # Offline PDF → Qdrant ingestion
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings
│   ├── main.py                 # FastAPI entrypoint + Phoenix setup
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API endpoints (/chat, /query, /health)
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   └── engine/
│       ├── __init__.py
│       ├── components.py       # LLM, Embedding, Reranker factories
│       ├── retriever.py        # HybridRetriever (Vector + BM25 + RRF)
│       └── chat_engine.py      # SemanticRouter + CondensePlusContextChatEngine
│
└── frontend/
    ├── app.py                  # Chainlit UI application
    ├── .env.example            # Frontend config
    └── .chainlit/
        └── config.toml         # Chainlit UI configuration
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Qdrant Cloud account (or local Qdrant)
- Google Gemini API key (or OpenAI)

### 2. Installation

```bash
# Clone or navigate to RAG_v3
cd RAG_v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:
```env
GEMINI_API_KEY=your-gemini-api-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 4. Ingest Data

Place your PDF in `data/VIETNAM_LABOR_LAW.pdf`, then run:

```bash
python scripts/ingest.py --pdf data/VIETNAM_LABOR_LAW.pdf
```

### 5. Start Backend Server

```bash
# From project root
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 6. Start Frontend

```bash
# In a new terminal
cd frontend
cp .env.example .env
chainlit run app.py --port 8501
```

Visit http://localhost:8501 to start chatting!

## 🔌 API Endpoints

### POST /chat
Main chat endpoint with semantic routing.

```json
{
  "message": "Thời gian làm việc tối đa trong một tuần?",
  "session_id": "optional-session-id",
  "stream": false,
  "skip_routing": false
}
```

Response:
```json
{
  "answer": "Theo Điều 105, Khoản 1...",
  "intent": "LAW",
  "source_nodes": [...],
  "session_id": "..."
}
```

### POST /query
Simple query (backward compatible with v2).

```json
{
  "question": "Quyền của người lao động?",
  "top_k": 5
}
```

### POST /reset-memory
Reset conversation history.

### GET /health
Health check endpoint.

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌────────────────────────────────┐
│             │     │             │     │           RAG Engine            │
│   Chainlit  │────▶│   FastAPI   │────▶│                                │
│   Frontend  │     │   Backend   │     │  ┌──────────┐  ┌────────────┐  │
│             │◀────│             │◀────│  │ Semantic │  │CondensePlus│  │
└─────────────┘     └─────────────┘     │  │  Router  │─▶│ ChatEngine │  │
                                        │  └──────────┘  └────────────┘  │
                                        │        │              │        │
                                        │        ▼              ▼        │
                                        │  ┌──────────────────────────┐  │
                                        │  │    Hybrid Retriever      │  │
                                        │  │  ┌────────┐ ┌────────┐   │  │
                                        │  │  │ Vector │ │  BM25  │   │  │
                                        │  │  │ Search │ │ Search │   │  │
                                        │  │  └────────┘ └────────┘   │  │
                                        │  │         │ RRF │          │  │
                                        │  │         ▼     ▼          │  │
                                        │  │    ┌────────────┐        │  │
                                        │  │    │  Reranker  │        │  │
                                        │  │    └────────────┘        │  │
                                        │  └──────────────────────────┘  │
                                        └────────────────────────────────┘
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │                             │
                                        ▼                             ▼
                                  ┌──────────┐                 ┌──────────┐
                                  │  Qdrant  │                 │  Gemini  │
                                  │  Cloud   │                 │   LLM    │
                                  └──────────┘                 └──────────┘
```

## 📊 Observability (Phoenix)

Start Arize Phoenix for tracing:

```bash
# Install Phoenix
pip install arize-phoenix

# Start Phoenix server
phoenix serve

# Phoenix UI: http://localhost:6006
```

Configure in `.env`:
```env
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
ENABLE_TRACING=true
```

## 🔧 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider (gemini/openai) |
| `LLM_MODEL_GEMINI` | `models/gemini-1.5-flash` | Gemini model |
| `LLM_MODEL_OPENAI` | `gpt-4o-mini` | OpenAI model |
| `EMBEDDING_MODEL` | `bkai-foundation-models/vietnamese-bi-encoder` | Vietnamese embedding |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `VECTOR_TOP_K` | `20` | Vector search results |
| `BM25_TOP_K` | `20` | BM25 search results |
| `RERANKER_TOP_N` | `5` | Final reranked results |
| `RRF_K` | `60` | RRF fusion constant |

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
