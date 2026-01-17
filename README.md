# Vietnam Labor Law RAG System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LlamaIndex-0.14.x-green.svg" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/FastAPI-0.127+-red.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Chainlit-2.9+-purple.svg" alt="Chainlit">
</p>

Hệ thống Hỏi đáp Pháp luật Lao động Việt Nam sử dụng RAG (Retrieval-Augmented Generation).

---

## 📁 Project Structure

```
RAG/
├── data/
│   ├── Law/                    # Các file luật gốc (.docx)
│   ├── ND/                     # Các file nghị định (.docx)
│   ├── script/
│   │   └── processing.py       # Script parse văn bản pháp luật
│   ├── legal_decrees.json      # Dataset sau xử lý (727 điều)
│   └── ingest_to_qdrant.py     # Script ingest vào Qdrant
│
├── src/
│   ├── config.py               # Cấu hình hệ thống
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Pydantic models
│   └── engine/
│       ├── chat_engine.py      # RAG pipeline + SemanticRouter
│       ├── retriever.py        # HybridRetriever (Vector + BM25)
│       └── components.py       # LLM, Embedding, Reranker factories
│
├── frontend/
│   └── app.py                  # Chainlit UI
│
├── eval/
│   ├── test_case/              # 37 test cases
│   └── eval_*.py               # Scripts đánh giá
│
└── rag_architecture.png        # Sơ đồ kiến trúc
```

---

## 🏗️ Architecture

![RAG Architecture](rag_architecture.png)

### Components

| Component | Model |
|-----------|-------|
| **Embedding** | AITeamVN/Vietnamese_Embedding |
| **Reranker** | BAAI/bge-reranker-v2-m3 |
| **LLM** | Groq/llama-3.3-70b-versatile |
| **Vector DB** | Qdrant Cloud |

---

## 🚀 Quick Start

### 1. Installation

```bash
conda create -n RAG python=3.11
conda activate RAG
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

Required environment variables:
```env
GROQ_API_KEY=your-groq-api-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 3. Process & Ingest Data

```bash
cd data/script
python processing.py

cd ..
python ingest_to_qdrant.py
```

### 4. Start Server

```bash
# Backend
python -m src.main

# Frontend (new terminal)
cd frontend
chainlit run app.py --port 8501
```

- **Backend API**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:8501

---

## ⚙️ Main Parameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| VECTOR_TOP_K | 15 | Dense search results |
| BM25_TOP_K | 15 | Sparse search results |
| RERANKER_TOP_N | 7 | Final context for LLM |
| LLM_TEMPERATURE | 0.05 | Near-deterministic |
| MEMORY_TOKEN_LIMIT | 4096 | Conversation history |
...

---

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Streaming chat (SSE) |
| `/query` | POST | Simple RAG query |
| `/reset-memory` | POST | Clear history |
| `/health` | GET | Health check |
