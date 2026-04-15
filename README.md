# 🏛️ Document Intelligence Platform

### Production-Grade Hybrid RAG System for Legal & Compliance

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade document Q&A and comparison engine built for legal and compliance use cases. Ingests PDFs, DOCX, and TXT files, performs hybrid retrieval combining dense vector search (Qdrant + OpenAI embeddings) with sparse BM25 keyword search, fuses results using Reciprocal Rank Fusion, reranks with Cohere Rerank v3, and generates citation-grounded answers with hallucination detection. Evaluated using the RAGAS framework achieving an **overall score of 82.4%** across 15 GDPR test cases.

---

## 🎯 Problem Statement

Law firms, banks, insurance companies, and enterprises manage thousands of legal documents. Finding specific clauses, comparing policies across documents, and getting accurate answers from dense regulatory text is slow and error-prone. This system solves that with a production-grade RAG pipeline that returns precise, cited answers in under 3 seconds.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    React Frontend                    │
│         Document Upload │ Chat UI │ Compare Mode     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│                   FastAPI Backend                    │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Ingestion  │  │   Retrieval  │  │    LLM     │  │
│  │  Pipeline   │  │   Engine     │  │   Layer    │  │
│  │             │  │              │  │            │  │
│  │ PyMuPDF     │  │ Dense Search │  │ LangChain  │  │
│  │ Unstructured│  │ BM25 Sparse  │  │ LCEL Chain │  │
│  │ Semantic    │  │ RRF Fusion   │  │ Citations  │  │
│  │ Chunking    │  │ Cohere       │  │ Halluc.    │  │
│  │             │  │ Reranker     │  │ Detection  │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                │                │          │
└─────────┼────────────────┼────────────────┼──────────┘
          │                │                │
    ┌─────▼─────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │  Qdrant   │    │   BM25    │   │  OpenAI   │
    │ Vector DB │    │   Index   │   │  GPT-4o   │
    └───────────┘    └───────────┘   └───────────┘
```

---

## ✨ Key Features

- 🔍 **Hybrid RAG Retrieval** — dense vector search + sparse BM25 + Reciprocal Rank Fusion
- 🎯 **Cohere Reranking** — precision layer over merged results for top-N selection
- 📄 **Multi-format Ingestion** — PDF, DOCX, TXT with automatic parsing
- 🧠 **Semantic Chunking** — topic-aware splitting with fallback to recursive character splitting
- 📎 **Citation-grounded Answers** — every claim linked to exact document and page number
- 🛡️ **Hallucination Detection** — three-layer system: keyword contradiction, self-consistency checking, and post-generation faithfulness filter
- 📊 **RAGAS Evaluation** — automated quality metrics with Streamlit dashboard
- 🐳 **Docker Compose** — one-command deployment of all services
- ⚡ **Async FastAPI** — production-ready, fully asynchronous API design
- 🖥️ **React Frontend** — clean chat UI with document sidebar and compare mode

---

## 📊 RAGAS Evaluation Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| 🎯 Faithfulness | 0.77 | > 0.70 | ✅ |
| 📊 Answer Relevancy | 0.78 | > 0.70 | ✅ |
| 🔍 Context Recall | 0.90 | > 0.70 | ✅ |
| ⚡ Context Precision | 0.85 | > 0.70 | ✅ |
| 📈 **Overall** | **0.82** | > 0.70 | ✅ |

*Evaluated on 15 GDPR compliance document test cases using the RAGAS framework*

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic v2 | Async API framework with data validation |
| **AI/ML** | LangChain LCEL, OpenAI GPT-4o-mini, text-embedding-3-small, Cohere Rerank v3 | LLM orchestration, embeddings, reranking |
| **Vector Database** | Qdrant (local Docker) | Dense similarity search with cosine distance |
| **Document Processing** | PyMuPDF, Unstructured.io, python-docx | PDF/DOCX/TXT parsing and text extraction |
| **Retrieval** | rank-bm25, Reciprocal Rank Fusion (custom) | Sparse keyword search + rank merging |
| **Evaluation** | RAGAS, Streamlit, Plotly | Automated RAG quality metrics and dashboard |
| **Frontend** | React, Vite, Tailwind CSS, Axios | Modern SPA with responsive chat interface |
| **Infrastructure** | Docker, Docker Compose, Nginx | Containerized deployment with reverse proxy |

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Cohere API key

### 1. Clone the repository

```bash
git clone https://github.com/MuneebMM/document-intelligence-platform.git
cd document-intelligence-platform
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your API keys:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here
```

### 3. Start all services

```bash
docker compose up --build
```

### 4. Access the application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Evaluation Dashboard | `streamlit run backend/evaluation_dashboard.py` → http://localhost:8501 |

---

## 📁 Project Structure

```
document-intelligence-platform/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── documents.py       # Upload & ingest endpoints
│   │   │       ├── query.py           # Ask, search, compare endpoints
│   │   │       └── evaluation.py      # RAGAS evaluation endpoints
│   │   ├── core/
│   │   │   └── config.py             # Pydantic settings + env loading
│   │   ├── evaluation/
│   │   │   ├── ragas_evaluator.py    # RAGAS evaluation pipeline
│   │   │   └── test_dataset.py       # 15 GDPR ground-truth test cases
│   │   ├── models/
│   │   │   └── schemas.py            # Pydantic v2 request/response models
│   │   ├── services/
│   │   │   ├── document_parser.py    # PDF/DOCX/TXT parsing
│   │   │   ├── chunker.py           # Semantic + recursive chunking
│   │   │   ├── embedding_service.py  # OpenAI embedding batching
│   │   │   ├── vector_store.py       # Qdrant upsert & dense search
│   │   │   ├── bm25_service.py       # BM25 sparse index + search
│   │   │   ├── ingestion.py          # Full ingest pipeline orchestrator
│   │   │   ├── retrieval_service.py  # Hybrid search + RRF + reranking
│   │   │   ├── prompt_templates.py   # LangChain prompt templates
│   │   │   ├── llm_service.py        # Answer gen + citations + halluc.
│   │   │   └── qa_service.py         # Top-level QA orchestrator
│   │   └── main.py                   # FastAPI app entry point
│   ├── evaluation_dashboard.py       # Streamlit RAGAS dashboard
│   ├── Dockerfile                    # Multi-stage production build
│   ├── entrypoint.sh                 # Container startup script
│   └── requirements.txt              # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DocumentSidebar.jsx   # Upload + document list
│   │   │   ├── ChatMessage.jsx       # Message bubbles + citations
│   │   │   └── ChatInput.jsx         # Auto-resize input + compare toggle
│   │   ├── services/
│   │   │   └── api.js                # Axios API client
│   │   ├── App.jsx                   # Main application component
│   │   └── App.css                   # Application styles
│   ├── nginx.conf                    # Reverse proxy + SPA config
│   ├── Dockerfile                    # Multi-stage Node + Nginx build
│   └── package.json                  # Frontend dependencies
├── docker-compose.yml                # Production orchestration
├── docker-compose.dev.yml            # Dev override (hot reload)
├── CLAUDE.md                         # Coding conventions
└── README.md                         # This file
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and ingest a document (PDF/DOCX/TXT) |
| `POST` | `/api/v1/query/ask` | Ask a question with citation-grounded answer |
| `POST` | `/api/v1/query/search` | Raw hybrid search returning ranked chunks |
| `POST` | `/api/v1/query/compare` | Compare answers across multiple documents |
| `POST` | `/api/v1/evaluation/run` | Run full RAGAS evaluation (15 test cases) |
| `POST` | `/api/v1/evaluation/run-single` | Evaluate a single custom question |
| `GET` | `/api/v1/evaluation/results` | List all saved evaluation results |
| `GET` | `/api/v1/evaluation/results/{filename}` | Retrieve a specific evaluation result |
| `GET` | `/health` | Service health check |

---

## 🧪 Running Evaluation

```bash
# Start evaluation via API (takes 5-10 minutes)
curl --max-time 900 -X POST http://localhost:8000/api/v1/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "documents", "save_results": true}'

# View the Streamlit dashboard
streamlit run backend/evaluation_dashboard.py
# Open http://localhost:8501
```

---

## 🔮 Future Improvements

- [ ] Multi-tenancy support (per-user document collections)
- [ ] Streaming responses for long answers
- [ ] Support for Excel and PowerPoint ingestion
- [ ] Fine-tuned embedding model for legal domain
- [ ] Redis caching for repeated queries
- [ ] CI/CD pipeline with automated RAGAS regression tests

---

## 👤 Author

**Muneeb** — Aspiring AI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-MuneebMM-181717?logo=github)](https://github.com/MuneebMM)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
