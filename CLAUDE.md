# Document Intelligence Platform — CLAUDE.md

## Project Overview
Production-grade Hybrid RAG system for legal & compliance document Q&A.
Ingests PDFs/DOCX, performs hybrid retrieval (dense + sparse), reranks results,
and returns citation-grounded answers with hallucination detection.

## Tech Stack
- Backend: FastAPI (Python 3.11), async
- Vector DB: Qdrant (local Docker)
- Embeddings: OpenAI text-embedding-3-small OR Cohere embed-v3
- Reranker: Cohere Rerank v3
- Sparse Search: BM25 (rank_bm25 library)
- LLM Chains: LangChain LCEL
- Document Parsing: PyMuPDF, Unstructured.io
- Evaluation: RAGAS
- Frontend: React + Tailwind CSS
- Deployment: Docker Compose

## Project Structure
- /backend — FastAPI application
- /frontend — React application
- /qdrant_storage — Qdrant vector DB local storage

## Coding Conventions
- Python: use async/await everywhere in FastAPI
- Use Pydantic v2 for all data models
- All services go in backend/app/services/
- Never hardcode API keys — always use .env + python-dotenv
- Every function must have a docstring
- Type hints on every function signature

## Current Phase
Phase 5 — Docker Compose complete

## Docker Commands
```bash
# Production
docker compose up --build           # Build and start all services
docker compose up --build -d        # Detached mode
docker compose down                 # Stop all containers
docker compose down -v              # Stop + delete volumes
docker compose logs -f backend      # Stream backend logs
docker compose ps                   # Check container status

# Development (hot reload)
docker compose -f docker-compose.yml \
  -f docker-compose.dev.yml up --build
```

## Service URLs (Docker)
- Frontend:         http://localhost:3000
- Backend API:      http://localhost:8000
- Backend Docs:     http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

## IMPORTANT — Do Not
- Do not use synchronous code in FastAPI route handlers
- Do not use deprecated LangChain v1 syntax — use LCEL (pipe operator |)
- Do not put business logic inside route handlers — use service layer
- Do not commit .env file