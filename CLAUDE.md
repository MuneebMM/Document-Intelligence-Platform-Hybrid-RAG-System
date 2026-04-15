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
Phase 0 — Initial setup

## IMPORTANT — Do Not
- Do not use synchronous code in FastAPI route handlers
- Do not use deprecated LangChain v1 syntax — use LCEL (pipe operator |)
- Do not put business logic inside route handlers — use service layer
- Do not commit .env file