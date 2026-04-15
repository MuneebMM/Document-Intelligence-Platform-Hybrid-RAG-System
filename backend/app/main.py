"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, query
from app.api.routes import evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required runtime directories on startup."""
    Path("evaluation_results").mkdir(exist_ok=True)
    Path("bm25_indexes").mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="Document Intelligence Platform",
    description="Hybrid RAG system for legal & compliance document Q&A.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Return a top-level liveness response for the platform."""
    return {"status": "ok", "service": "document-intelligence-platform"}


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}
