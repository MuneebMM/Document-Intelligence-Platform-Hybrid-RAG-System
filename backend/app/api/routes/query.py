"""Query routes: raw retrieval, LLM-grounded Q&A, and multi-document comparison."""

import time

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.schemas import (
    ChunkResult,
    CitationResult,
    CompareRequest,
    CompareResponse,
    DocumentComparison,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.qa_service import QAService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStoreService

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


async def _build_retrieval_service() -> RetrievalService:
    """Instantiate RetrievalService with BM25 indexes loaded from disk.

    Constructed per-request.  In a production deployment this would be a
    long-lived singleton managed via FastAPI's lifespan context.
    """
    settings = get_settings()
    bm25 = BM25Service()
    await bm25.load_all_indexes("bm25_indexes")
    return RetrievalService(
        settings=settings,
        vector_store=VectorStoreService(settings=settings),
        bm25=bm25,
        embedding_service=EmbeddingService(settings=settings),
    )


def _build_qa_service() -> QAService:
    """Instantiate QAService from cached application settings.

    BM25 index loading is deferred to the first query call inside
    ``QAService._ensure_ready``.
    """
    return QAService(settings=get_settings())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Raw hybrid search (debug — no LLM)",
)
async def search(body: SearchRequest) -> SearchResponse:
    """Return raw reranked chunks without an LLM-generated answer.

    Useful for inspecting what the retrieval pipeline surfaces before any
    LLM processing.  Combines dense Qdrant search, BM25 sparse search, RRF
    fusion, and Cohere reranking.

    Args:
        body: Query, collection name, and desired result count.

    Returns:
        ``SearchResponse`` with ranked chunk list and total count.

    Raises:
        HTTPException 500: Propagated from the retrieval pipeline.
    """
    try:
        service = await _build_retrieval_service()
        raw_results = await service.hybrid_search(
            query=body.query,
            collection_name=body.collection_name,
            final_top_n=body.top_n,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc

    return SearchResponse(
        query=body.query,
        results=[
            ChunkResult(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                filename=r.get("filename", ""),
                page_number=int(r.get("page_number", 0)),
                relevance_score=float(r.get("relevance_score", 0.0)),
                chunk_index=int(r.get("chunk_index", 0)),
            )
            for r in raw_results
        ],
        total_results=len(raw_results),
    )


@router.post(
    "/ask",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="LLM-grounded Q&A with citations and hallucination check",
)
async def ask(body: QueryRequest) -> QueryResponse:
    """Answer a question using the full RAG pipeline.

    Retrieves relevant chunks, generates a grounded answer with GPT-4o-mini,
    extracts structured citations, and runs a self-consistency hallucination
    check.  ``response_time_ms`` is measured around the entire pipeline.

    Args:
        body: Query, collection name, and number of context chunks.

    Returns:
        ``QueryResponse`` with answer, citations, confidence, and latency.

    Raises:
        HTTPException 500: Propagated from the QA pipeline.
    """
    try:
        service = _build_qa_service()
        t_start = time.time()
        result = await service.answer_query(
            query=body.query,
            collection_name=body.collection_name,
            top_n=body.top_n,
        )
        response_time_ms = int((time.time() - t_start) * 1000)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA pipeline failed: {exc}",
        ) from exc

    confidence_raw = result.get("confidence", {})
    citations_raw = result.get("citations", [])

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=[
            CitationResult(
                chunk_id=c.get("chunk_id", ""),
                filename=c.get("filename", ""),
                page_number=int(c.get("page_number", 0)),
                quote=c.get("quote", ""),
                relevance_score=float(c.get("relevance_score", 0.0)),
            )
            for c in citations_raw
        ],
        confidence=confidence_raw,
        retrieved_chunks_count=int(result.get("retrieved_chunks_count", 0)),
        model_used=result.get("model_used", "gpt-4o-mini"),
        response_time_ms=response_time_ms,
        hallucination_guard_triggered=bool(result.get("hallucination_guard_triggered", False)),
        guard_reason=result.get("guard_reason"),
    )


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer the same query across multiple documents and compare",
)
async def compare(body: CompareRequest) -> CompareResponse:
    """Run the same query against each specified document independently.

    Each document is answered using only its own retrieved chunks so the
    responses can be compared side by side.  Useful for spotting
    inconsistencies or differences between contract versions, regulatory
    texts, and similar corpora.

    Args:
        body: Query, list of filenames to compare, and collection name.

    Returns:
        ``CompareResponse`` with per-document answers and citations.

    Raises:
        HTTPException 500: Propagated from the QA pipeline.
    """
    try:
        service = _build_qa_service()
        result = await service.multi_document_compare(
            query=body.query,
            filenames=body.filenames,
            collection_name=body.collection_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {exc}",
        ) from exc

    return CompareResponse(
        query=result["query"],
        comparisons=[
            DocumentComparison(
                filename=comp["filename"],
                answer=comp["answer"],
                citations=[
                    CitationResult(
                        chunk_id=c.get("chunk_id", ""),
                        filename=c.get("filename", ""),
                        page_number=int(c.get("page_number", 0)),
                        quote=c.get("quote", ""),
                        relevance_score=float(c.get("relevance_score", 0.0)),
                    )
                    for c in comp.get("citations", [])
                ],
            )
            for comp in result["comparisons"]
        ],
    )
