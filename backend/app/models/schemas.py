"""Pydantic v2 request and response models for all API endpoints."""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------


class ChunkResult(BaseModel):
    """A single retrieved and reranked document chunk."""

    chunk_id: str = Field(..., description="UUID of the chunk.")
    text: str = Field(..., description="Chunk text content.")
    filename: str = Field(..., description="Source document filename.")
    page_number: int = Field(..., description="1-based source page number.")
    relevance_score: float = Field(..., description="Cohere reranker relevance score.")
    chunk_index: int = Field(..., description="0-based position within the document.")


class CitationResult(BaseModel):
    """A structured citation linking an answer claim to a source chunk."""

    chunk_id: str = Field(..., description="UUID of the cited chunk.")
    filename: str = Field(..., description="Source document filename.")
    page_number: int = Field(..., description="1-based source page number.")
    quote: str = Field(..., description="Relevant excerpt from the source chunk.")
    relevance_score: float = Field(
        default=0.0, description="Reranker score of the cited chunk."
    )


class ConfidenceResult(BaseModel):
    """Self-consistency hallucination detection result."""

    is_consistent: bool = Field(
        ..., description="True when entity overlap between two independent answers ≥ 0.8."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Entity overlap ratio in [0.0, 1.0]."
    )
    warning: str | None = Field(
        default=None,
        description="Human-readable warning when inconsistency is detected.",
    )


class DocumentComparison(BaseModel):
    """Per-document answer produced by the multi-document compare endpoint."""

    filename: str = Field(..., description="Source document filename.")
    answer: str = Field(..., description="LLM-generated answer for this document.")
    citations: list[CitationResult] = Field(
        default_factory=list,
        description="Citations extracted from this document's answer.",
    )


# ---------------------------------------------------------------------------
# Document upload
# ---------------------------------------------------------------------------


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful document ingestion."""

    status: str = Field(..., description="Always 'success' on a 200 response.")
    filename: str = Field(..., description="Original uploaded filename.")
    total_chunks: int = Field(..., description="Number of chunks stored in Qdrant.")
    message: str = Field(..., description="Human-readable confirmation message.")


# ---------------------------------------------------------------------------
# Raw hybrid search  (POST /query/search)
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for the raw hybrid search endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language question.")
    collection_name: str = Field(
        default="documents", description="Qdrant collection to search."
    )
    top_n: int = Field(
        default=5, ge=1, le=20, description="Number of results to return."
    )


class SearchResponse(BaseModel):
    """Response body for the raw hybrid search endpoint."""

    query: str = Field(..., description="The original query string.")
    results: list[ChunkResult] = Field(..., description="Ranked chunk results.")
    total_results: int = Field(..., description="Number of results returned.")


# ---------------------------------------------------------------------------
# LLM Q&A  (POST /query/ask)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the LLM-grounded Q&A endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language question.")
    collection_name: str = Field(
        default="documents", description="Qdrant collection to search."
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve and use as LLM context.",
    )


class QueryResponse(BaseModel):
    """Response body for the LLM-grounded Q&A endpoint."""

    query: str = Field(..., description="The original question.")
    answer: str = Field(..., description="LLM-generated, citation-grounded answer.")
    citations: list[CitationResult] = Field(
        ..., description="Structured source references extracted from the answer."
    )
    confidence: ConfidenceResult = Field(
        ..., description="Self-consistency hallucination detection result."
    )
    retrieved_chunks_count: int = Field(
        ..., description="Number of chunks used as LLM context."
    )
    model_used: str = Field(..., description="Name of the LLM used to generate the answer.")
    response_time_ms: int = Field(
        ..., description="End-to-end pipeline latency in milliseconds."
    )
    hallucination_guard_triggered: bool = Field(
        default=False,
        description="True when the post-generation answer guard fired.",
    )
    guard_reason: str | None = Field(
        default=None,
        description="Reason the guard fired, or None when it did not.",
    )


# ---------------------------------------------------------------------------
# Multi-document comparison  (POST /query/compare)
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    """Request body for the multi-document comparison endpoint."""

    query: str = Field(..., min_length=1, description="Question to answer per document.")
    filenames: list[str] = Field(
        ..., min_length=1, description="Filenames to compare answers across."
    )
    collection_name: str = Field(
        default="documents", description="Qdrant collection to search."
    )


class CompareResponse(BaseModel):
    """Response body for the multi-document comparison endpoint."""

    query: str = Field(..., description="The original question.")
    comparisons: list[DocumentComparison] = Field(
        ..., description="Per-document answers and citations."
    )
