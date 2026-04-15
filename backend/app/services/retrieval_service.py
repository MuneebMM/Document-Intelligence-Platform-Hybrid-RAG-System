"""Hybrid retrieval service: dense search + BM25 + RRF fusion + Cohere reranking."""

import logging
from typing import Any

import cohere

from app.core.config import Settings
from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrate hybrid retrieval over ingested document chunks.

    Pipeline:
    1. Embed the query with OpenAI.
    2. Dense search against Qdrant (cosine similarity).
    3. Sparse search against the in-memory BM25 index.
    4. Merge both ranked lists with Reciprocal Rank Fusion (RRF).
    5. Rerank the fused candidates with Cohere Rerank v3.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        bm25: BM25Service,
        embedding_service: EmbeddingService,
    ) -> None:
        """Wire up all retrieval dependencies.

        Args:
            settings:          Application settings; Cohere API key is read
                               from here.
            vector_store:      Initialised ``VectorStoreService`` for dense
                               search.
            bm25:              Initialised ``BM25Service`` with a built index
                               for sparse search.
            embedding_service: Initialised ``EmbeddingService`` for query
                               embedding.
        """
        self._vector_store = vector_store
        self._bm25 = bm25
        self._embedding_service = embedding_service
        self._cohere = cohere.AsyncClient(api_key=settings.cohere_api_key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

        RRF assigns each result a score of ``1 / (k + rank)`` (1-based) and
        sums scores for results that appear in both lists.  The constant ``k``
        (default 60, per the original RRF paper) dampens the impact of very
        high-ranked results and reduces sensitivity to rank ties.

        Args:
            dense_results:  Ranked results from Qdrant dense search.
            sparse_results: Ranked results from BM25 sparse search.
            k:              RRF smoothing constant.

        Returns:
            Up to 10 merged result dicts sorted by descending RRF score.
            Each dict carries all keys from the source result plus an
            ``"rrf_score"`` float.
        """
        scores: dict[str, float] = {}
        # chunk_id → best source dict (dense preferred when both present)
        candidates: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id: str = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            candidates.setdefault(chunk_id, result)

        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            candidates.setdefault(chunk_id, result)

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        return [
            {**candidates[cid], "rrf_score": scores[cid]}
            for cid in sorted_ids[:10]
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks with Cohere Rerank v3.

        The Cohere reranker scores each chunk against the query using a
        cross-encoder model, which is significantly more accurate than
        bi-encoder similarity for final result selection.

        Args:
            query:  The user's natural-language question.
            chunks: Candidate chunks to rerank; each must contain ``"text"``.
            top_n:  Maximum number of results to return after reranking.

        Returns:
            Up to ``top_n`` chunk dicts sorted by descending relevance score,
            each with a ``"relevance_score"`` float added.
        """
        if not chunks:
            return []

        documents = [chunk["text"] for chunk in chunks]

        response = await self._cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked: list[dict[str, Any]] = []
        for result in response.results:
            chunk = dict(chunks[result.index])
            chunk["relevance_score"] = result.relevance_score
            reranked.append(chunk)

        return reranked

    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        final_top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Run the full hybrid retrieval pipeline for a user query.

        Steps:
        1. Embed the query with ``EmbeddingService``.
        2. Dense search: retrieve ``top_k_dense`` candidates from Qdrant.
        3. Sparse search: retrieve ``top_k_sparse`` candidates from BM25.
        4. Fuse both ranked lists with ``_reciprocal_rank_fusion``.
        5. Rerank the fused candidates with Cohere; return the best
           ``final_top_n``.

        Args:
            query:           Natural-language query string.
            collection_name: Qdrant collection to search.
            top_k_dense:     Number of dense candidates to retrieve.
            top_k_sparse:    Number of sparse candidates to retrieve.
            final_top_n:     Number of results to return after reranking.

        Returns:
            Up to ``final_top_n`` chunk dicts ordered by Cohere relevance
            score, each containing ``chunk_id``, ``text``, ``filename``,
            ``page_number``, ``chunk_index``, ``rrf_score``, and
            ``relevance_score``.
        """
        # Step 1 — embed query
        logger.info("Embedding query for hybrid search.")
        query_embedding = await self._embedding_service.embed_query(query)

        # Step 2 — dense search
        logger.info(
            "Dense search: top_k=%d, collection='%s'.", top_k_dense, collection_name
        )
        dense_results = await self._vector_store.dense_search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            top_k=top_k_dense,
        )
        logger.info("Dense search returned %d results.", len(dense_results))

        # Step 3 — sparse search
        logger.info("Sparse BM25 search: top_k=%d.", top_k_sparse)
        sparse_results = await self._bm25.sparse_search(
            query=query,
            top_k=top_k_sparse,
        )
        logger.info("Sparse search returned %d results.", len(sparse_results))

        # Step 4 — RRF fusion
        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )
        logger.info("RRF fusion produced %d merged candidates.", len(fused))

        # Step 5 — rerank
        logger.info("Reranking %d candidates (top_n=%d).", len(fused), final_top_n)
        final = await self.rerank(query=query, chunks=fused, top_n=final_top_n)
        logger.info("Reranking complete. Returning %d results.", len(final))

        return final
