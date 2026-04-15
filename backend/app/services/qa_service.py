"""Top-level QA orchestrator: retrieval → answer → citations → confidence."""

import logging
from typing import Any

from app.core.config import Settings
from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

_MULTI_DOC_TOP_N_PER_FILE = 10

# Words in the *query* that suggest a lower-bound / absence concept.
_NEGATIVE_QUESTION_WORDS: list[str] = [
    "minimum", "least", "lowest", "never", "prohibited",
    "not allowed", "cannot", "no fine", "no penalty",
]

# Words in the *answer* that suggest an upper-bound / presence concept,
# which would contradict a lower-bound question.
_POSITIVE_ANSWER_WORDS: list[str] = [
    "maximum", "up to", "at most", "highest", "allowed",
    "permitted", "can be", "may be",
]

# Phrases that indicate the answer already acknowledges missing info.
_CANNOT_FIND_PHRASES: list[str] = [
    "cannot find", "not specified", "not stated",
    "not mentioned", "not provided", "no information",
    "does not specify", "does not state",
]


class QAService:
    """Orchestrate the full RAG pipeline: retrieve → generate → cite → verify.

    Responsibilities:
    1. Hybrid retrieval via ``RetrievalService`` (dense + BM25 + RRF + rerank).
    2. Answerability pre-check + grounded answer generation via ``LLMService``.
    3. Structured citation extraction.
    4. Self-consistency / keyword-contradiction hallucination detection.
    5. Post-generation answer guard (polarity + answerability cross-check).

    BM25 index loading is deferred to the first query call (``_ensure_ready``)
    because ``__init__`` is synchronous and loading requires ``await``.
    """

    def __init__(self, settings: Settings) -> None:
        """Instantiate all pipeline dependencies from application settings.

        Args:
            settings: Application settings from ``core.config``.
        """
        self._settings = settings
        self._embedding_service = EmbeddingService(settings=settings)
        self._vector_store = VectorStoreService(settings=settings)
        self._bm25 = BM25Service()
        self._llm_service = LLMService(settings=settings)
        self._retrieval: RetrievalService | None = None
        self._ready = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    async def _ensure_ready(self) -> None:
        """Load BM25 indexes and build ``RetrievalService`` on first call.

        Subsequent calls are no-ops (guarded by ``self._ready``).
        """
        if self._ready:
            return

        logger.info("QAService: loading BM25 indexes from disk.")
        await self._bm25.load_all_indexes("bm25_indexes")

        self._retrieval = RetrievalService(
            settings=self._settings,
            vector_store=self._vector_store,
            bm25=self._bm25,
            embedding_service=self._embedding_service,
        )
        self._ready = True
        logger.info("QAService: ready.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_relevance_into_citations(
        self,
        citations: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach ``relevance_score`` from retrieved chunks to each citation.

        Args:
            citations: Output of ``LLMService.extract_citations``.
            chunks:    Reranked chunks from ``RetrievalService``.

        Returns:
            Citations with ``relevance_score`` added (0.0 when not matched).
        """
        score_map: dict[str, float] = {
            c["chunk_id"]: float(c.get("relevance_score", 0.0)) for c in chunks
        }
        return [
            {**cit, "relevance_score": score_map.get(cit.get("chunk_id", ""), 0.0)}
            for cit in citations
        ]

    def _apply_answer_guard(
        self,
        query: str,
        answer: str,
        answerability: dict[str, Any],
    ) -> dict[str, Any]:
        """Post-generation safety check that catches two failure modes.

        **Negation guard** — triggers when the query contains a lower-bound
        keyword (e.g. "minimum") and the answer contains an upper-bound
        keyword (e.g. "maximum").  This catches cases where the model answered
        a semantically adjacent but factually opposite question.

        **Existence guard** — triggers when the answerability pre-check
        returned ``is_answerable=False`` but the generated answer does not
        contain any "cannot find" acknowledgement phrases.  This means the
        model ignored the pre-check signal and generated a potentially
        hallucinated response.

        Args:
            query:          The original user question (lower-cased internally).
            answer:         The LLM-generated answer to inspect.
            answerability:  Dict returned by ``LLMService.generate_answer``
                            containing the pre-generation answerability result.

        Returns:
            Dict with keys:

            - ``flagged`` – ``True`` when either guard condition fires.
            - ``reason``  – Human-readable description of why it fired, or
                            ``None`` when not flagged.
        """
        query_lower = query.lower()
        answer_lower = answer.lower()

        # --- Negation guard ---
        query_has_negative = any(w in query_lower for w in _NEGATIVE_QUESTION_WORDS)
        answer_has_positive = any(w in answer_lower for w in _POSITIVE_ANSWER_WORDS)

        if query_has_negative and answer_has_positive:
            matched_q = next(w for w in _NEGATIVE_QUESTION_WORDS if w in query_lower)
            matched_a = next(w for w in _POSITIVE_ANSWER_WORDS if w in answer_lower)
            return {
                "flagged": True,
                "reason": (
                    f"Answer polarity mismatch — question asks about '{matched_q}' "
                    f"but answer describes '{matched_a}' (the opposite extreme)."
                ),
            }

        # --- Existence guard ---
        was_unanswerable = not answerability.get("is_answerable", True)
        answer_acknowledges_missing = any(
            phrase in answer_lower for phrase in _CANNOT_FIND_PHRASES
        )

        if was_unanswerable and not answer_acknowledges_missing:
            return {
                "flagged": True,
                "reason": (
                    "Answerability pre-check determined the context does not "
                    "directly answer this question, but the generated answer "
                    "did not acknowledge this limitation."
                ),
            }

        return {"flagged": False, "reason": None}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer_query(
        self,
        query: str,
        collection_name: str,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline with hallucination guards for a single query.

        Pipeline:
        1. Hybrid search → top reranked chunks.
        2. Answerability pre-check + answer generation (``LLMService``).
        3. Post-generation answer guard (polarity + existence checks).
        4. Citation extraction.
        5. Self-consistency hallucination check.

        If the answer guard fires the answer is replaced with a safe fallback
        and confidence is overridden to 0.1.

        Args:
            query:           Natural-language question.
            collection_name: Qdrant collection to search.
            top_n:           Number of chunks to retrieve and use as context.

        Returns:
            Dict with keys:

            - ``query``                       – The original question.
            - ``answer``                      – Grounded answer or safe fallback.
            - ``citations``                   – Structured source references.
            - ``confidence``                  – Consistency check result.
            - ``retrieved_chunks_count``      – Chunks used as LLM context.
            - ``model_used``                  – Name of the LLM.
            - ``hallucination_guard_triggered``– ``True`` when the guard fired.
            - ``guard_reason``                – Why the guard fired, or ``None``.
        """
        await self._ensure_ready()
        assert self._retrieval is not None

        # Step 1 — Retrieve
        logger.info("answer_query: retrieving chunks for query: %.80s", query)
        chunks = await self._retrieval.hybrid_search(
            query=query,
            collection_name=collection_name,
            final_top_n=top_n,
        )
        logger.info("answer_query: retrieved %d chunk(s).", len(chunks))

        # Step 2 — Answerability check + answer generation
        logger.info("answer_query: generating answer.")
        answer_result = await self._llm_service.generate_answer(
            query=query,
            chunks=chunks,
        )
        answer: str = answer_result["answer"]
        answerability: dict[str, Any] = answer_result.get("answerability", {})

        # Step 3 — Post-generation guard
        guard = self._apply_answer_guard(query, answer, answerability)
        guard_triggered = guard["flagged"]

        if guard_triggered:
            logger.warning(
                "Answer guard triggered for query '%.80s': %s",
                query,
                guard["reason"],
            )
            answer = (
                "I cannot confidently answer this question based on the provided "
                f"documents. {guard['reason']} "
                "The documents do contain related information — please rephrase "
                "your question or ask about what the documents explicitly state."
            )

        # Step 4 — Citation extraction
        logger.info("answer_query: extracting citations.")
        raw_citations = await self._llm_service.extract_citations(
            answer=answer,
            chunks=chunks,
        )
        citations = self._merge_relevance_into_citations(raw_citations, chunks)

        # Step 5 — Hallucination check
        logger.info("answer_query: running hallucination check.")
        confidence = await self._llm_service.check_hallucination(
            query=query,
            chunks=chunks,
            original_answer=answer,
        )

        # Override confidence when guard fired.
        if guard_triggered:
            confidence = {
                "is_consistent": False,
                "confidence_score": 0.1,
                "warning": guard["reason"],
            }

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "retrieved_chunks_count": len(chunks),
            "model_used": "gpt-4o-mini",
            "hallucination_guard_triggered": guard_triggered,
            "guard_reason": guard["reason"],
        }

    async def multi_document_compare(
        self,
        query: str,
        filenames: list[str],
        collection_name: str,
    ) -> dict[str, Any]:
        """Answer the same query independently for each specified document.

        Retrieves a broad candidate set across all documents, then
        post-filters by filename so each document is answered using only its
        own content.

        Args:
            query:           Natural-language question to answer for each file.
            filenames:       Filenames to compare (must match Qdrant payload).
            collection_name: Qdrant collection to search.

        Returns:
            Dict with ``query`` and ``comparisons`` (list of per-document
            answer dicts containing ``filename``, ``answer``, ``citations``).
        """
        await self._ensure_ready()
        assert self._retrieval is not None

        broad_top_n = max(len(filenames) * _MULTI_DOC_TOP_N_PER_FILE, 20)
        logger.info(
            "multi_document_compare: fetching %d candidates for %d file(s).",
            broad_top_n,
            len(filenames),
        )
        all_chunks = await self._retrieval.hybrid_search(
            query=query,
            collection_name=collection_name,
            top_k_dense=broad_top_n * 2,
            top_k_sparse=broad_top_n * 2,
            final_top_n=broad_top_n,
        )

        comparisons: list[dict[str, Any]] = []

        for filename in filenames:
            doc_chunks = [c for c in all_chunks if c.get("filename") == filename]

            if not doc_chunks:
                logger.warning(
                    "multi_document_compare: no chunks found for '%s'.", filename
                )
                comparisons.append(
                    {
                        "filename": filename,
                        "answer": "No content found for this document in the search results.",
                        "citations": [],
                    }
                )
                continue

            logger.info(
                "multi_document_compare: generating answer for '%s' (%d chunk(s)).",
                filename,
                len(doc_chunks),
            )
            answer_result = await self._llm_service.generate_answer(
                query=query,
                chunks=doc_chunks,
            )
            raw_citations = await self._llm_service.extract_citations(
                answer=answer_result["answer"],
                chunks=doc_chunks,
            )
            citations = self._merge_relevance_into_citations(raw_citations, doc_chunks)

            comparisons.append(
                {
                    "filename": filename,
                    "answer": answer_result["answer"],
                    "citations": citations,
                }
            )

        return {"query": query, "comparisons": comparisons}
