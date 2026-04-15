"""Embedding service for generating OpenAI vector embeddings."""

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_MAX_RETRIES = 2


class EmbeddingService:
    """Generate vector embeddings for document chunks and queries.

    Wraps ``langchain_openai.OpenAIEmbeddings`` with batching and a single
    retry so the ingestion pipeline is resilient to transient API errors
    without silently swallowing failures.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the OpenAI embeddings client.

        Args:
            settings: Application settings; the OpenAI API key is read from
                      here rather than the environment so the service remains
                      independently testable.
        """
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )

    async def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch-embed all chunks and attach the embedding vector to each.

        Chunks are processed in batches of ``_BATCH_SIZE`` (100) to stay
        within OpenAI rate limits.  Each batch is retried once on failure
        before the exception is re-raised, stopping the pipeline with a clear
        error rather than storing incomplete data.

        Args:
            chunks: Output of ``IngestionService.ingest_document``; each dict
                    must contain at minimum a ``"text"`` key.

        Returns:
            The same list of dicts with an ``"embedding"`` key added to every
            element.  The list order and all existing keys are preserved.

        Raises:
            Exception: Re-raised from the OpenAI client after one failed retry.
        """
        if not chunks:
            return chunks

        enriched: list[dict[str, Any]] = [dict(chunk) for chunk in chunks]

        for batch_start in range(0, len(enriched), _BATCH_SIZE):
            batch = enriched[batch_start : batch_start + _BATCH_SIZE]
            texts = [chunk["text"] for chunk in batch]
            batch_num = batch_start // _BATCH_SIZE + 1
            total_batches = (len(enriched) + _BATCH_SIZE - 1) // _BATCH_SIZE

            vectors = await self._embed_texts_with_retry(
                texts=texts,
                label=f"chunk batch {batch_num}/{total_batches}",
            )

            for chunk, vector in zip(batch, vectors):
                chunk["embedding"] = vector

        return enriched

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for similarity search.

        Args:
            query: The user's natural-language question.

        Returns:
            A list of floats representing the query's embedding vector.

        Raises:
            Exception: Re-raised from the OpenAI client after one failed retry.
        """
        vectors = await self._embed_texts_with_retry(
            texts=[query],
            label="query",
        )
        return vectors[0]

    async def _embed_texts_with_retry(
        self,
        texts: list[str],
        label: str,
    ) -> list[list[float]]:
        """Call the embeddings API with one retry on failure.

        ``OpenAIEmbeddings.aembed_documents`` is used for all calls
        (including single queries) to keep the implementation uniformly async.

        Args:
            texts: List of strings to embed.
            label: Human-readable description of the batch used in log messages.

        Returns:
            List of embedding vectors in the same order as ``texts``.

        Raises:
            Exception: The exception from the second (retry) attempt if both
                       calls fail.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                vectors: list[list[float]] = await self._embeddings.aembed_documents(
                    texts
                )
                if attempt > 1:
                    logger.info(
                        "Embedding succeeded on retry %d for %s.", attempt, label
                    )
                return vectors
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Embedding failed for %s (attempt %d/%d): %s — retrying.",
                        label,
                        attempt,
                        _MAX_RETRIES,
                        exc,
                    )
                else:
                    logger.error(
                        "Embedding failed for %s after %d attempts: %s",
                        label,
                        _MAX_RETRIES,
                        exc,
                    )
                    raise
