"""Qdrant vector store service for upserting and searching document chunks."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 100


class VectorStoreService:
    """Manage document chunk storage and dense vector search in Qdrant.

    All operations use ``AsyncQdrantClient`` so they compose naturally with
    the async FastAPI / ingestion pipeline without blocking the event loop.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the async Qdrant client.

        Args:
            settings: Application settings; ``qdrant_host`` and ``qdrant_port``
                      are used to connect to the running Qdrant instance.
        """
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
    ) -> None:
        """Create a Qdrant collection if one with that name does not yet exist.

        Skips creation silently when the collection is already present so the
        method is safe to call on every application startup.

        Args:
            collection_name: Name of the collection to create.
            vector_size:     Dimensionality of the stored vectors.  Defaults
                             to 1536 (``text-embedding-3-small`` output size).
        """
        existing = await self._client.get_collections()
        existing_names = {col.name for col in existing.collections}

        if collection_name in existing_names:
            logger.info("Collection '%s' already exists — skipping.", collection_name)
            return

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (vector_size=%d, distance=COSINE).",
            collection_name,
            vector_size,
        )

    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        collection_name: str,
    ) -> int:
        """Upsert embedded chunks into a Qdrant collection in batches.

        Each chunk is converted to a ``PointStruct`` whose ``id`` is derived
        from the first 8 hex characters of the ``chunk_id`` UUID.  The full
        ``chunk_id`` is preserved in the payload so it can be recovered during
        retrieval.

        Args:
            chunks:          Embedded chunk dicts; each must contain
                             ``chunk_id``, ``text``, ``filename``,
                             ``page_number``, ``chunk_index``, and
                             ``embedding`` keys.
            collection_name: Target Qdrant collection.

        Returns:
            Total number of points successfully upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0

        for batch_start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _UPSERT_BATCH_SIZE]

            points = [
                PointStruct(
                    id=_chunk_uuid_to_int(chunk["chunk_id"]),
                    vector=chunk["embedding"],
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
                for chunk in batch
            ]

            await self._client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_upserted += len(points)
            logger.info(
                "Upserted %d/%d points into '%s'.",
                total_upserted,
                len(chunks),
                collection_name,
            )

        return total_upserted

    async def dense_search(
        self,
        query_embedding: list[float],
        collection_name: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Search a Qdrant collection using a dense query vector.

        Args:
            query_embedding: Embedding vector produced by ``EmbeddingService``.
            collection_name: Collection to search.
            top_k:           Maximum number of results to return.

        Returns:
            List of result dicts ordered by descending similarity score, each
            containing ``chunk_id``, ``text``, ``filename``, ``page_number``,
            ``chunk_index``, and ``score``.
        """
        results = await self._client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id", ""),
                "text": hit.payload.get("text", ""),
                "filename": hit.payload.get("filename", ""),
                "page_number": hit.payload.get("page_number", 0),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": hit.score,
            }
            for hit in results
        ]


def _chunk_uuid_to_int(chunk_id: str) -> int:
    """Convert a UUID4 string to a uint64 suitable for a Qdrant point ID.

    Qdrant point IDs must be unsigned 64-bit integers or UUID strings.
    We pass the raw UUID string directly — Qdrant accepts UUID format natively.
    This helper converts to int as a fallback for clients that require numeric IDs
    by taking the first 16 hex digits (64 bits) of the UUID.

    Args:
        chunk_id: UUID4 string, e.g. ``"550e8400-e29b-41d4-a716-446655440000"``.

    Returns:
        An unsigned 64-bit integer derived from the UUID.
    """
    return int(chunk_id.replace("-", "")[:16], 16)
