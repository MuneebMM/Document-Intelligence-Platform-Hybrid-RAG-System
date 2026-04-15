"""Ingestion pipeline: parse → chunk → embed → store → index."""

import os
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.services.bm25_service import BM25Service
from app.services.chunker import SemanticChunkerService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

_BM25_INDEX_DIR = Path("bm25_indexes")

IngestionResult = dict[str, str | int | list[dict]]


class IngestionService:
    """Orchestrate the full document ingestion pipeline.

    Responsibilities (in order):
    1. Parse raw bytes into page-level text (``DocumentParser``).
    2. Semantically chunk pages (``SemanticChunkerService``).
    3. Generate OpenAI embeddings for every chunk (``EmbeddingService``).
    4. Ensure the Qdrant collection exists and upsert vectors
       (``VectorStoreService``).
    5. Build and persist a BM25 index to disk (``BM25Service``).
    """

    def __init__(self, settings: Settings) -> None:
        """Instantiate all pipeline services from application settings.

        Args:
            settings: Application settings instance from ``core.config``.
                      API keys and connection parameters are forwarded to
                      each service so nothing reads environment variables
                      directly.
        """
        self._settings = settings
        self._parser = DocumentParser()
        self._chunker = SemanticChunkerService(
            openai_api_key=settings.openai_api_key,
        )
        self._embedding_service = EmbeddingService(settings=settings)
        self._vector_store = VectorStoreService(settings=settings)
        self._bm25 = BM25Service()

        # Ensure the BM25 persistence directory exists at startup.
        _BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    async def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: Literal[".pdf", ".docx", ".txt"],
    ) -> IngestionResult:
        """Run the complete parse → chunk → embed → store → index pipeline.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename; propagated into every chunk and
                        used to name the BM25 index file on disk.
            file_type:  Lowercase file extension including the leading dot.
                        Accepted values: ``".pdf"``, ``".docx"``, ``".txt"``.

        Returns:
            A summary dict with keys:

            - ``filename``       – The original filename.
            - ``total_pages``    – Pages extracted by the parser.
            - ``total_chunks``   – Chunks produced by the chunker.
            - ``vectors_stored`` – Points upserted into Qdrant.
            - ``status``         – ``"fully_ingested"`` on success.
            - ``chunks``         – Full list of enriched chunk dicts.

        Raises:
            ValueError: Propagated from ``DocumentParser`` for unsupported
                        file types.
        """
        # Step 1 — Parse
        print(f"[IngestionService] Parsing '{filename}' (type={file_type})")
        pages = await self._parser.parse_document(file_bytes, filename, file_type)
        print(f"[IngestionService] Parsed {len(pages)} page(s) from '{filename}'")

        # Step 2 — Chunk
        print(f"[IngestionService] Chunking {len(pages)} page(s) from '{filename}'")
        chunks = await self._chunker.chunk_pages(pages)
        print(f"[IngestionService] Produced {len(chunks)} chunk(s) from '{filename}'")

        # Step 3 — Embed
        print(f"[IngestionService] Embedding {len(chunks)} chunk(s) from '{filename}'")
        embedded_chunks = await self._embedding_service.embed_chunks(chunks)
        print(f"[IngestionService] Embeddings generated for '{filename}'")

        # Step 4 — Ensure collection exists
        collection_name = self._settings.qdrant_collection_name
        print(
            f"[IngestionService] Ensuring Qdrant collection '{collection_name}' exists"
        )
        await self._vector_store.create_collection(
            collection_name=collection_name,
            vector_size=1536,
        )

        # Step 5 — Upsert into Qdrant
        print(
            f"[IngestionService] Upserting {len(embedded_chunks)} vectors "
            f"into '{collection_name}'"
        )
        vectors_stored = await self._vector_store.upsert_chunks(
            chunks=embedded_chunks,
            collection_name=collection_name,
        )
        print(f"[IngestionService] Stored {vectors_stored} vectors for '{filename}'")

        # Step 6 — Build BM25 index and persist to disk
        print(f"[IngestionService] Building BM25 index for '{filename}'")
        await self._bm25.build_index(embedded_chunks)

        safe_stem = Path(filename).stem
        bm25_path = str(_BM25_INDEX_DIR / f"{safe_stem}.pkl")
        await self._bm25.save_index(bm25_path)
        print(f"[IngestionService] BM25 index saved to '{bm25_path}'")

        return {
            "filename": filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "vectors_stored": vectors_stored,
            "status": "fully_ingested",
            "chunks": embedded_chunks,
        }
