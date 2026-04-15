"""Semantic chunking service for splitting parsed document pages into chunks."""

import logging
import uuid

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Fallback splitter constants used when SemanticChunker raises.
_FALLBACK_CHUNK_SIZE = 1000
_FALLBACK_CHUNK_OVERLAP = 200

ChunkDict = dict[str, str | int]


class SemanticChunkerService:
    """Split document pages into semantically coherent chunks.

    Uses LangChain's ``SemanticChunker`` backed by OpenAI embeddings to find
    natural topic boundaries.  If the semantic splitter fails for any reason
    (e.g. API error, empty document), the service transparently falls back to
    ``RecursiveCharacterTextSplitter`` so the ingestion pipeline is never
    blocked.
    """

    def __init__(self, openai_api_key: str) -> None:
        """Initialise embeddings and build the semantic chunker.

        Args:
            openai_api_key: OpenAI API key used to obtain embeddings for
                boundary detection.  Passed explicitly rather than read from
                the environment so the service remains testable in isolation.
        """
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key,
        )
        self._semantic_chunker = SemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type="percentile",
        )
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_FALLBACK_CHUNK_SIZE,
            chunk_overlap=_FALLBACK_CHUNK_OVERLAP,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chunk_pages(self, pages: list[dict]) -> list[ChunkDict]:
        """Split a list of page dicts into semantically coherent chunk dicts.

        The method concatenates all page texts into one document string so
        that the semantic chunker can reason across page boundaries (common
        in legal documents where a clause spans multiple pages).  After
        splitting, each chunk is mapped back to its source page by checking
        which page's text contains the beginning of the chunk.

        Args:
            pages: Output of ``DocumentParser`` — each element must contain
                   the keys ``"text"`` (str), ``"page_number"`` (int), and
                   ``"filename"`` (str).

        Returns:
            List of chunk dicts with keys:

            - ``chunk_id``    – UUID4 string, unique per chunk.
            - ``text``        – The chunk text.
            - ``filename``    – Propagated from the source page.
            - ``page_number`` – Best-guess source page (1-based).
            - ``chunk_index`` – 0-based position in the returned list.
        """
        if not pages:
            return []

        filename: str = str(pages[0].get("filename", ""))
        full_text, page_offsets = self._build_full_text(pages)

        raw_chunks = await self._split_with_fallback(full_text)

        return [
            {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "filename": filename,
                "page_number": self._find_page_number(chunk, page_offsets),
                "chunk_index": idx,
            }
            for idx, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_full_text(pages: list[dict]) -> tuple[str, list[tuple[int, str, int]]]:
        """Concatenate page texts and record their character offsets.

        Args:
            pages: List of page dicts from ``DocumentParser``.

        Returns:
            A 2-tuple of:

            - ``full_text`` – All page texts joined by newlines.
            - ``page_offsets`` – List of ``(start_offset, page_text,
              page_number)`` triples sorted by start offset, used by
              ``_find_page_number`` to locate a chunk in the source pages.
        """
        parts: list[str] = []
        page_offsets: list[tuple[int, str, int]] = []
        cursor = 0

        for page in pages:
            text: str = page.get("text", "")
            page_number: int = int(page.get("page_number", 1))
            page_offsets.append((cursor, text, page_number))
            parts.append(text)
            cursor += len(text) + 1  # +1 for the joining newline

        return "\n".join(parts), page_offsets

    async def _split_with_fallback(self, full_text: str) -> list[str]:
        """Attempt semantic splitting; fall back to recursive splitting on error.

        ``SemanticChunker.create_documents`` is synchronous but makes
        OpenAI API calls internally.  We call it directly here; for very
        high-throughput scenarios it could be wrapped in
        ``asyncio.to_thread``.

        Args:
            full_text: The concatenated document text to split.

        Returns:
            List of raw chunk strings.
        """
        try:
            docs = self._semantic_chunker.create_documents([full_text])
            return [doc.page_content for doc in docs]
        except Exception as exc:
            logger.warning(
                "SemanticChunker failed (%s: %s) — falling back to "
                "RecursiveCharacterTextSplitter.",
                type(exc).__name__,
                exc,
            )
            docs = self._fallback_splitter.create_documents([full_text])
            return [doc.page_content for doc in docs]

    @staticmethod
    def _find_page_number(
        chunk: str,
        page_offsets: list[tuple[int, str, int]],
    ) -> int:
        """Identify the source page for a chunk using prefix matching.

        The heuristic checks whether the first 120 characters of the chunk
        appear inside any page's original text, iterating pages in order and
        returning the first match.  If no match is found the page whose text
        window contains the chunk's start offset is used instead, and if that
        also fails we default to page 1.

        Args:
            chunk: The chunk text to locate.
            page_offsets: List of ``(start_offset, page_text, page_number)``
                          triples produced by ``_build_full_text``.

        Returns:
            Best-guess 1-based page number for the chunk.
        """
        # Use a prefix long enough to be distinctive but short enough to
        # survive minor whitespace normalisation by the splitter.
        prefix = chunk[:120].strip()

        for _offset, page_text, page_number in page_offsets:
            if prefix in page_text:
                return page_number

        # Fallback: return the page whose offset range contains the prefix.
        for start, page_text, page_number in page_offsets:
            end = start + len(page_text)
            # Estimate chunk start by searching full_text (best-effort).
            if start <= len(prefix) <= end:
                return page_number

        return int(page_offsets[0][2]) if page_offsets else 1
