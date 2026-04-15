"""BM25 sparse retrieval service backed by rank_bm25."""

import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Service:
    """Build and query a BM25 sparse retrieval index over document chunks.

    The index is held in memory and can be persisted to / restored from disk
    with ``save_index`` / ``load_index`` so a full rebuild is not required on
    every process restart.

    ``build_index``, ``sparse_search``, ``save_index``, and ``load_index``
    are all synchronous operations (rank_bm25 and pickle have no async APIs),
    but they are exposed as ``async`` methods so callers in the async
    ingestion and query pipelines can ``await`` them uniformly.  If corpus
    size grows large enough to block the event loop, the bodies can be
    offloaded to ``asyncio.to_thread`` without changing the public interface.
    """

    def __init__(self) -> None:
        """Initialise the service with an empty index and corpus."""
        self.bm25_index: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []

    async def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Tokenize chunks and build a BM25Okapi index.

        Tokenization is intentionally simple (lowercase + whitespace split) to
        match what ``sparse_search`` does at query time.  Both sides must use
        the same tokenizer or scores will be wrong.

        Args:
            chunks: List of chunk dicts; each must contain a ``"text"`` key.
                    All other keys are retained and returned by
                    ``sparse_search``.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list — index cleared.")
            self.bm25_index = None
            self._chunks = []
            return

        self._chunks = chunks
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d chunks.", len(chunks))

    async def sparse_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the top-k chunks most relevant to ``query`` by BM25 score.

        Scores are normalised to [0, 1] by dividing by the maximum score in
        the result set so they can be combined with dense similarity scores
        during hybrid fusion without unit mismatch.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts ordered by descending BM25 score, each
            containing ``chunk_id``, ``text``, ``filename``, ``page_number``,
            ``chunk_index``, and ``score`` (normalised float in [0, 1]).

        Raises:
            RuntimeError: If ``build_index`` has not been called yet.
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 index is not built. Call build_index() before searching."
            )

        tokenized_query = _tokenize(query)
        scores: list[float] = self.bm25_index.get_scores(tokenized_query).tolist()

        # Pair each score with its chunk and sort descending.
        scored = sorted(
            zip(scores, self._chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )
        top = scored[:top_k]

        # Normalise scores to [0, 1].
        max_score = top[0][0] if top and top[0][0] > 0 else 1.0
        return [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "filename": chunk.get("filename", ""),
                "page_number": chunk.get("page_number", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "score": score / max_score,
            }
            for score, chunk in top
        ]

    async def save_index(self, filepath: str) -> None:
        """Persist the BM25 index and corpus to disk using pickle.

        Args:
            filepath: Destination path for the pickle file.  Parent
                      directories are created automatically.

        Raises:
            RuntimeError: If there is no index to save.
        """
        if self.bm25_index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"bm25_index": self.bm25_index, "chunks": self._chunks}
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

        logger.info("BM25 index saved to '%s' (%d chunks).", filepath, len(self._chunks))

    async def load_all_indexes(self, directory: str) -> None:
        """Load every pkl file in ``directory`` and build a combined index.

        Each pkl file stores the chunks for one ingested document.  This
        method collects all chunks across every file and rebuilds a single
        BM25 index over the full corpus so queries span all ingested
        documents.  Files that fail to load are skipped with a warning so
        one corrupt file does not block the entire retrieval path.

        Args:
            directory: Path to the directory containing ``*.pkl`` files
                       written by ``save_index``.
        """
        index_dir = Path(directory)
        if not index_dir.exists():
            logger.warning(
                "BM25 index directory '%s' does not exist — index not loaded.",
                directory,
            )
            return

        pkl_files = sorted(index_dir.glob("*.pkl"))
        if not pkl_files:
            logger.warning("No BM25 index files found in '%s'.", directory)
            return

        all_chunks: list[dict[str, Any]] = []
        for pkl_path in pkl_files:
            try:
                with pkl_path.open("rb") as fh:
                    payload: dict[str, Any] = pickle.load(fh)
                all_chunks.extend(payload["chunks"])
                logger.info(
                    "Loaded %d chunks from '%s'.", len(payload["chunks"]), pkl_path.name
                )
            except Exception as exc:
                logger.warning("Failed to load BM25 index '%s': %s", pkl_path, exc)

        if all_chunks:
            await self.build_index(all_chunks)
            logger.info(
                "Combined BM25 index built from %d file(s), %d total chunks.",
                len(pkl_files),
                len(all_chunks),
            )

    async def load_index(self, filepath: str) -> None:
        """Restore a BM25 index and corpus from a pickle file.

        Args:
            filepath: Path to a pickle file previously written by
                      ``save_index``.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: '{filepath}'")

        with path.open("rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)

        self.bm25_index = payload["bm25_index"]
        self._chunks = payload["chunks"]
        logger.info(
            "BM25 index loaded from '%s' (%d chunks).", filepath, len(self._chunks)
        )


def _tokenize(text: str) -> list[str]:
    """Lowercase and whitespace-tokenize a string.

    This same function is used for both indexing and querying so the
    tokenization is always symmetric.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens split on whitespace.
    """
    return text.lower().split()
