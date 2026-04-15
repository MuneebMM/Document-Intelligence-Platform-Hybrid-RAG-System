"""LLM service: answer generation, citation extraction, and hallucination detection."""

import logging
import re
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.services.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

# Pairs of (question_keyword, answer_keyword) that signal the answer
# addressed the opposite of what was asked.
_POLARITY_PAIRS: list[tuple[str, str]] = [
    ("minimum", "maximum"),
    ("maximum", "minimum"),
    ("lowest", "highest"),
    ("highest", "lowest"),
    ("least", "most"),
    ("most", "least"),
    ("never", "always"),
    ("always", "never"),
    ("prohibited", "permitted"),
    ("permitted", "prohibited"),
    ("not allowed", "allowed"),
    ("cannot", "can"),
]


class LLMService:
    """Orchestrate LLM calls for answer generation, citation extraction, and
    self-consistency / hallucination detection.

    All chains are built using LangChain LCEL (pipe operator ``|``) as
    Uses LCEL (pipe operator) — no deprecated v1 LLMChain usage.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the LLM client and supporting components.

        Args:
            settings: Application settings; the OpenAI API key is read from
                      here so the service is independently testable.
        """
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=800,
            openai_api_key=settings.openai_api_key,
        )
        self._templates = PromptTemplates()
        self._json_parser = JsonOutputParser()
        self._str_parser = StrOutputParser()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove characters that are invalid in a JSON string body.

        PDF extraction can leave null bytes (``\\x00``) and other C0 control
        characters in chunk text.  When LangChain serialises the prompt as
        JSON to send to OpenAI these characters produce a malformed request
        body, resulting in a 400 "could not parse JSON body" error.

        Keeps printable ASCII, standard whitespace (space, tab, newline,
        carriage-return), and all Unicode above U+001F.

        Args:
            text: Raw chunk text that may contain control characters.

        Returns:
            Sanitized string safe for JSON embedding.
        """
        return "".join(
            ch for ch in text
            if ch in (" ", "\t", "\n", "\r") or (ord(ch) > 0x1F)
        )

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Render retrieved chunks into a numbered context block for the prompt.

        Args:
            chunks: Retrieved chunk dicts; each must contain ``filename``,
                    ``page_number``, and ``text`` keys.

        Returns:
            A multi-line string with one section per chunk, each prefixed by
            its source metadata and terminated by ``---``.
        """
        parts: list[str] = []
        for chunk in chunks:
            parts.append(
                f"[Source: {chunk.get('filename', 'unknown')}, "
                f"Page {chunk.get('page_number', '?')}]\n"
                f"{self._sanitize_text(chunk.get('text', ''))}\n"
                "---"
            )
        return "\n".join(parts)

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract key entities used for self-consistency comparison.

        Entities are: numbers/percentages, dates, and article/section/clause
        references.  These are the elements most likely to differ between two
        answers if one has hallucinated.

        Args:
            text: Raw answer text to scan.

        Returns:
            A set of lower-cased entity strings found in ``text``.
        """
        patterns = [
            r"\b\d+(?:[.,]\d+)?%?\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b(?:article|art\.|section|clause|recital)\s+\d+\w*\b",
        ]
        entities: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.add(match.group().lower().strip())
        return entities

    @staticmethod
    def _detect_keyword_contradiction(
        query: str,
        answer: str,
    ) -> dict[str, Any] | None:
        """Check whether the answer addresses the opposite polarity to the query.

        Scans for known antonym pairs (minimum/maximum, never/always, etc.).
        If the query contains word A and the answer contains its opposite word B,
        the answer likely addressed a different aspect of the topic.

        Args:
            query:  The original user question.
            answer: The LLM-generated answer to check.

        Returns:
            A dict with ``triggered`` (bool), ``query_keyword``, and
            ``answer_keyword`` if a contradiction is found, otherwise ``None``.
        """
        query_lower = query.lower()
        answer_lower = answer.lower()

        for query_kw, answer_kw in _POLARITY_PAIRS:
            if re.search(rf"\b{re.escape(query_kw)}\b", query_lower) and \
               re.search(rf"\b{re.escape(answer_kw)}\b", answer_lower):
                return {
                    "triggered": True,
                    "query_keyword": query_kw,
                    "answer_keyword": answer_kw,
                }
        return None

    # ------------------------------------------------------------------
    # Private async methods
    # ------------------------------------------------------------------

    async def _check_question_answerability(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Determine whether the retrieved context directly answers the query.

        Runs a pre-generation LLM check asking the model to judge — before
        any answer is produced — whether the chunks contain a *direct* answer
        to the *exact* question asked.  Returning ``is_answerable: False``
        allows ``generate_answer`` to short-circuit and return a structured
        "cannot find" response rather than risk a hallucinated answer.

        LCEL chain: ``answerability_prompt | llm | JsonOutputParser``

        Args:
            query:  The user's natural-language question.
            chunks: Retrieved chunks that will be used as context.

        Returns:
            Dict with keys:

            - ``is_answerable``              – ``True`` only when a direct answer
                                               exists in the context.
            - ``reason``                     – Why the context is or is not sufficient.
            - ``what_document_actually_says``– Related information present, if any.
        """
        context = self._format_context(chunks)
        chain = (
            self._templates.get_answerability_prompt()
            | self._llm
            | self._json_parser
        )

        try:
            result: Any = await chain.ainvoke({"query": query, "context": context})
            logger.info(
                "Answerability check: is_answerable=%s, reason=%.80s",
                result.get("is_answerable"),
                result.get("reason", ""),
            )
            return {
                "is_answerable": bool(result.get("is_answerable", True)),
                "reason": result.get("reason", ""),
                "what_document_actually_says": result.get(
                    "what_document_actually_says", ""
                ),
            }
        except Exception as exc:
            logger.warning(
                "Answerability check failed (%s: %s) — defaulting to answerable.",
                type(exc).__name__,
                exc,
            )
            return {
                "is_answerable": True,
                "reason": "check failed",
                "what_document_actually_says": "",
            }

    async def _strip_ungrounded_claims(
        self,
        answer: str,
        context: str,
    ) -> str:
        """Remove sentences from the answer that are not supported by context.

        Runs a fast secondary LLM call that acts as a strict editor: it
        keeps only sentences directly grounded in the provided context and
        deletes everything else.  Nothing new is added.

        Args:
            answer:  The raw LLM-generated answer.
            context: The formatted context string used to generate the answer.

        Returns:
            Cleaned answer with only context-grounded sentences.  Returns the
            original answer unchanged if the editing call fails.
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict editor. Remove any sentence from the answer "
                "that is NOT directly supported by the provided context. Keep "
                "sentences that ARE supported. Do not add anything new. Return "
                "only the cleaned answer.",
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Answer to clean:\n{answer}\n\n"
                "Return the answer with only context-grounded sentences. "
                "If a sentence has ANY information not in context, remove it entirely.",
            ),
        ])

        chain = prompt | self._llm | self._str_parser

        try:
            cleaned: str = await chain.ainvoke({
                "context": context,
                "answer": answer,
            })
            logger.info(
                "Faithfulness filter: %d chars → %d chars.",
                len(answer),
                len(cleaned),
            )
            return cleaned
        except Exception as exc:
            logger.warning(
                "Faithfulness filter failed (%s: %s) — using original answer.",
                type(exc).__name__,
                exc,
            )
            return answer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a grounded answer from retrieved chunks using the RAG prompt.

        Before invoking the RAG chain, runs ``_check_question_answerability``
        to determine whether the context contains a direct answer.  If it does
        not, returns a structured "cannot find" response immediately without
        calling the LLM for an answer, preventing confident hallucinations.

        LCEL chain (when answerable): ``rag_prompt | llm | StrOutputParser``

        Args:
            query:  The user's natural-language question.
            chunks: Retrieved and reranked chunks from ``RetrievalService``.

        Returns:
            Dict with keys:

            - ``answer``        – The LLM-generated answer, or a structured
                                  "cannot find" message.
            - ``context_used``  – The chunk list passed to the prompt.
            - ``answerability`` – Result of the pre-generation answerability
                                  check.
        """
        logger.info("Running answerability check for query: %.80s", query)
        answerability = await self._check_question_answerability(query, chunks)

        if not answerability["is_answerable"]:
            related = answerability.get("what_document_actually_says", "").strip()
            related_clause = (
                f" {related}" if related else
                " Please check the document directly for this specific detail."
            )
            answer = (
                f"I cannot find information about '{query}' in the provided documents."
                f"{related_clause}"
            )
            logger.info("Answerability check returned False — skipping RAG chain.")
            return {
                "answer": answer,
                "context_used": chunks,
                "answerability": answerability,
            }

        context = self._format_context(chunks)
        chain = self._templates.get_rag_prompt() | self._llm | self._str_parser

        logger.info("Generating answer for query: %.80s", query)
        raw_answer = await chain.ainvoke({"query": query, "context": context})
        logger.info("Raw answer generated (%d chars).", len(raw_answer))

        # Post-processing faithfulness filter — strip non-grounded claims.
        answer = await self._strip_ungrounded_claims(raw_answer, context)

        return {
            "answer": answer,
            "context_used": chunks,
            "answerability": answerability,
            "faithfulness_filtered": True,
        }

    async def extract_citations(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract structured citations linking answer claims to source chunks.

        LCEL chain: ``citation_prompt | llm | JsonOutputParser``

        If the LLM returns malformed JSON or the parser raises, the method
        falls back to returning a basic citation for every supplied chunk so
        the caller always receives a usable (if less precise) list.

        Args:
            answer: The LLM-generated answer from ``generate_answer``.
            chunks: The chunks that were available when the answer was produced.

        Returns:
            List of citation dicts, each containing ``chunk_id``,
            ``filename``, ``page_number``, and ``quote``.
        """
        context = self._format_context(chunks)
        chain = (
            self._templates.get_citation_extraction_prompt()
            | self._llm
            | self._json_parser
        )

        try:
            result: Any = await chain.ainvoke(
                {"answer": answer, "context": context}
            )
            citations: list[dict[str, Any]] = result.get("citations", [])
            logger.info("Extracted %d citation(s).", len(citations))
            return citations
        except Exception as exc:
            logger.warning(
                "Citation extraction failed (%s: %s) — falling back to chunk list.",
                type(exc).__name__,
                exc,
            )
            return [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "filename": c.get("filename", ""),
                    "page_number": c.get("page_number", 0),
                    "quote": c.get("text", "")[:200],
                }
                for c in chunks
            ]

    async def check_hallucination(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        original_answer: str,
    ) -> dict[str, Any]:
        """Detect potential hallucinations via self-consistency and keyword checks.

        Runs three complementary checks:

        1. **Keyword contradiction** — detects polarity mismatches (e.g. query
           asks "minimum" but answer discusses "maximum") without an LLM call.
           Returns immediately with ``confidence_score=0.1`` if triggered.

        2. **Self-consistency** — generates a second independent answer with a
           rephrased prompt and compares key entities (numbers, dates, article
           references).

        3. **Question-specificity** — the consistency prompt is instructed to
           flag when the answer addresses a related but different question.

        LCEL chain: ``consistency_check_prompt | llm | StrOutputParser``

        Args:
            query:           The original user question.
            chunks:          The same chunks used to produce ``original_answer``.
            original_answer: The answer produced by ``generate_answer``.

        Returns:
            Dict with keys:

            - ``is_consistent``   – ``True`` if no issues detected.
            - ``confidence_score``– Float in [0.0, 1.0].
            - ``warning``         – Human-readable warning, or ``None``.
        """
        # Check 1 — fast keyword contradiction (no LLM call needed).
        contradiction = self._detect_keyword_contradiction(query, original_answer)
        if contradiction:
            warning = (
                f"Answer may address a different aspect than what was asked. "
                f"Question asked about '{contradiction['query_keyword']}' but "
                f"answer discusses '{contradiction['answer_keyword']}'."
            )
            logger.warning("Keyword contradiction detected: %s", warning)
            return {
                "is_consistent": False,
                "confidence_score": 0.1,
                "warning": warning,
            }

        # Check 2 — self-consistency via second LLM generation.
        context = self._format_context(chunks)
        chain = (
            self._templates.get_consistency_check_prompt()
            | self._llm
            | self._str_parser
        )

        logger.info("Running consistency check for query: %.80s", query)
        second_answer: str = await chain.ainvoke(
            {"query": query, "context": context}
        )

        original_entities = self._extract_entities(original_answer)
        second_entities = self._extract_entities(second_answer)
        all_entities = original_entities | second_entities

        if not all_entities:
            return {
                "is_consistent": True,
                "confidence_score": 0.7,
                "warning": None,
            }

        overlap = original_entities & second_entities
        confidence_score = round(len(overlap) / len(all_entities), 3)
        is_consistent = confidence_score >= 0.8

        warning: str | None = None
        if not is_consistent:
            missing = all_entities - overlap
            warning = (
                f"Consistency check flagged potential hallucination. "
                f"Diverging entities: {', '.join(sorted(missing))}."
            )
            logger.warning(warning)
        else:
            logger.info(
                "Consistency check passed (score=%.3f, entities matched=%d/%d).",
                confidence_score,
                len(overlap),
                len(all_entities),
            )

        return {
            "is_consistent": is_consistent,
            "confidence_score": confidence_score,
            "warning": warning,
        }
