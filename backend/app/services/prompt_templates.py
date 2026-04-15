"""LangChain prompt templates for RAG, citation extraction, and consistency checks."""

from langchain_core.prompts import ChatPromptTemplate

_RAG_SYSTEM = """\
You are a strict legal document analyst with ONE rule: \
ONLY use information explicitly stated in the provided context chunks. Nothing else.

FAITHFULNESS RULES — NON NEGOTIABLE:
1. Every single sentence in your answer MUST be directly traceable to a specific \
context chunk provided.

2. NEVER add information from general knowledge, even if you are certain it is correct. \
If it is not in the context, it does not exist for this answer.

3. NEVER use phrases like:
   - 'Generally speaking...'
   - 'It is commonly understood...'
   - 'In addition to what is stated...'
   - 'This typically means...'
   These indicate you are adding outside knowledge.

4. If a question requires information spread across multiple chunks, only combine what is \
EXPLICITLY in those chunks — do not fill gaps with assumptions.

5. After writing each sentence, mentally ask: 'Which exact chunk does this come from?' \
If you cannot answer that — DELETE the sentence.

6. Structure your answer as:
   - Direct answer (from chunk X)
   - Supporting detail (from chunk Y)
   - Additional detail if present (from chunk Z)
   - If information is incomplete: explicitly state 'The provided context does not specify [X]'

7. Keep answers concise — do not pad with context you are unsure about."""

_CONSISTENCY_SYSTEM = """You are a precise legal and compliance document analyst.
Your job is to verify and answer questions based ONLY on the provided document context.

Rules:
- Only use information from the provided context
- Carefully verify each claim against the source chunks before including it
- Every claim must reference a specific source chunk
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Be precise and concise
- Always cite the document name and page number

Additionally verify: does the answer actually address what was asked, or did it answer a related but different question? If it answered a different question, flag as inconsistent."""

_CITATION_SYSTEM = "Extract citations from the answer. Return JSON only."

_ANSWERABILITY_SYSTEM = "You are a strict fact-checker."

_RAG_HUMAN = """\
{context}

Using ONLY the chunks provided above (no outside knowledge), answer this question: {query}

Remember: Every claim must come from a specific numbered chunk above."""

_CONSISTENCY_HUMAN = """\
Context:
{context}

Question: {query}

Verify the context carefully, then answer (cite document name and page number for every claim):"""

_CITATION_HUMAN = """\
Answer: {answer}
Context chunks: {context}

Return JSON: {{"citations": [{{"chunk_id": "...", "filename": "...", "page_number": 0, "quote": "relevant excerpt from chunk"}}]}}"""

_ANSWERABILITY_HUMAN = """\
Question: {query}

Context chunks:
{context}

Answer ONLY with valid JSON, no markdown:
{{
  "is_answerable": true or false,
  "reason": "why it is or is not answerable",
  "what_document_actually_says": "what related info exists in the chunks, if any"
}}

Rules:
- is_answerable = true ONLY if the context contains a DIRECT answer to the EXACT question asked
- If the question asks about X but context only has Y (even if related), is_answerable = false
- Be extremely literal about what is/is not in the text"""


class PromptTemplates:
    """Factory for LangChain ``ChatPromptTemplate`` objects used in the RAG pipeline.

    All methods are static — the class is a namespace, not a stateful object.
    Templates are constructed on each call rather than cached at class level
    so they are always fresh (no shared mutable state between requests).
    """

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Strip control characters that make JSON bodies invalid.

        PDF extraction can embed null bytes and other C0 control characters.
        OpenAI returns HTTP 400 "could not parse JSON body" when these appear
        in the serialised request payload.

        Args:
            text: Raw text that may contain control characters.

        Returns:
            Text with all characters below U+0020 removed except standard
            whitespace (space, tab, newline, carriage-return).
        """
        return "".join(
            ch for ch in text
            if ch in (" ", "\t", "\n", "\r") or (ord(ch) > 0x1F)
        )

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """Render a list of retrieved chunks into a single context string.

        Each chunk is wrapped with explicit ``[CHUNK N]`` / ``[END CHUNK N]``
        markers so the LLM can attribute every claim to a specific numbered
        chunk.  Chunk text is sanitized to remove control characters.

        Args:
            chunks: Retrieved chunk dicts; each must contain ``filename``,
                    ``page_number``, ``chunk_index``, and ``text`` keys.

        Returns:
            A multi-line string ready to be substituted into ``{context}``
            in any of the prompt templates.
        """
        sanitize = PromptTemplates._sanitize_text
        parts: list[str] = []
        for chunk in chunks:
            idx = chunk.get("chunk_index", "?")
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page_number", "?")
            text = sanitize(chunk.get("text", ""))
            parts.append(
                f"[CHUNK {idx} | {filename} | Page {page}]\n"
                f"{text}\n"
                f"[END CHUNK {idx}]\n"
                "---"
            )
        return "\n\n".join(parts)

    @staticmethod
    def get_rag_prompt() -> ChatPromptTemplate:
        """Return the primary RAG answer-generation prompt.

        The template expects two input variables:

        - ``context`` – pre-formatted source chunks, typically produced by
          ``format_context``.
        - ``query``   – the user's natural-language question.

        The system message enforces strict grounding: the model must not
        infer, extrapolate, or reframe information, and must explicitly
        acknowledge when the exact information is absent rather than
        substituting related but different facts.

        Returns:
            A ``ChatPromptTemplate`` ready for use in an LCEL chain::

                chain = get_rag_prompt() | llm | StrOutputParser()
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM),
                ("human", _RAG_HUMAN),
            ]
        )

    @staticmethod
    def get_citation_extraction_prompt() -> ChatPromptTemplate:
        """Return a prompt that extracts structured citations from an LLM answer.

        The template expects two input variables:

        - ``answer``  – the LLM-generated answer text.
        - ``context`` – the source chunks the answer was generated from.

        The model is instructed to return **only** a JSON object so the
        output can be parsed directly with ``json.loads`` without stripping
        markdown fences.

        Returns:
            A ``ChatPromptTemplate`` whose output should be parsed as JSON
            with the schema::

                {
                  "citations": [
                    {
                      "chunk_id": str,
                      "filename": str,
                      "page_number": int,
                      "quote": str
                    }
                  ]
                }
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", _CITATION_SYSTEM),
                ("human", _CITATION_HUMAN),
            ]
        )

    @staticmethod
    def get_consistency_check_prompt() -> ChatPromptTemplate:
        """Return a rephrased RAG prompt used for self-consistency checking.

        In addition to the standard grounding rules, the system message
        instructs the model to verify whether the answer addresses the exact
        question asked — catching cases where a model answered a related but
        distinct question (e.g. maximum vs minimum fine).

        The template expects the same input variables as ``get_rag_prompt``:

        - ``context`` – pre-formatted source chunks.
        - ``query``   – the user's natural-language question.

        Returns:
            A ``ChatPromptTemplate`` with the same interface as
            ``get_rag_prompt`` but distinct wording to reduce prompt-level
            correlation between the two answers.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", _CONSISTENCY_SYSTEM),
                ("human", _CONSISTENCY_HUMAN),
            ]
        )

    @staticmethod
    def get_answerability_prompt() -> ChatPromptTemplate:
        """Return a prompt that checks whether the context can directly answer a query.

        This pre-generation gate asks the model to judge — before any answer
        is produced — whether the retrieved chunks contain a *direct* answer
        to the *exact* question asked.  Returning ``is_answerable: false``
        short-circuits the RAG chain and returns a structured "cannot find"
        response instead of risking a hallucinated answer.

        The template expects two input variables:

        - ``query``   – the user's natural-language question.
        - ``context`` – pre-formatted source chunks.

        Returns:
            A ``ChatPromptTemplate`` whose output must be parsed as JSON::

                {
                  "is_answerable": bool,
                  "reason": str,
                  "what_document_actually_says": str
                }
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", _ANSWERABILITY_SYSTEM),
                ("human", _ANSWERABILITY_HUMAN),
            ]
        )
