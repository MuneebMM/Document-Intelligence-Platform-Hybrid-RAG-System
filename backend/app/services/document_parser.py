"""Document parsing service supporting PDF, DOCX, and plain-text formats."""

import io
from typing import Literal

import fitz  # PyMuPDF
from docx import Document


# Minimum character count for a PDF page to be included in output.
_PDF_MIN_CHARS = 50

# Target character budget per synthetic "page" for DOCX and TXT sources.
_PAGE_SIZE = 3000

PageDict = dict[str, int | str]


class DocumentParser:
    """Parse raw file bytes into page-level text chunks.

    All methods are async so they compose naturally with FastAPI route
    handlers and other async services without blocking the event loop.
    CPU-bound work (PyMuPDF, python-docx) is lightweight enough for the
    document sizes typical in legal/compliance workflows; if very large
    files become a bottleneck, offload to a thread pool via
    ``asyncio.to_thread``.
    """

    async def parse_pdf(self, file_bytes: bytes, filename: str) -> list[PageDict]:
        """Extract text from a PDF, one dict per page.

        Uses PyMuPDF (fitz) for fast, accurate text extraction.  Pages
        whose extracted text is shorter than ``_PDF_MIN_CHARS`` characters
        (e.g. scanned images, blank separators) are silently skipped.

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename:   Original filename; stored in every returned dict.

        Returns:
            List of dicts with keys ``page_number`` (1-based), ``text``,
            and ``filename``.  May be empty if the PDF has no extractable
            text.
        """
        pages: list[PageDict] = []

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text().strip()
                if len(text) < _PDF_MIN_CHARS:
                    continue
                pages.append(
                    {
                        "page_number": page.number + 1,  # fitz is 0-based
                        "text": text,
                        "filename": filename,
                    }
                )

        return pages

    async def parse_docx(self, file_bytes: bytes, filename: str) -> list[PageDict]:
        """Extract text from a DOCX file, grouped into synthetic pages.

        python-docx exposes paragraphs, not pages, so paragraphs are
        accumulated until the running total reaches ``_PAGE_SIZE``
        characters, at which point a new page is started.  This keeps
        downstream chunk sizes predictable regardless of how the author
        structured the document.

        Args:
            file_bytes: Raw bytes of the DOCX file.
            filename:   Original filename; stored in every returned dict.

        Returns:
            List of dicts with keys ``page_number`` (1-based), ``text``,
            and ``filename``.
        """
        doc = Document(io.BytesIO(file_bytes))

        pages: list[PageDict] = []
        current_parts: list[str] = []
        current_len: int = 0
        page_number: int = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            current_parts.append(text)
            current_len += len(text)

            if current_len >= _PAGE_SIZE:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": "\n".join(current_parts),
                        "filename": filename,
                    }
                )
                page_number += 1
                current_parts = []
                current_len = 0

        # Flush any remaining paragraphs that didn't fill a full page.
        if current_parts:
            pages.append(
                {
                    "page_number": page_number,
                    "text": "\n".join(current_parts),
                    "filename": filename,
                }
            )

        return pages

    async def parse_txt(self, file_bytes: bytes, filename: str) -> list[PageDict]:
        """Extract text from a plain-text file, split into fixed-size chunks.

        The bytes are decoded as UTF-8 (with replacement for any invalid
        sequences), then sliced into chunks of ``_PAGE_SIZE`` characters.
        Each chunk becomes one "page".

        Args:
            file_bytes: Raw bytes of the text file.
            filename:   Original filename; stored in every returned dict.

        Returns:
            List of dicts with keys ``page_number`` (1-based), ``text``,
            and ``filename``.
        """
        full_text = file_bytes.decode("utf-8", errors="replace").strip()

        pages: list[PageDict] = []
        for i, offset in enumerate(range(0, len(full_text), _PAGE_SIZE)):
            chunk = full_text[offset : offset + _PAGE_SIZE]
            pages.append(
                {
                    "page_number": i + 1,
                    "text": chunk,
                    "filename": filename,
                }
            )

        return pages

    async def parse_document(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: Literal[".pdf", ".docx", ".txt"],
    ) -> list[PageDict]:
        """Route file bytes to the appropriate parser based on file extension.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename; forwarded to the specific parser.
            file_type:  Lowercase file extension including the leading dot.
                        Accepted values: ``".pdf"``, ``".docx"``, ``".txt"``.

        Returns:
            List of page dicts as produced by the underlying parser.

        Raises:
            ValueError: If ``file_type`` is not one of the supported values.
        """
        parsers = {
            ".pdf": self.parse_pdf,
            ".docx": self.parse_docx,
            ".txt": self.parse_txt,
        }

        parser = parsers.get(file_type)
        if parser is None:
            supported = ", ".join(parsers)
            raise ValueError(
                f"Unsupported file type '{file_type}'. Supported: {supported}"
            )

        return await parser(file_bytes, filename)
