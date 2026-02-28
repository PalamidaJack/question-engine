"""Document ingestion: extract text chunks from PDF, DOCX, CSV, and more."""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Chunk-splitting defaults ────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE = 1000
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class DocumentChunk(BaseModel):
    """A chunk of text extracted from a document."""

    text: str
    page_number: int | None = None
    section: str = ""
    source_file: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestor:
    """Ingest documents of various formats into text chunks.

    Supported formats are registered in ``HANDLERS``.  Heavy third-party
    dependencies (pymupdf, python-docx) are imported lazily and guarded
    with try/except ImportError so the class remains importable without them.
    """

    HANDLERS: dict[str, str] = {
        ".pdf": "_ingest_pdf",
        ".docx": "_ingest_docx",
        ".csv": "_ingest_csv",
        ".txt": "_ingest_text",
        ".text": "_ingest_text",
        ".log": "_ingest_text",
        ".md": "_ingest_markdown",
        ".markdown": "_ingest_markdown",
        ".html": "_ingest_html",
        ".htm": "_ingest_html",
    }

    # ── Public API ───────────────────────────────────────────────────────

    async def ingest(self, file_path: Path | str) -> list[DocumentChunk]:
        """Ingest a document and return a list of text chunks."""
        path = Path(file_path)
        loop = asyncio.get_running_loop()
        exists = await loop.run_in_executor(None, os.path.exists, path)
        if not exists:
            raise FileNotFoundError(f"Document not found: {path}")

        ext = path.suffix.lower()
        handler_name = self.HANDLERS.get(ext)
        if handler_name is None:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(sorted(self.HANDLERS))}"
            )

        handler = getattr(self, handler_name)
        chunks = await handler(path)
        log.info(
            "Ingested %d chunks from %s (%s)", len(chunks), path.name, ext
        )
        return chunks

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return list of supported file extensions."""
        return sorted(cls.HANDLERS.keys())

    # ── Format handlers ──────────────────────────────────────────────────

    async def _ingest_pdf(self, path: Path) -> list[DocumentChunk]:
        """Extract text from a PDF using pymupdf (fitz)."""
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF ingestion. "
                "Install with: pip install pymupdf"
            ) from None

        def _extract() -> list[DocumentChunk]:
            chunks: list[DocumentChunk] = []
            chunk_idx = 0

            with fitz.open(str(path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if not text:
                        continue

                    page_chunks = _split_text(text)
                    for chunk_text in page_chunks:
                        chunks.append(
                            DocumentChunk(
                                text=chunk_text,
                                page_number=page_num + 1,
                                section=f"page_{page_num + 1}",
                                source_file=str(path),
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1

            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)

    async def _ingest_docx(self, path: Path) -> list[DocumentChunk]:
        """Extract text from a DOCX file using python-docx."""
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX ingestion. "
                "Install with: pip install python-docx"
            ) from None

        def _extract() -> list[DocumentChunk]:
            document = docx.Document(str(path))
            chunks: list[DocumentChunk] = []
            chunk_idx = 0

            current_section = ""
            section_text_parts: list[str] = []

            for para in document.paragraphs:
                # Detect headings as section boundaries
                if para.style and para.style.name and para.style.name.startswith("Heading"):
                    # Flush current section
                    if section_text_parts:
                        full_text = "\n".join(section_text_parts).strip()
                        if full_text:
                            for chunk_text in _split_text(full_text):
                                chunks.append(
                                    DocumentChunk(
                                        text=chunk_text,
                                        section=current_section,
                                        source_file=str(path),
                                        chunk_index=chunk_idx,
                                    )
                                )
                                chunk_idx += 1
                        section_text_parts = []

                    current_section = para.text.strip()
                else:
                    text = para.text.strip()
                    if text:
                        section_text_parts.append(text)

            # Flush remaining section
            if section_text_parts:
                full_text = "\n".join(section_text_parts).strip()
                if full_text:
                    for chunk_text in _split_text(full_text):
                        chunks.append(
                            DocumentChunk(
                                text=chunk_text,
                                section=current_section,
                                source_file=str(path),
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1

            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)

    async def _ingest_csv(self, path: Path) -> list[DocumentChunk]:
        """Ingest a CSV file, converting rows into text chunks."""

        def _extract() -> list[DocumentChunk]:
            chunks: list[DocumentChunk] = []

            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    # Represent each row as "key: value" pairs
                    parts = [
                        f"{col}: {val}" for col, val in row.items() if val
                    ]
                    text = "; ".join(parts)
                    if text:
                        chunks.append(
                            DocumentChunk(
                                text=text,
                                section="data",
                                source_file=str(path),
                                chunk_index=row_idx,
                                metadata={"row_number": row_idx + 1},
                            )
                        )
            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)

    async def _ingest_text(self, path: Path) -> list[DocumentChunk]:
        """Ingest a plain text file with chunk splitting."""

        def _extract() -> list[DocumentChunk]:
            text = path.read_text(encoding="utf-8")
            chunks: list[DocumentChunk] = []

            for idx, chunk_text in enumerate(_split_text(text)):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        section="body",
                        source_file=str(path),
                        chunk_index=idx,
                    )
                )
            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)

    async def _ingest_markdown(self, path: Path) -> list[DocumentChunk]:
        """Ingest a Markdown file, splitting by headings."""

        def _extract() -> list[DocumentChunk]:
            text = path.read_text(encoding="utf-8")
            chunks: list[DocumentChunk] = []
            chunk_idx = 0

            # Split on Markdown headings (# ... or ## ... etc.)
            heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
            positions: list[tuple[int, str]] = []

            for match in heading_re.finditer(text):
                positions.append((match.start(), match.group(2).strip()))

            if not positions:
                # No headings: treat as plain text
                for chunk_text in _split_text(text):
                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            section="body",
                            source_file=str(path),
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                return chunks

            # Handle text before the first heading
            if positions[0][0] > 0:
                preamble = text[: positions[0][0]].strip()
                if preamble:
                    for chunk_text in _split_text(preamble):
                        chunks.append(
                            DocumentChunk(
                                text=chunk_text,
                                section="preamble",
                                source_file=str(path),
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1

            # Process each section
            for i, (start_pos, heading) in enumerate(positions):
                end_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
                section_text = text[start_pos:end_pos].strip()

                # Remove the heading line itself from the body
                first_newline = section_text.find("\n")
                if first_newline > 0:
                    body = section_text[first_newline:].strip()
                else:
                    body = ""

                if body:
                    for chunk_text in _split_text(body):
                        chunks.append(
                            DocumentChunk(
                                text=chunk_text,
                                section=heading,
                                source_file=str(path),
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1

            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)

    async def _ingest_html(self, path: Path) -> list[DocumentChunk]:
        """Ingest an HTML file by stripping tags and extracting text."""

        def _extract() -> list[DocumentChunk]:
            raw = path.read_text(encoding="utf-8")

            # Strip HTML tags using a simple regex approach
            # Remove script and style elements entirely
            cleaned = re.sub(
                r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove all remaining HTML tags
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            # Collapse whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            # Decode common HTML entities
            cleaned = (
                cleaned.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
                .replace("&nbsp;", " ")
            )

            chunks: list[DocumentChunk] = []
            for idx, chunk_text in enumerate(_split_text(cleaned)):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        section="body",
                        source_file=str(path),
                        chunk_index=idx,
                    )
                )
            return chunks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _extract)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _split_text(
    text: str, max_chars: int = _DEFAULT_CHUNK_SIZE
) -> list[str]:
    """Split text into chunks of roughly *max_chars* at sentence boundaries.

    Tries to break at sentence-ending punctuation (.!?) followed by
    whitespace.  If a sentence itself exceeds *max_chars* it is included
    as-is to avoid losing content.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Would adding this sentence exceed the limit?
        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks
