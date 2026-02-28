"""OCR processing: extract text from images and scanned PDF pages."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)


class OCRResult(BaseModel):
    """Result of an OCR extraction."""

    text: str
    confidence: float
    language: str
    source_file: str
    page_number: int | None = None


class OCRProcessor:
    """Extract text from images using Tesseract OCR.

    Both ``pytesseract`` and ``Pillow`` are optional dependencies.
    Use :meth:`is_available` to check at runtime whether the
    underlying ``tesseract`` binary is reachable.
    """

    def __init__(self, language: str = "eng") -> None:
        self._language = language

    # ── Public API ───────────────────────────────────────────────────────

    async def extract(self, image_path: Path | str) -> OCRResult:
        """Run OCR on a single image file."""
        path = Path(image_path)
        loop = asyncio.get_running_loop()
        exists = await loop.run_in_executor(None, os.path.exists, path)
        if not exists:
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pytesseract and Pillow are required for OCR. "
                "Install with: pip install pytesseract Pillow"
            ) from None

        def _run() -> OCRResult:
            img = Image.open(str(path))
            # Get detailed data for confidence calculation
            data = pytesseract.image_to_data(
                img, lang=self._language, output_type=pytesseract.Output.DICT
            )

            # Compute average confidence across words with valid conf values
            confidences = [
                float(c) for c in data.get("conf", []) if int(c) >= 0
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            # Extract full text
            text = pytesseract.image_to_string(
                img, lang=self._language
            ).strip()

            return OCRResult(
                text=text,
                confidence=avg_confidence,
                language=self._language,
                source_file=str(path),
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run)
        log.info(
            "OCR extracted %d chars from %s (confidence=%.2f)",
            len(result.text),
            path.name,
            result.confidence,
        )
        return result

    async def extract_from_pdf_page(
        self, pdf_path: Path | str, page_num: int
    ) -> OCRResult:
        """Render a single PDF page to an image and run OCR on it."""
        path = Path(pdf_path)
        loop = asyncio.get_running_loop()
        exists = await loop.run_in_executor(None, os.path.exists, path)
        if not exists:
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required to render PDF pages for OCR. "
                "Install with: pip install pymupdf"
            ) from None

        try:
            import PIL  # noqa: F401
        except ImportError:
            raise ImportError(
                "Pillow is required for OCR. "
                "Install with: pip install Pillow"
            ) from None

        def _render() -> Path:
            doc = fitz.open(str(path))
            if page_num < 0 or page_num >= len(doc):
                doc.close()
                raise ValueError(
                    f"Page {page_num} out of range (document has {len(doc)} pages)"
                )

            page = doc[page_num]
            # Render at 300 DPI for good OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)

            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix="qe_ocr_"
            )
            pix.save(tmp.name)
            doc.close()
            return Path(tmp.name)

        loop = asyncio.get_running_loop()
        rendered_path = await loop.run_in_executor(None, _render)

        try:
            result = await self.extract(rendered_path)
            # Override source and page info
            result = result.model_copy(
                update={
                    "source_file": str(path),
                    "page_number": page_num + 1,
                }
            )
            return result
        finally:
            if rendered_path.exists():
                rendered_path.unlink(missing_ok=True)

    @staticmethod
    def is_available() -> bool:
        """Check whether tesseract is installed and reachable."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except ImportError:
            return False
        except Exception:
            return False
