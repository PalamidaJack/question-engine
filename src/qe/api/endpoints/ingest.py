"""Ingestion API endpoints: document, voice, and image upload."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


async def _save_upload(upload: UploadFile, suffix: str = "") -> Path:
    """Persist an uploaded file to a temporary path and return it."""
    if not suffix and upload.filename:
        suffix = Path(upload.filename).suffix
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="qe_upload_"
    )
    contents = await upload.read()
    tmp.write(contents)
    tmp.close()
    return Path(tmp.name)


# ── Document ─────────────────────────────────────────────────────────────────


_file_upload = File(...)


@router.post("/document")
async def ingest_document(file: UploadFile = _file_upload):
    """Upload a document file and return extracted text chunks.

    Supported formats: PDF, DOCX, CSV, TXT, Markdown, HTML.
    """
    from qe.ingest.documents import DocumentIngestor

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in DocumentIngestor.HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(DocumentIngestor.supported_formats())}"
            ),
        )

    tmp_path = await _save_upload(file, suffix=ext)
    try:
        ingestor = DocumentIngestor()
        chunks = await ingestor.ingest(tmp_path)
        return {
            "filename": file.filename,
            "format": ext,
            "chunk_count": len(chunks),
            "chunks": [c.model_dump(mode="json") for c in chunks],
        }
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Document ingestion failed for %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Voice ────────────────────────────────────────────────────────────────────


@router.post("/voice")
async def ingest_voice(file: UploadFile = _file_upload):
    """Upload an audio file and return voice observations (transcription).

    Requires ffmpeg and a whisper backend (faster-whisper or openai-whisper).
    """
    from qe.ingest.voice import VoiceIngestor

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    tmp_path = await _save_upload(file, suffix=ext)
    try:
        ingestor = VoiceIngestor()
        observations = await ingestor.ingest(tmp_path)
        return {
            "filename": file.filename,
            "observation_count": len(observations),
            "observations": [o.model_dump(mode="json") for o in observations],
        }
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Voice ingestion failed for %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Image / OCR ──────────────────────────────────────────────────────────────


@router.post("/image")
async def ingest_image(file: UploadFile = _file_upload):
    """Upload an image and return OCR-extracted text.

    Requires tesseract and pytesseract + Pillow.
    """
    from qe.ingest.ocr import OCRProcessor

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: {ext}. "
                f"Supported: {', '.join(sorted(allowed_exts))}"
            ),
        )

    tmp_path = await _save_upload(file, suffix=ext)
    try:
        processor = OCRProcessor()
        if not processor.is_available():
            raise HTTPException(
                status_code=501,
                detail=(
                    "Tesseract OCR is not available. "
                    "Install tesseract and pytesseract + Pillow."
                ),
            )
        result = await processor.extract(tmp_path)
        return {
            "filename": file.filename,
            "result": result.model_dump(mode="json"),
        }
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Image OCR failed for %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"OCR failed: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
