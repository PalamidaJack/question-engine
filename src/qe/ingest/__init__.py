"""Voice & multimodal ingestion layer for Question Engine."""

from __future__ import annotations

from qe.ingest.documents import DocumentIngestor
from qe.ingest.ocr import OCRProcessor
from qe.ingest.voice import VoiceIngestor

__all__ = ["VoiceIngestor", "DocumentIngestor", "OCRProcessor"]
