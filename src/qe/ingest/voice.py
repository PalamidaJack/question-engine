"""Voice ingestion: transcription, diarization, and structuring."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class VoiceObservation(BaseModel):
    """A structured observation extracted from a voice recording."""

    text: str
    speaker: str
    timestamp_start: float
    timestamp_end: float
    confidence: float
    source_file: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TranscriptSegment:
    """A single segment of a transcript."""

    text: str
    start: float
    end: float
    confidence: float
    speaker: str = ""


class VoiceIngestor:
    """Transcribe and diarize audio files into structured observations.

    Heavy dependencies (faster-whisper, pyannote.audio) are optional.
    Falls back to subprocess-based whisper CLI or no diarization when
    the libraries are not installed.
    """

    def __init__(self, model_size: str = "base", device: str = "auto") -> None:
        self._model_size = model_size
        self._device = device
        self._whisper_model: Any = None

    # ── Public API ───────────────────────────────────────────────────────

    async def ingest(
        self,
        audio_path: Path | str,
        speaker_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[VoiceObservation]:
        """Full pipeline: preprocess, transcribe, diarize, structure."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        meta = metadata or {}
        log.info("Starting voice ingestion for %s", audio_path.name)

        preprocessed = await self._preprocess(audio_path)
        try:
            segments = await self._transcribe(preprocessed)
            segments = await self._diarize(preprocessed, segments)
            observations = await self._structure(
                segments, speaker_names, str(audio_path), meta
            )
        finally:
            # Clean up temp file only if we created one
            if preprocessed != audio_path and preprocessed.exists():
                preprocessed.unlink(missing_ok=True)

        log.info(
            "Voice ingestion complete: %d observations from %s",
            len(observations),
            audio_path.name,
        )
        return observations

    # ── Preprocessing ────────────────────────────────────────────────────

    async def _preprocess(self, audio_path: Path) -> Path:
        """Normalize audio to 16 kHz mono WAV using ffmpeg."""
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                log.warning(
                    "ffmpeg preprocessing failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:500],
                )
                # Fall back to original file
                return audio_path

            log.debug("Preprocessed audio to %s", output_path)
            return output_path

        except FileNotFoundError:
            log.warning(
                "ffmpeg not found; skipping audio preprocessing. "
                "Install ffmpeg for best results."
            )
            return audio_path

    # ── Transcription ────────────────────────────────────────────────────

    async def _transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe audio using faster-whisper or whisper CLI fallback."""
        # Try faster-whisper first
        segments = await self._transcribe_faster_whisper(audio_path)
        if segments is not None:
            return segments

        # Fall back to whisper CLI
        segments = await self._transcribe_whisper_cli(audio_path)
        if segments is not None:
            return segments

        raise RuntimeError(
            "No transcription backend available. Install faster-whisper "
            "(pip install faster-whisper) or the whisper CLI "
            "(pip install openai-whisper)."
        )

    async def _transcribe_faster_whisper(
        self, audio_path: Path
    ) -> list[TranscriptSegment] | None:
        """Attempt transcription with faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.debug("faster-whisper not installed; trying fallback.")
            return None

        def _run() -> list[TranscriptSegment]:
            if self._whisper_model is None:
                device = self._device
                if device == "auto":
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                self._whisper_model = WhisperModel(
                    self._model_size, device=device, compute_type="int8"
                )

            raw_segments, _info = self._whisper_model.transcribe(
                str(audio_path), beam_size=5
            )

            results: list[TranscriptSegment] = []
            for seg in raw_segments:
                results.append(
                    TranscriptSegment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        confidence=seg.avg_logprob,
                    )
                )
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)

    async def _transcribe_whisper_cli(
        self, audio_path: Path
    ) -> list[TranscriptSegment] | None:
        """Fall back to the whisper CLI (openai-whisper package)."""
        import json as json_mod

        output_dir = Path(tempfile.mkdtemp(prefix="qe_whisper_"))
        cmd = [
            "whisper",
            str(audio_path),
            "--model", self._model_size,
            "--output_format", "json",
            "--output_dir", str(output_dir),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                log.warning(
                    "whisper CLI failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:500],
                )
                return None

            # Find the JSON output file
            json_files = [
                p for p in os.listdir(output_dir) if p.endswith(".json")
            ]
            if not json_files:
                log.warning("whisper CLI produced no JSON output")
                return None

            json_path = output_dir / json_files[0]
            data = json_mod.loads(json_path.read_text(encoding="utf-8"))
            segments: list[TranscriptSegment] = []
            for seg in data.get("segments", []):
                segments.append(
                    TranscriptSegment(
                        text=seg.get("text", "").strip(),
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        confidence=float(seg.get("avg_logprob", 0.0)),
                    )
                )
            return segments

        except FileNotFoundError:
            log.debug("whisper CLI not found on PATH.")
            return None
        finally:
            # Cleanup temp output dir
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    # ── Diarization ──────────────────────────────────────────────────────

    async def _diarize(
        self, audio_path: Path, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Assign speaker labels via pyannote.audio if available."""
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            log.debug(
                "pyannote.audio not installed; skipping diarization. "
                "Install with: pip install pyannote.audio"
            )
            # Assign a default speaker label
            for seg in segments:
                if not seg.speaker:
                    seg.speaker = "speaker_0"
            return segments

        def _run() -> list[TranscriptSegment]:
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            diarization = pipeline(str(audio_path))

            # Build a timeline of speaker turns
            speaker_turns: list[tuple[float, float, str]] = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                speaker_turns.append((turn.start, turn.end, speaker_label))

            # Assign speakers to transcript segments by overlap
            for seg in segments:
                best_speaker = "speaker_0"
                best_overlap = 0.0

                for turn_start, turn_end, spk in speaker_turns:
                    overlap_start = max(seg.start, turn_start)
                    overlap_end = min(seg.end, turn_end)
                    overlap = max(0.0, overlap_end - overlap_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = spk

                seg.speaker = best_speaker

            return segments

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)

    # ── Structuring ──────────────────────────────────────────────────────

    async def _structure(
        self,
        segments: list[TranscriptSegment],
        speaker_names: list[str] | None,
        source_file: str,
        metadata: dict[str, Any],
    ) -> list[VoiceObservation]:
        """Convert transcript segments into VoiceObservation models."""
        # Build a mapping from raw speaker labels to friendly names
        speaker_map: dict[str, str] = {}
        if speaker_names:
            # Collect unique raw labels in order of first appearance
            seen_labels: list[str] = []
            for seg in segments:
                if seg.speaker not in seen_labels:
                    seen_labels.append(seg.speaker)

            for idx, label in enumerate(seen_labels):
                if idx < len(speaker_names):
                    speaker_map[label] = speaker_names[idx]
                else:
                    speaker_map[label] = label

        observations: list[VoiceObservation] = []
        for seg in segments:
            speaker = speaker_map.get(seg.speaker, seg.speaker)
            observations.append(
                VoiceObservation(
                    text=seg.text,
                    speaker=speaker,
                    timestamp_start=seg.start,
                    timestamp_end=seg.end,
                    confidence=seg.confidence,
                    source_file=source_file,
                    metadata=metadata,
                )
            )

        return observations
