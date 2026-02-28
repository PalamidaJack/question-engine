"""Vector store backed by SQLite for semantic search and contradiction detection.

Uses litellm for embeddings by default (supports all providers).
Optional upgrade to sentence-transformers for free local embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import struct
from datetime import UTC, datetime
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pack_floats(floats: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob."""
    return struct.pack(f"{len(floats)}f", *floats)


def _unpack_floats(data: bytes) -> list[float]:
    """Unpack a binary blob into a list of floats."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class SearchResult:
    """A single search result with similarity score."""

    __slots__ = ("id", "text", "similarity", "metadata")

    def __init__(
        self,
        id: str,
        text: str,
        similarity: float,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.text = text
        self.similarity = similarity
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.id!r}, sim={self.similarity:.3f}, "
            f"text={self.text[:60]!r})"
        )


class Contradiction:
    """A potential contradiction between a query and an existing embedding."""

    __slots__ = ("id", "text", "similarity", "metadata")

    def __init__(
        self,
        id: str,
        text: str,
        similarity: float,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.text = text
        self.similarity = similarity
        self.metadata = metadata


class EmbeddingStore:
    """Vector store backed by SQLite for zero-dependency operation.

    Embedding generation uses litellm (supports OpenAI, Anthropic, Cohere,
    local models via Ollama, etc.). Falls back to sentence-transformers
    if available and configured.
    """

    def __init__(
        self,
        db_path: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._db_path = db_path
        self._model = model
        self._local_model: Any = None

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding via configured model."""
        # Try sentence-transformers first if model looks local
        if self._model.startswith("local/"):
            return self._embed_local(text)

        # Use litellm for API-based embeddings
        import litellm

        response = await litellm.aembedding(
            model=self._model,
            input=[text],
        )
        return response.data[0]["embedding"]

    def _embed_local(self, text: str) -> list[float]:
        """Use sentence-transformers for free local embeddings."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self._model.removeprefix("local/")
                self._local_model = SentenceTransformer(model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from exc
        embedding = self._local_model.encode(text)
        return embedding.tolist()

    async def store(
        self,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Embed and store a vector with metadata."""
        if embedding is None:
            embedding = await self.embed_text(text)

        packed = _pack_floats(embedding)
        meta_json = json.dumps(metadata or {})
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (id, text, embedding, dimensions, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (id, text, packed, len(embedding), meta_json, now),
            )
            await db.commit()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Semantic similarity search."""
        if query_embedding is None:
            query_embedding = await self.embed_text(query)

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, text, embedding, metadata FROM embeddings"
            )
            rows = await cursor.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            stored_embedding = _unpack_floats(row[2])
            sim = _cosine_similarity(query_embedding, stored_embedding)
            if sim >= min_similarity:
                results.append(
                    SearchResult(
                        id=row[0],
                        text=row[1],
                        similarity=sim,
                        metadata=json.loads(row[3]),
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    async def find_contradictions(
        self,
        claim_text: str,
        threshold: float = 0.8,
        claim_embedding: list[float] | None = None,
    ) -> list[Contradiction]:
        """Find claims that are semantically similar but potentially contradictory.

        High similarity with different content suggests contradiction.
        """
        results = await self.search(
            claim_text,
            top_k=20,
            min_similarity=threshold,
            query_embedding=claim_embedding,
        )
        contradictions: list[Contradiction] = []
        for r in results:
            # Skip if same text (exact match, not a contradiction)
            if r.text.strip().lower() == claim_text.strip().lower():
                continue
            contradictions.append(
                Contradiction(
                    id=r.id,
                    text=r.text,
                    similarity=r.similarity,
                    metadata=r.metadata,
                )
            )
        return contradictions

    async def delete(self, id: str) -> bool:
        """Remove an embedding by ID."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM embeddings WHERE id = ?", (id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def count(self) -> int:
        """Return total number of stored embeddings."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
            row = await cursor.fetchone()
        return row[0]
