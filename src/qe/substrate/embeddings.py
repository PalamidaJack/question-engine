"""Vector store backed by SQLite for semantic search and contradiction detection.

Uses litellm for embeddings by default (supports all providers).
Optional upgrade to sentence-transformers for free local embeddings.
HNSW index (via hnswlib) accelerates search from O(n) to O(log n).
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

try:
    import hnswlib

    _HNSW_AVAILABLE = True
except ImportError:
    _HNSW_AVAILABLE = False


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
    """Vector store backed by SQLite with optional HNSW acceleration.

    Embedding generation uses litellm (supports OpenAI, Anthropic, Cohere,
    local models via Ollama, etc.). Falls back to sentence-transformers
    if available and configured.

    When hnswlib is installed, search uses an in-memory HNSW index for
    O(log n) approximate nearest-neighbor lookup instead of brute-force
    cosine scan.  The index is lazily built on first search and kept in
    sync by ``store()`` / ``delete()``.
    """

    def __init__(
        self,
        db_path: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._db_path = db_path
        self._model = model
        self._local_model: Any = None
        # HNSW index state
        self._hnsw_index: Any | None = None
        self._hnsw_id_map: dict[int, str] = {}  # internal int → external str id
        self._hnsw_reverse: dict[str, int] = {}  # external str id → internal int
        self._hnsw_next_label: int = 0
        self._hnsw_dirty: bool = True  # needs rebuild on first use

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

    # ── HNSW index management ─────────────────────────────────────────────

    async def _ensure_hnsw(self) -> None:
        """Lazily build the HNSW index from SQLite on first search."""
        if not _HNSW_AVAILABLE or not self._hnsw_dirty:
            return

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, embedding, dimensions FROM embeddings"
            )
            rows = await cursor.fetchall()

        if not rows:
            self._hnsw_index = None
            self._hnsw_id_map = {}
            self._hnsw_reverse = {}
            self._hnsw_next_label = 0
            self._hnsw_dirty = False
            return

        dim = rows[0][2]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=max(len(rows) * 2, 1024),
            ef_construction=200,
            M=16,
        )
        index.set_ef(50)

        id_map: dict[int, str] = {}
        reverse: dict[str, int] = {}
        for label, row in enumerate(rows):
            vec = _unpack_floats(row[1])
            index.add_items([vec], [label])
            id_map[label] = row[0]
            reverse[row[0]] = label

        self._hnsw_index = index
        self._hnsw_id_map = id_map
        self._hnsw_reverse = reverse
        self._hnsw_next_label = len(rows)
        self._hnsw_dirty = False
        log.debug("hnsw.rebuilt entries=%d dim=%d", len(rows), dim)

    def _hnsw_add(self, ext_id: str, embedding: list[float]) -> None:
        """Add a single vector to the live HNSW index."""
        if not _HNSW_AVAILABLE or self._hnsw_index is None:
            return

        # If id already exists, mark dirty for rebuild (HNSW can't update)
        if ext_id in self._hnsw_reverse:
            self._hnsw_dirty = True
            return

        label = self._hnsw_next_label
        # Resize if needed
        if label >= self._hnsw_index.get_max_elements():
            self._hnsw_index.resize_index(
                self._hnsw_index.get_max_elements() * 2
            )
        self._hnsw_index.add_items([embedding], [label])
        self._hnsw_id_map[label] = ext_id
        self._hnsw_reverse[ext_id] = label
        self._hnsw_next_label = label + 1

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

        # Keep HNSW in sync
        self._hnsw_add(id, embedding)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Semantic similarity search.

        Uses HNSW index for O(log n) search when hnswlib is installed,
        otherwise falls back to brute-force cosine scan.
        """
        if query_embedding is None:
            query_embedding = await self.embed_text(query)

        if _HNSW_AVAILABLE:
            return await self._search_hnsw(
                query_embedding, top_k, min_similarity
            )
        return await self._search_brute(
            query_embedding, top_k, min_similarity
        )

    async def _search_hnsw(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[SearchResult]:
        """HNSW-accelerated approximate nearest-neighbor search."""
        await self._ensure_hnsw()

        if self._hnsw_index is None or self._hnsw_index.get_current_count() == 0:
            return []

        # Query more than top_k to account for min_similarity filtering
        k = min(top_k * 2, self._hnsw_index.get_current_count())
        labels, distances = self._hnsw_index.knn_query([query_embedding], k=k)

        # hnswlib cosine space returns 1 - cosine_similarity as distance
        candidate_ids = []
        candidate_sims = []
        for label, dist in zip(labels[0], distances[0], strict=True):
            sim = 1.0 - dist
            if sim >= min_similarity:
                ext_id = self._hnsw_id_map.get(int(label))
                if ext_id is not None:
                    candidate_ids.append(ext_id)
                    candidate_sims.append(sim)

        if not candidate_ids:
            return []

        # Fetch text + metadata from SQLite for the candidates
        placeholders = ",".join("?" for _ in candidate_ids)
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                f"SELECT id, text, metadata FROM embeddings WHERE id IN ({placeholders})",
                candidate_ids,
            )
            rows = await cursor.fetchall()

        row_map = {r[0]: (r[1], r[2]) for r in rows}
        results: list[SearchResult] = []
        for ext_id, sim in zip(candidate_ids, candidate_sims, strict=True):
            if ext_id in row_map:
                text, meta_json = row_map[ext_id]
                results.append(
                    SearchResult(
                        id=ext_id,
                        text=text,
                        similarity=sim,
                        metadata=json.loads(meta_json),
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    async def _search_brute(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[SearchResult]:
        """Brute-force cosine scan fallback when hnswlib is not installed."""
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
            deleted = cursor.rowcount > 0
        if deleted:
            # HNSW doesn't support single-item removal; mark for rebuild
            self._hnsw_dirty = True
        return deleted

    async def count(self) -> int:
        """Return total number of stored embeddings."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
            row = await cursor.fetchone()
        return row[0]
