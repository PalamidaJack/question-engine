"""Model Intelligence Service — continuous model discovery, profiling, and ranking."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from qe.models.envelope import Envelope
from qe.services.model_intelligence.knowledge import ModelKnowledgeAgent
from qe.services.model_intelligence.profiler import ModelProfiler

log = logging.getLogger(__name__)

# Re-profile models older than 7 days (seconds)
_STALE_THRESHOLD_S = 7 * 24 * 3600
# Background loop interval
_POLL_INTERVAL_S = 300


class ModelIntelligenceService:
    """Coordinates model discovery, benchmark profiling, and knowledge management.

    Maintains a SQLite database of model profiles, benchmark results,
    aggregate scores, rankings, user preferences, and A/B comparison outcomes.
    A background task periodically checks the discovery service for new or
    stale models and triggers automated profiling.
    """

    service_name = "model_intelligence"

    def __init__(
        self,
        bus: Any,
        discovery_service: Any | None = None,
        db_path: str = "data/model_intelligence/models.db",
        profiles_dir: str = "data/model_intelligence/profiles",
    ) -> None:
        self._bus = bus
        self._discovery = discovery_service
        self._db_path = db_path
        self._profiles_dir = profiles_dir

        self._profiler = ModelProfiler()
        self._knowledge = ModelKnowledgeAgent(profiles_dir=profiles_dir)

        self._db: aiosqlite.Connection | None = None
        self._running = False
        self._bg_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the SQLite database and start background profiling."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        Path(self._profiles_dir).mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._initialize_db()

        self._running = True
        self._bg_task = asyncio.create_task(self._background_profiler())
        log.info(
            "model_intelligence.started db=%s profiles=%s",
            self._db_path,
            self._profiles_dir,
        )

    async def stop(self) -> None:
        """Cancel background tasks and close the database."""
        self._running = False
        if self._bg_task is not None:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None

        if self._db is not None:
            await self._db.close()
            self._db = None
        log.info("model_intelligence.stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_model_profile(self, model_id: str) -> dict | None:
        """Return full profile for a model (row + scores + narrative)."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM models WHERE model_id = ?", (model_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None

        profile = dict(row)
        profile["capabilities"] = json.loads(profile.get("capabilities", "[]"))
        profile["metadata"] = json.loads(profile.get("metadata", "{}"))
        profile["scores"] = await self.get_model_scores(model_id)
        profile["narrative"] = self._knowledge.load_profile(model_id)
        return profile

    async def get_model_scores(self, model_id: str) -> dict | None:
        """Return aggregate benchmark scores for a model."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM model_scores WHERE model_id = ?", (model_id,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return None
        return {
            row["category"]: {
                "avg_score": row["avg_score"],
                "p50_latency_ms": row["p50_latency_ms"],
                "p95_latency_ms": row["p95_latency_ms"],
                "consistency_score": row["consistency_score"],
                "sample_count": row["sample_count"],
                "last_updated": row["last_updated"],
            }
            for row in rows
        }

    async def list_models(
        self, filters: dict | None = None,
    ) -> list[dict]:
        """List all tracked models with summary information."""
        assert self._db is not None
        query = "SELECT * FROM models"
        params: list[Any] = []

        clauses: list[str] = []
        if filters:
            if "provider" in filters:
                clauses.append("provider = ?")
                params.append(filters["provider"])
            if "maker" in filters:
                clauses.append("maker = ?")
                params.append(filters["maker"])
            if "status" in filters:
                clauses.append("status = ?")
                params.append(filters["status"])
            if "is_free" in filters:
                clauses.append("is_free = ?")
                params.append(1 if filters["is_free"] else 0)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY last_seen DESC"

        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()

        results: list[dict] = []
        for row in rows:
            entry = dict(row)
            entry["capabilities"] = json.loads(
                entry.get("capabilities", "[]"),
            )
            entry["metadata"] = json.loads(entry.get("metadata", "{}"))
            results.append(entry)
        return results

    async def get_rankings(
        self, task_type: str | None = None,
    ) -> list[dict]:
        """Return ranked models, optionally filtered by task type."""
        assert self._db is not None
        if task_type:
            query = (
                "SELECT * FROM model_rankings "
                "WHERE task_type = ? ORDER BY rank ASC"
            )
            params: tuple[Any, ...] = (task_type,)
        else:
            query = "SELECT * FROM model_rankings ORDER BY task_type, rank ASC"
            params = ()

        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def trigger_profile(self, model_id: str) -> bool:
        """Manually trigger (re-)profiling for a specific model.

        Returns True if profiling was initiated, False if the model
        is unknown to the discovery service.
        """
        if self._discovery is not None:
            model = self._discovery.get_model(model_id)
            if model is None:
                log.warning(
                    "model_intelligence.trigger_profile unknown model=%s",
                    model_id,
                )
                return False

        await self._profile_model(model_id)
        return True

    async def get_markdown_profile(self, model_id: str) -> str:
        """Return the narrative markdown profile for a model."""
        md = self._knowledge.load_profile(model_id)
        return md or ""

    async def set_user_preference(
        self,
        model_id: str,
        task_type: str,
        preference: str,
    ) -> None:
        """Record a user preference (pin / block / prefer) for a model."""
        assert self._db is not None
        now = time.time()
        await self._db.execute(
            "INSERT INTO user_preferences (model_id, task_type, preference, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(model_id, task_type) DO UPDATE SET "
            "preference = excluded.preference, updated_at = excluded.updated_at",
            (model_id, task_type, preference, now),
        )
        await self._db.commit()
        log.info(
            "model_intelligence.preference_set model=%s task=%s pref=%s",
            model_id,
            task_type,
            preference,
        )

    async def record_preference(
        self,
        model_a: str,
        model_b: str,
        winner: str,
        task_context: str,
    ) -> None:
        """Record the outcome of a multi-model comparison."""
        assert self._db is not None
        now = time.time()
        row_id = uuid.uuid4().hex[:16]
        await self._db.execute(
            "INSERT INTO comparison_results "
            "(id, model_a, model_b, winner, task_context, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row_id, model_a, model_b, winner, task_context, now),
        )
        await self._db.commit()
        log.info(
            "model_intelligence.comparison_recorded a=%s b=%s winner=%s",
            model_a,
            model_b,
            winner,
        )

    async def stats(self) -> dict:
        """Return overview statistics for the service."""
        assert self._db is not None
        counts: dict[str, int] = {}
        for table in (
            "models",
            "benchmark_results",
            "model_scores",
            "model_rankings",
            "user_preferences",
            "comparison_results",
        ):
            async with self._db.execute(
                f"SELECT COUNT(*) FROM {table}",  # noqa: S608
            ) as cur:
                row = await cur.fetchone()
                counts[table] = row[0] if row else 0

        profiles_on_disk = len(self._knowledge.list_profiles())
        return {
            "service": self.service_name,
            "running": self._running,
            "db_path": self._db_path,
            "profiles_dir": self._profiles_dir,
            "table_counts": counts,
            "narrative_profiles": profiles_on_disk,
        }

    # ------------------------------------------------------------------
    # SQLite initialisation
    # ------------------------------------------------------------------

    async def _initialize_db(self) -> None:
        """Create tables if they do not exist."""
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                provider TEXT,
                maker TEXT,
                context_length INTEGER,
                is_free INTEGER DEFAULT 0,
                capabilities TEXT DEFAULT '[]',
                first_seen REAL,
                last_seen REAL,
                last_profiled REAL,
                status TEXT DEFAULT 'discovered',
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS benchmark_results (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                category TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                score REAL,
                latency_ms INTEGER,
                raw_response TEXT,
                timestamp REAL,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );

            CREATE TABLE IF NOT EXISTS model_scores (
                model_id TEXT NOT NULL,
                category TEXT NOT NULL,
                avg_score REAL,
                p50_latency_ms INTEGER,
                p95_latency_ms INTEGER,
                consistency_score REAL,
                sample_count INTEGER DEFAULT 0,
                last_updated REAL,
                PRIMARY KEY (model_id, category)
            );

            CREATE TABLE IF NOT EXISTS model_rankings (
                task_type TEXT NOT NULL,
                rank INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                composite_score REAL,
                reason TEXT,
                last_updated REAL,
                PRIMARY KEY (task_type, rank)
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                model_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                preference TEXT NOT NULL,
                updated_at REAL,
                PRIMARY KEY (model_id, task_type)
            );

            CREATE TABLE IF NOT EXISTS comparison_results (
                id TEXT PRIMARY KEY,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                winner TEXT NOT NULL,
                task_context TEXT,
                timestamp REAL
            );
        """)
        await self._db.commit()

    # ------------------------------------------------------------------
    # Background profiler
    # ------------------------------------------------------------------

    async def _background_profiler(self) -> None:
        """Periodically scan for new or stale models and profile them."""
        while self._running:
            try:
                await asyncio.sleep(_POLL_INTERVAL_S)
                if not self._running:
                    break
                await self._profiler_tick()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("model_intelligence.bg_profiler_error")

    async def _profiler_tick(self) -> None:
        """Single iteration of the background profiler."""
        if self._discovery is None:
            return
        assert self._db is not None

        now = time.time()
        discovered = self._discovery.get_available_models(free_only=False)

        for model in discovered:
            mid = model.model_id
            # Upsert into models table
            await self._db.execute(
                "INSERT INTO models "
                "(model_id, provider, maker, context_length, is_free, "
                "capabilities, first_seen, last_seen, status, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET "
                "last_seen = excluded.last_seen, "
                "status = excluded.status",
                (
                    mid,
                    model.provider,
                    getattr(model, "base_model_name", ""),
                    model.context_length,
                    1 if model.is_free else 0,
                    json.dumps(self._extract_capabilities(model)),
                    now,
                    now,
                    "discovered",
                    "{}",
                ),
            )
            await self._db.commit()

            # Check if profiling is needed
            async with self._db.execute(
                "SELECT last_profiled FROM models WHERE model_id = ?",
                (mid,),
            ) as cur:
                row = await cur.fetchone()

            needs_profile = False
            if row is None:
                needs_profile = True
            else:
                last_profiled = row["last_profiled"]
                if last_profiled is None:
                    needs_profile = True
                elif (now - last_profiled) > _STALE_THRESHOLD_S:
                    needs_profile = True

            if needs_profile:
                try:
                    await self._profile_model(mid)
                except Exception:
                    log.exception(
                        "model_intelligence.profile_failed model=%s", mid,
                    )

    # ------------------------------------------------------------------
    # Profiling pipeline
    # ------------------------------------------------------------------

    async def _profile_model(self, model_id: str) -> None:
        """Run the full benchmark suite on a model and persist results."""
        assert self._db is not None
        log.info("model_intelligence.profile_started model=%s", model_id)
        self._publish(
            "model_intelligence.profile_started",
            {"model_id": model_id},
        )

        now = time.time()
        results = await self._profiler.run_benchmarks(model_id)

        # Persist individual benchmark results
        for _category, benchmarks in results.items():
            for b in benchmarks:
                row_id = uuid.uuid4().hex[:16]
                await self._db.execute(
                    "INSERT OR REPLACE INTO benchmark_results "
                    "(id, model_id, category, benchmark_name, score, "
                    "latency_ms, raw_response, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row_id,
                        model_id,
                        b["category"],
                        b["name"],
                        b["score"],
                        b["latency_ms"],
                        b["raw"],
                        now,
                    ),
                )

        # Compute and persist aggregate scores per category
        for category, benchmarks in results.items():
            scores = [b["score"] for b in benchmarks]
            latencies = [b["latency_ms"] for b in benchmarks]
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            sorted_lat = sorted(latencies)
            p50_idx = len(sorted_lat) // 2
            p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
            p50 = sorted_lat[p50_idx] if sorted_lat else 0
            p95 = sorted_lat[p95_idx] if sorted_lat else 0

            # Consistency: low std-dev relative to mean => high consistency
            if len(scores) > 1:
                mean = avg_score
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                std_dev = variance ** 0.5
                consistency = max(0.0, 1.0 - std_dev)
            else:
                consistency = 1.0

            await self._db.execute(
                "INSERT OR REPLACE INTO model_scores "
                "(model_id, category, avg_score, p50_latency_ms, "
                "p95_latency_ms, consistency_score, sample_count, "
                "last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    category,
                    round(avg_score, 4),
                    p50,
                    p95,
                    round(consistency, 4),
                    len(scores),
                    now,
                ),
            )

        # Mark model as profiled
        await self._db.execute(
            "UPDATE models SET last_profiled = ?, status = 'profiled' "
            "WHERE model_id = ?",
            (now, model_id),
        )
        await self._db.commit()

        # Generate narrative profile
        scores_dict = await self.get_model_scores(model_id) or {}
        metadata = await self._load_model_metadata(model_id)
        await self._generate_narrative(model_id, scores_dict, metadata)

        # Update rankings
        await self._recompute_rankings()

        self._publish(
            "model_intelligence.profile_completed",
            {
                "model_id": model_id,
                "categories": list(results.keys()),
                "benchmark_count": sum(len(v) for v in results.values()),
            },
        )
        log.info(
            "model_intelligence.profile_completed model=%s categories=%d",
            model_id,
            len(results),
        )

    async def _generate_narrative(
        self,
        model_id: str,
        scores: dict,
        metadata: dict | None = None,
    ) -> None:
        """Generate and save a markdown narrative profile."""
        try:
            md = await self._knowledge.generate_profile(
                model_id, scores, metadata or {},
            )
            self._knowledge.save_profile(model_id, md)
        except Exception:
            log.exception(
                "model_intelligence.narrative_failed model=%s", model_id,
            )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    async def _recompute_rankings(self) -> None:
        """Recompute model rankings for each benchmark category."""
        assert self._db is not None
        now = time.time()

        # Gather all scored models
        async with self._db.execute(
            "SELECT DISTINCT model_id FROM model_scores",
        ) as cur:
            model_ids = [row["model_id"] for row in await cur.fetchall()]

        if not model_ids:
            return

        # Rank per category (= task_type)
        categories = self._profiler.BENCHMARK_CATEGORIES
        for category in categories:
            scored: list[tuple[str, float]] = []
            for mid in model_ids:
                async with self._db.execute(
                    "SELECT avg_score FROM model_scores "
                    "WHERE model_id = ? AND category = ?",
                    (mid, category),
                ) as cur:
                    row = await cur.fetchone()
                if row and row["avg_score"] is not None:
                    scored.append((mid, row["avg_score"]))

            scored.sort(key=lambda x: x[1], reverse=True)

            # Clear old rankings for this task type
            await self._db.execute(
                "DELETE FROM model_rankings WHERE task_type = ?",
                (category,),
            )

            for rank, (mid, score) in enumerate(scored, start=1):
                await self._db.execute(
                    "INSERT INTO model_rankings "
                    "(task_type, rank, model_id, composite_score, "
                    "reason, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        category,
                        rank,
                        mid,
                        round(score, 4),
                        f"avg_score={score:.4f}",
                        now,
                    ),
                )

        # Overall ranking (mean of category averages)
        overall: list[tuple[str, float]] = []
        for mid in model_ids:
            async with self._db.execute(
                "SELECT AVG(avg_score) AS overall "
                "FROM model_scores WHERE model_id = ?",
                (mid,),
            ) as cur:
                row = await cur.fetchone()
            if row and row["overall"] is not None:
                overall.append((mid, row["overall"]))

        overall.sort(key=lambda x: x[1], reverse=True)
        await self._db.execute(
            "DELETE FROM model_rankings WHERE task_type = 'overall'",
        )
        for rank, (mid, score) in enumerate(overall, start=1):
            await self._db.execute(
                "INSERT INTO model_rankings "
                "(task_type, rank, model_id, composite_score, "
                "reason, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "overall",
                    rank,
                    mid,
                    round(score, 4),
                    f"mean_category_avg={score:.4f}",
                    now,
                ),
            )

        await self._db.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_model_metadata(self, model_id: str) -> dict:
        """Load metadata dict from the models table."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT provider, maker, context_length, is_free, "
            "capabilities, metadata FROM models WHERE model_id = ?",
            (model_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return {}
        return {
            "provider": row["provider"],
            "maker": row["maker"],
            "context_length": row["context_length"],
            "is_free": bool(row["is_free"]),
            "capabilities": json.loads(row["capabilities"] or "[]"),
            "extra": json.loads(row["metadata"] or "{}"),
        }

    @staticmethod
    def _extract_capabilities(model: Any) -> list[str]:
        """Pull capability flags from a DiscoveredModel."""
        caps: list[str] = []
        if getattr(model, "supports_tool_calling", False):
            caps.append("tool_calling")
        if getattr(model, "supports_json_mode", False):
            caps.append("json_mode")
        if getattr(model, "supports_streaming", False):
            caps.append("streaming")
        if getattr(model, "supports_system_messages", False):
            caps.append("system_messages")
        return caps

    def _publish(self, topic: str, payload: dict) -> None:
        """Publish an event to the bus."""
        if self._bus is not None:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id=self.service_name,
                    payload=payload,
                ),
            )
