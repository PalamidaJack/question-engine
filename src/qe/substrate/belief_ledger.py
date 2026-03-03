import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from yoyo import get_backend, read_migrations

from qe.models.claim import Claim, NullResult, Prediction

log = logging.getLogger(__name__)


class BeliefLedger:
    def __init__(
        self,
        db_path: str,
        migrations_dir: Path,
        decay_half_life_hours: float = 720.0,
    ) -> None:
        self._db_path = db_path
        self._migrations_dir = migrations_dir
        self._decay_half_life_hours = decay_half_life_hours
        import asyncio
        self._db_lock = asyncio.Lock()
        self._db_conn = None

    def _apply_decay(self, claim: Claim, now: datetime) -> Claim:
        """Return a copy of *claim* with confidence adjusted for age.

        Uses exponential decay: ``confidence × 0.5^(age_hours / half_life)``.
        Claims past their ``valid_until`` are set to ``confidence = 0.0``.
        """
        if claim.valid_until is not None and now > claim.valid_until:
            return claim.model_copy(update={"confidence": 0.0})

        age_hours = (now - claim.created_at).total_seconds() / 3600
        if age_hours <= 0 or self._decay_half_life_hours <= 0:
            return claim

        factor = math.pow(0.5, age_hours / self._decay_half_life_hours)
        return claim.model_copy(update={"confidence": claim.confidence * factor})

    async def _get_db(self):
        if self._db_conn is None:
            self._db_conn = await aiosqlite.connect(self._db_path)
            self._db_conn.row_factory = aiosqlite.Row
            await self._db_conn.execute("PRAGMA journal_mode=WAL;")
            await self._db_conn.execute("PRAGMA foreign_keys=ON;")
            await self._db_conn.commit()
        return self._db_conn

    async def initialize(self) -> None:
        """Initialize the database connection and apply migrations."""
        await self._get_db()
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        """Apply any unapplied migrations. Safe to call on every startup."""
        backend = get_backend(f"sqlite:///{self._db_path}")
        migrations = read_migrations(str(self._migrations_dir))

        with backend.lock():
            backend.apply_migrations(backend.to_apply(migrations))

    async def commit_claim(self, claim: Claim) -> Claim:
        """
        Write claim. If a claim with the same (subject_entity_id, predicate) exists:
        - If new confidence > old confidence: supersede the old
        - If new confidence <= old confidence: store as alternative (no supersession)
        """
        db = await self._get_db()
        async with self._db_lock:
            # Acquire write lock before read-modify-write to prevent race conditions
            await db.execute("BEGIN IMMEDIATE")

            # Check for existing claims with same subject and predicate
            cursor = await db.execute(
                """
                SELECT claim_id, confidence
                FROM claims
                WHERE subject_entity_id = ? AND predicate = ? AND superseded_by IS NULL
                """,
                (claim.subject_entity_id, claim.predicate)
            )
            existing = await cursor.fetchone()

            if existing and existing[1] < claim.confidence:
                # Supersede the old claim
                old_claim_id = existing[0]
                await db.execute(
                    "UPDATE claims SET superseded_by = ? WHERE claim_id = ?",
                    (claim.claim_id, old_claim_id)
                )
                log.info("Superseded claim %s with %s", old_claim_id, claim.claim_id)

            # Insert the new claim
            await db.execute(
                """
                INSERT INTO claims (
                    claim_id, schema_version, subject_entity_id, predicate,
                    object_value, confidence, source_service_id, source_envelope_ids,
                    created_at, valid_until, superseded_by, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    claim.schema_version,
                    claim.subject_entity_id,
                    claim.predicate,
                    claim.object_value,
                    claim.confidence,
                    claim.source_service_id,
                    json.dumps(claim.source_envelope_ids),
                    claim.created_at.isoformat(),
                    claim.valid_until.isoformat() if claim.valid_until else None,
                    claim.superseded_by,
                    json.dumps(claim.tags),
                    json.dumps(claim.metadata)
                )
            )
            await db.commit()

        return claim

    async def get_claims(
        self,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        min_confidence: float = 0.0,
        tags: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[Claim]:
        """Standard filtered read. include_superseded=False returns only current beliefs."""
        query = "SELECT * FROM claims WHERE confidence >= ?"
        params: list = [min_confidence]

        if subject_entity_id:
            query += " AND subject_entity_id = ?"
            params.append(subject_entity_id)

        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)

        if not include_superseded:
            query += " AND superseded_by IS NULL"

        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        now = datetime.now(UTC)
        claims = []
        for row in rows:
            claim = self._apply_decay(self._row_to_claim(row), now)
            if claim.confidence >= min_confidence:
                claims.append(claim)

        return claims

    async def commit_prediction(self, prediction: Prediction) -> Prediction:
        """Write a prediction to the ledger."""
        db = await self._get_db()
        async with self._db_lock:
            await db.execute(
                """
                INSERT INTO predictions (
                    prediction_id, schema_version, statement, confidence,
                    resolution_criteria, resolution_deadline, source_service_id,
                    created_at, resolved_at, resolution, resolution_evidence_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction.prediction_id,
                    prediction.schema_version,
                    prediction.statement,
                    prediction.confidence,
                    prediction.resolution_criteria,
                    (
                        prediction.resolution_deadline.isoformat()
                        if prediction.resolution_deadline
                        else None
                    ),
                    prediction.source_service_id,
                    prediction.created_at.isoformat(),
                    prediction.resolved_at.isoformat() if prediction.resolved_at else None,
                    prediction.resolution,
                    json.dumps(prediction.resolution_evidence_ids)
                )
            )
            await db.commit()

        return prediction

    async def resolve_prediction(
        self,
        prediction_id: str,
        resolution: str,
        evidence_envelope_ids: list[str],
    ) -> Prediction:
        """Resolve a prediction with evidence."""
        db = await self._get_db()
        async with self._db_lock:
            await db.execute(
                """
                UPDATE predictions
                SET resolution = ?, resolved_at = ?, resolution_evidence_ids = ?
                WHERE prediction_id = ?
                """,
                (
                    resolution,
                    datetime.now(UTC).isoformat(),
                    json.dumps(evidence_envelope_ids),
                    prediction_id,
                )
            )
            await db.commit()

            # Fetch within the same connection for atomicity
            cursor = await db.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,)
            )
            row = await cursor.fetchone()

        return self._row_to_prediction(row)

    async def get_open_predictions(self, past_deadline: bool = False) -> list[Prediction]:
        """
        Get open predictions. past_deadline=True returns predictions where
        resolution_deadline < now and unresolved.
        """
        if past_deadline:
            query = """
                SELECT * FROM predictions
                WHERE resolution = 'unresolved' AND resolution_deadline < ?
                ORDER BY resolution_deadline ASC
            """
            params = (datetime.now(UTC).isoformat(),)
        else:
            query = """
                SELECT * FROM predictions
                WHERE resolution = 'unresolved'
                ORDER BY resolution_deadline ASC
            """
            params = ()

        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [self._row_to_prediction(row) for row in rows]

    async def commit_null_result(self, null_result: NullResult) -> NullResult:
        """Write a null result to the ledger."""
        db = await self._get_db()
        async with self._db_lock:
            await db.execute(
                """
                INSERT INTO null_results (
                    null_result_id, schema_version, query, search_scope,
                    source_service_id, created_at, significance, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    null_result.null_result_id,
                    null_result.schema_version,
                    null_result.query,
                    null_result.search_scope,
                    null_result.source_service_id,
                    null_result.created_at.isoformat(),
                    null_result.significance,
                    null_result.notes
                )
            )
            await db.commit()

        return null_result

    async def get_claim_by_id(self, claim_id: str) -> Claim | None:
        """Fetch a single claim by its ID."""
        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(
                "SELECT * FROM claims WHERE claim_id = ?",
                (claim_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_claim(row)

    async def count_claims(self, include_superseded: bool = False) -> int:
        """Return the total number of claims in the ledger."""
        query = "SELECT COUNT(*) FROM claims"
        if not include_superseded:
            query += " WHERE superseded_by IS NULL"
        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(query)
            row = await cursor.fetchone()
        return row[0]

    async def get_claims_since(
        self,
        since: datetime,
        include_superseded: bool = False,
    ) -> list[Claim]:
        """Return claims created after the given timestamp."""
        query = "SELECT * FROM claims WHERE created_at > ?"
        params: list = [since.isoformat()]
        if not include_superseded:
            query += " AND superseded_by IS NULL"
        query += " ORDER BY created_at DESC"
        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        now = datetime.now(UTC)
        return [self._apply_decay(self._row_to_claim(row), now) for row in rows]

    async def retract_claim(self, claim_id: str) -> bool:
        """Soft-retract a claim by marking it as superseded by 'retracted'."""
        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(
                "SELECT claim_id FROM claims WHERE claim_id = ?",
                (claim_id,),
            )
            if not await cursor.fetchone():
                return False

            await db.execute(
                "UPDATE claims SET superseded_by = ? WHERE claim_id = ?",
                ("retracted", claim_id),
            )
            await db.commit()
        return True

    async def search_full_text(self, query: str, limit: int = 20) -> list[Claim]:
        """SQLite FTS5 search across claim fields."""
        # Sanitize: strip FTS5 special characters, keep only words
        words = re.findall(r"\w+", query)
        if not words:
            return []
        # Join with OR for broader matching
        fts_query = " OR ".join(f'"{w}"' for w in words)

        db = await self._get_db()
        async with self._db_lock:
            cursor = await db.execute(
                """
                SELECT c.*
                FROM claims c
                JOIN claims_fts ON c.rowid = claims_fts.rowid
                WHERE claims_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            rows = await cursor.fetchall()

        now = datetime.now(UTC)
        return [self._apply_decay(self._row_to_claim(row), now) for row in rows]

    def _row_to_claim(self, row: aiosqlite.Row) -> Claim:
        """Convert a database row to a Claim object."""
        return Claim(
            claim_id=row["claim_id"],
            schema_version=row["schema_version"],
            subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            confidence=row["confidence"],
            source_service_id=row["source_service_id"],
            source_envelope_ids=json.loads(row["source_envelope_ids"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            superseded_by=row["superseded_by"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"])
        )

    def _row_to_prediction(self, row: aiosqlite.Row) -> Prediction:
        """Convert a database row to a Prediction object."""
        return Prediction(
            prediction_id=row["prediction_id"],
            schema_version=row["schema_version"],
            statement=row["statement"],
            confidence=row["confidence"],
            resolution_criteria=row["resolution_criteria"],
            resolution_deadline=datetime.fromisoformat(row["resolution_deadline"]) if row["resolution_deadline"] else None,
            source_service_id=row["source_service_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
            resolution=row["resolution"],
            resolution_evidence_ids=json.loads(row["resolution_evidence_ids"])
        )
