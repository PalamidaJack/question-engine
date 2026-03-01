"""Procedural Memory — Tier 3: learned question templates and tool sequences.

Tracks which question patterns and tool sequences have been effective,
enabling the system to improve its inquiry strategy over time.
"""

from __future__ import annotations

import json
import logging
import uuid

import aiosqlite
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class QuestionTemplate(BaseModel):
    """A learned question pattern with success tracking."""

    template_id: str = Field(default_factory=lambda: f"qt_{uuid.uuid4().hex[:12]}")
    pattern: str
    question_type: str = "factual"
    domain: str = "general"
    success_count: int = 0
    failure_count: int = 0
    avg_info_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class ToolSequence(BaseModel):
    """A learned tool execution sequence with success tracking."""

    sequence_id: str = Field(default_factory=lambda: f"ts_{uuid.uuid4().hex[:12]}")
    tool_names: list[str] = Field(default_factory=list)
    description: str = ""
    domain: str = "general"
    success_count: int = 0
    failure_count: int = 0
    avg_cost_usd: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class ProceduralMemory:
    """Tier 3 memory: persists learned question templates and tool sequences.

    Supports in-memory mode (db_path=None) for testing.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._in_memory = db_path is None
        # In-memory storage when no db_path
        self._templates: dict[str, QuestionTemplate] = {}
        self._sequences: dict[str, ToolSequence] = {}

    async def initialize(self) -> None:
        """Create tables if using SQLite."""
        if self._in_memory:
            return

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS procedural_templates (
                    template_id TEXT PRIMARY KEY,
                    pattern TEXT NOT NULL,
                    question_type TEXT DEFAULT 'factual',
                    domain TEXT DEFAULT 'general',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_info_gain REAL DEFAULT 0.0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS procedural_sequences (
                    sequence_id TEXT PRIMARY KEY,
                    tool_names TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    domain TEXT DEFAULT 'general',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_cost_usd REAL DEFAULT 0.0
                )
            """)
            await db.commit()

    async def record_template_outcome(
        self,
        template_id: str | None,
        pattern: str,
        question_type: str,
        success: bool,
        info_gain: float = 0.0,
        domain: str = "general",
    ) -> QuestionTemplate:
        """Record the outcome of using a question template."""
        tid = template_id or f"qt_{uuid.uuid4().hex[:12]}"

        if self._in_memory:
            tmpl = self._templates.get(tid)
            if tmpl is None:
                tmpl = QuestionTemplate(
                    template_id=tid,
                    pattern=pattern,
                    question_type=question_type,
                    domain=domain,
                )
                self._templates[tid] = tmpl

            if success:
                tmpl.success_count += 1
            else:
                tmpl.failure_count += 1

            total = tmpl.success_count + tmpl.failure_count
            tmpl.avg_info_gain = (
                (tmpl.avg_info_gain * (total - 1) + info_gain) / total
            )
            return tmpl

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM procedural_templates WHERE template_id = ?",
                (tid,),
            )
            row = await cursor.fetchone()

            if row is None:
                s_count = 1 if success else 0
                f_count = 0 if success else 1
                await db.execute(
                    """
                    INSERT INTO procedural_templates
                    (template_id, pattern, question_type, domain,
                     success_count, failure_count, avg_info_gain)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tid, pattern, question_type, domain, s_count, f_count, info_gain),
                )
            else:
                s_delta = 1 if success else 0
                f_delta = 0 if success else 1
                old_total = row[4] + row[5]
                new_total = old_total + 1
                new_avg = (row[6] * old_total + info_gain) / new_total
                await db.execute(
                    """
                    UPDATE procedural_templates
                    SET success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        avg_info_gain = ?
                    WHERE template_id = ?
                    """,
                    (s_delta, f_delta, new_avg, tid),
                )
            await db.commit()

            cursor = await db.execute(
                "SELECT * FROM procedural_templates WHERE template_id = ?",
                (tid,),
            )
            row = await cursor.fetchone()

        return QuestionTemplate(
            template_id=row[0],
            pattern=row[1],
            question_type=row[2],
            domain=row[3],
            success_count=row[4],
            failure_count=row[5],
            avg_info_gain=row[6],
        )

    async def record_sequence_outcome(
        self,
        sequence_id: str | None,
        tool_names: list[str],
        success: bool,
        cost_usd: float = 0.0,
        domain: str = "general",
    ) -> ToolSequence:
        """Record the outcome of using a tool sequence."""
        sid = sequence_id or f"ts_{uuid.uuid4().hex[:12]}"

        if self._in_memory:
            seq = self._sequences.get(sid)
            if seq is None:
                seq = ToolSequence(
                    sequence_id=sid,
                    tool_names=tool_names,
                    domain=domain,
                )
                self._sequences[sid] = seq

            if success:
                seq.success_count += 1
            else:
                seq.failure_count += 1

            total = seq.success_count + seq.failure_count
            seq.avg_cost_usd = (
                (seq.avg_cost_usd * (total - 1) + cost_usd) / total
            )
            return seq

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM procedural_sequences WHERE sequence_id = ?",
                (sid,),
            )
            row = await cursor.fetchone()

            if row is None:
                s_count = 1 if success else 0
                f_count = 0 if success else 1
                await db.execute(
                    """
                    INSERT INTO procedural_sequences
                    (sequence_id, tool_names, description, domain,
                     success_count, failure_count, avg_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sid, json.dumps(tool_names), "", domain, s_count, f_count, cost_usd),
                )
            else:
                s_delta = 1 if success else 0
                f_delta = 0 if success else 1
                old_total = row[4] + row[5]
                new_total = old_total + 1
                new_avg = (row[6] * old_total + cost_usd) / new_total
                await db.execute(
                    """
                    UPDATE procedural_sequences
                    SET success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        avg_cost_usd = ?
                    WHERE sequence_id = ?
                    """,
                    (s_delta, f_delta, new_avg, sid),
                )
            await db.commit()

            cursor = await db.execute(
                "SELECT * FROM procedural_sequences WHERE sequence_id = ?",
                (sid,),
            )
            row = await cursor.fetchone()

        return ToolSequence(
            sequence_id=row[0],
            tool_names=json.loads(row[1]),
            description=row[2],
            domain=row[3],
            success_count=row[4],
            failure_count=row[5],
            avg_cost_usd=row[6],
        )

    async def get_best_templates(
        self,
        domain: str = "general",
        top_k: int = 5,
    ) -> list[QuestionTemplate]:
        """Return top-performing question templates for a domain."""
        if self._in_memory:
            candidates = [
                t for t in self._templates.values()
                if t.domain == domain and (t.success_count + t.failure_count) > 0
            ]
            candidates.sort(key=lambda t: t.success_rate, reverse=True)
            return candidates[:top_k]

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT * FROM procedural_templates
                WHERE domain = ? AND (success_count + failure_count) > 0
                ORDER BY CAST(success_count AS REAL) / (success_count + failure_count) DESC
                LIMIT ?
                """,
                (domain, top_k),
            )
            rows = await cursor.fetchall()

        return [
            QuestionTemplate(
                template_id=r[0],
                pattern=r[1],
                question_type=r[2],
                domain=r[3],
                success_count=r[4],
                failure_count=r[5],
                avg_info_gain=r[6],
            )
            for r in rows
        ]

    async def get_best_sequences(
        self,
        domain: str = "general",
        top_k: int = 5,
    ) -> list[ToolSequence]:
        """Return top-performing tool sequences for a domain."""
        if self._in_memory:
            candidates = [
                s for s in self._sequences.values()
                if s.domain == domain and (s.success_count + s.failure_count) > 0
            ]
            candidates.sort(key=lambda s: s.success_rate, reverse=True)
            return candidates[:top_k]

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT * FROM procedural_sequences
                WHERE domain = ? AND (success_count + failure_count) > 0
                ORDER BY CAST(success_count AS REAL) / (success_count + failure_count) DESC
                LIMIT ?
                """,
                (domain, top_k),
            )
            rows = await cursor.fetchall()

        return [
            ToolSequence(
                sequence_id=r[0],
                tool_names=json.loads(r[1]),
                description=r[2],
                domain=r[3],
                success_count=r[4],
                failure_count=r[5],
                avg_cost_usd=r[6],
            )
            for r in rows
        ]
