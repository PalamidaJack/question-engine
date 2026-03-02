"""SQLite CRUD for Innovation Scout proposals, findings, and feedback."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from qe.models.scout import (
    CodeChange,
    ImprovementIdea,
    ImprovementProposal,
    ScoutFeedbackRecord,
    ScoutFinding,
    TestResult,
)

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class ScoutStore:
    """Async SQLite store for scout proposals, findings, and feedback."""

    def __init__(self, db_path: str = "data/qe.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Apply migration to ensure tables exist."""
        migration = _MIGRATIONS_DIR / "0014_scout_proposals.sql"
        if not migration.exists():
            log.warning("scout_store.migration_missing path=%s", migration)
            return
        sql = migration.read_text(encoding="utf-8")
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(sql)
            await db.commit()

    # ── Findings ──────────────────────────────────────────────────────────

    async def save_finding(self, finding: ScoutFinding) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO scout_findings
                   (finding_id, url, title, snippet, full_content,
                    source_type, relevance_score, discovered_at, tags_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    finding.finding_id,
                    finding.url,
                    finding.title,
                    finding.snippet,
                    finding.full_content,
                    finding.source_type,
                    finding.relevance_score,
                    finding.discovered_at.isoformat(),
                    json.dumps(finding.tags),
                ),
            )
            await db.commit()

    async def has_finding_url(self, url: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM scout_findings WHERE url = ? LIMIT 1", (url,)
            )
            row = await cursor.fetchone()
            return row is not None

    async def get_finding(self, finding_id: str) -> ScoutFinding | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM scout_findings WHERE finding_id = ?",
                (finding_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_finding(row)

    # ── Proposals ─────────────────────────────────────────────────────────

    async def save_proposal(self, proposal: ImprovementProposal) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO scout_proposals
                   (proposal_id, idea_json, status, changes_json,
                    test_result_json, impact_assessment, risk_assessment,
                    rollback_plan, branch_name, worktree_path,
                    hil_envelope_id, reviewer_feedback,
                    created_at, decided_at, applied_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal.proposal_id,
                    proposal.idea.model_dump_json(),
                    proposal.status,
                    json.dumps([c.model_dump() for c in proposal.changes]),
                    proposal.test_result.model_dump_json() if proposal.test_result else None,
                    proposal.impact_assessment,
                    proposal.risk_assessment,
                    proposal.rollback_plan,
                    proposal.branch_name,
                    proposal.worktree_path,
                    proposal.hil_envelope_id,
                    proposal.reviewer_feedback,
                    proposal.created_at.isoformat(),
                    proposal.decided_at.isoformat() if proposal.decided_at else None,
                    proposal.applied_at.isoformat() if proposal.applied_at else None,
                ),
            )
            await db.commit()

    async def get_proposal(self, proposal_id: str) -> ImprovementProposal | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM scout_proposals WHERE proposal_id = ?",
                (proposal_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_proposal(row)

    async def list_proposals(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ImprovementProposal]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if status:
                cursor = await db.execute(
                    "SELECT * FROM scout_proposals WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM scout_proposals ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [_row_to_proposal(r) for r in rows]

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
        reviewer_feedback: str = "",
        decided_at: datetime | None = None,
        applied_at: datetime | None = None,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE scout_proposals
                   SET status = ?,
                       reviewer_feedback = CASE WHEN ? != '' THEN ? ELSE reviewer_feedback END,
                       decided_at = COALESCE(?, decided_at),
                       applied_at = COALESCE(?, applied_at)
                   WHERE proposal_id = ?""",
                (
                    status,
                    reviewer_feedback,
                    reviewer_feedback,
                    decided_at.isoformat() if decided_at else None,
                    applied_at.isoformat() if applied_at else None,
                    proposal_id,
                ),
            )
            await db.commit()

    async def count_proposals(self, status: str | None = None) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            if status:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM scout_proposals WHERE status = ?",
                    (status,),
                )
            else:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM scout_proposals"
                )
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ── Feedback ──────────────────────────────────────────────────────────

    async def save_feedback(self, record: ScoutFeedbackRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO scout_feedback
                   (record_id, proposal_id, decision, feedback,
                    category, source_type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.proposal_id,
                    record.decision,
                    record.feedback,
                    record.category,
                    record.source_type,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()

    async def get_feedback_stats(self) -> dict:
        """Return approval rate by category and source type."""
        async with aiosqlite.connect(self._db_path) as db:
            # Overall
            cursor = await db.execute("SELECT COUNT(*) FROM scout_feedback")
            total = (await cursor.fetchone())[0]
            cursor = await db.execute(
                "SELECT COUNT(*) FROM scout_feedback WHERE decision = 'approved'"
            )
            approved = (await cursor.fetchone())[0]

            # By category
            cursor = await db.execute(
                """SELECT category, decision, COUNT(*) as cnt
                   FROM scout_feedback
                   WHERE category != ''
                   GROUP BY category, decision"""
            )
            by_category: dict[str, dict[str, int]] = {}
            for row in await cursor.fetchall():
                cat = row[0]
                if cat not in by_category:
                    by_category[cat] = {"approved": 0, "rejected": 0}
                by_category[cat][row[1]] = row[2]

            # By source type
            cursor = await db.execute(
                """SELECT source_type, decision, COUNT(*) as cnt
                   FROM scout_feedback
                   WHERE source_type != ''
                   GROUP BY source_type, decision"""
            )
            by_source: dict[str, dict[str, int]] = {}
            for row in await cursor.fetchall():
                src = row[0]
                if src not in by_source:
                    by_source[src] = {"approved": 0, "rejected": 0}
                by_source[src][row[1]] = row[2]

            return {
                "total": total,
                "approved": approved,
                "rejected": total - approved,
                "approval_rate": approved / total if total > 0 else 0.0,
                "by_category": by_category,
                "by_source_type": by_source,
            }

    async def get_rejected_patterns(self) -> dict:
        """Return categories and source types that humans tend to reject."""
        stats = await self.get_feedback_stats()
        rejected_categories: list[str] = []
        for cat, counts in stats.get("by_category", {}).items():
            total = counts.get("approved", 0) + counts.get("rejected", 0)
            if total >= 3 and counts.get("rejected", 0) / total > 0.7:
                rejected_categories.append(cat)

        rejected_sources: list[str] = []
        for src, counts in stats.get("by_source_type", {}).items():
            total = counts.get("approved", 0) + counts.get("rejected", 0)
            if total >= 3 and counts.get("rejected", 0) / total > 0.7:
                rejected_sources.append(src)

        return {
            "rejected_categories": rejected_categories,
            "rejected_sources": rejected_sources,
        }


# ── Helpers ──────────────────────────────────────────────────────────────


def _row_to_finding(row) -> ScoutFinding:
    return ScoutFinding(
        finding_id=row["finding_id"],
        url=row["url"],
        title=row["title"],
        snippet=row["snippet"],
        full_content=row["full_content"],
        source_type=row["source_type"],
        relevance_score=row["relevance_score"],
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
        tags=json.loads(row["tags_json"]),
    )


def _row_to_proposal(row) -> ImprovementProposal:
    idea = ImprovementIdea.model_validate_json(row["idea_json"])
    changes_raw = json.loads(row["changes_json"])
    changes = [CodeChange.model_validate(c) for c in changes_raw]
    test_result = None
    if row["test_result_json"]:
        test_result = TestResult.model_validate_json(row["test_result_json"])

    return ImprovementProposal(
        proposal_id=row["proposal_id"],
        idea=idea,
        status=row["status"],
        changes=changes,
        test_result=test_result,
        impact_assessment=row["impact_assessment"],
        risk_assessment=row["risk_assessment"],
        rollback_plan=row["rollback_plan"],
        branch_name=row["branch_name"],
        worktree_path=row["worktree_path"],
        hil_envelope_id=row["hil_envelope_id"],
        reviewer_feedback=row["reviewer_feedback"],
        created_at=datetime.fromisoformat(row["created_at"]),
        decided_at=(
            datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None
        ),
        applied_at=(
            datetime.fromisoformat(row["applied_at"]) if row["applied_at"] else None
        ),
    )
