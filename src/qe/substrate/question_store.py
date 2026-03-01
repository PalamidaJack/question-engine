"""SQLite persistence for inquiry question trees."""

from __future__ import annotations

import json
import logging

import aiosqlite

from qe.services.inquiry.schemas import Question

log = logging.getLogger(__name__)


class QuestionStore:
    """Persists questions for inquiry sessions in SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create the inquiry_questions table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS inquiry_questions (
                    question_id TEXT PRIMARY KEY,
                    inquiry_id TEXT NOT NULL,
                    parent_id TEXT,
                    text TEXT NOT NULL,
                    question_type TEXT NOT NULL DEFAULT 'factual',
                    expected_info_gain REAL DEFAULT 0.5,
                    relevance_to_goal REAL DEFAULT 0.5,
                    novelty_score REAL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'pending',
                    answer TEXT DEFAULT '',
                    evidence TEXT DEFAULT '[]',
                    confidence_in_answer REAL DEFAULT 0.0,
                    hypothesis_id TEXT,
                    iteration_generated INTEGER DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_iq_inquiry
                ON inquiry_questions(inquiry_id)
            """)
            await db.commit()

    async def save_question(self, inquiry_id: str, question: Question) -> None:
        """Insert or replace a question for an inquiry."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO inquiry_questions (
                    question_id, inquiry_id, parent_id, text, question_type,
                    expected_info_gain, relevance_to_goal, novelty_score,
                    status, answer, evidence, confidence_in_answer,
                    hypothesis_id, iteration_generated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question.question_id,
                    inquiry_id,
                    question.parent_id,
                    question.text,
                    question.question_type,
                    question.expected_info_gain,
                    question.relevance_to_goal,
                    question.novelty_score,
                    question.status,
                    question.answer,
                    json.dumps(question.evidence),
                    question.confidence_in_answer,
                    question.hypothesis_id,
                    question.iteration_generated,
                ),
            )
            await db.commit()

    async def get_questions_for_inquiry(self, inquiry_id: str) -> list[Question]:
        """Get all questions for an inquiry, ordered by iteration."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT question_id, parent_id, text, question_type,
                       expected_info_gain, relevance_to_goal, novelty_score,
                       status, answer, evidence, confidence_in_answer,
                       hypothesis_id, iteration_generated
                FROM inquiry_questions
                WHERE inquiry_id = ?
                ORDER BY iteration_generated, question_id
                """,
                (inquiry_id,),
            )
            rows = await cursor.fetchall()

        return [self._row_to_question(row) for row in rows]

    async def get_question_tree(self, inquiry_id: str) -> list[Question]:
        """Get questions in parent-child ordering (roots first, then children)."""
        questions = await self.get_questions_for_inquiry(inquiry_id)

        # Separate roots from children
        roots = [q for q in questions if q.parent_id is None]
        children = [q for q in questions if q.parent_id is not None]

        # Build parent->children index
        by_parent: dict[str, list[Question]] = {}
        for c in children:
            by_parent.setdefault(c.parent_id, []).append(c)  # type: ignore[arg-type]

        # BFS ordering
        ordered: list[Question] = []
        queue = list(roots)
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for child in by_parent.get(node.question_id, []):
                queue.append(child)

        return ordered

    async def update_question_status(
        self,
        question_id: str,
        status: str,
        answer: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Update a question's status, answer, and confidence."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE inquiry_questions
                SET status = ?, answer = ?, confidence_in_answer = ?
                WHERE question_id = ?
                """,
                (status, answer, confidence, question_id),
            )
            await db.commit()

    async def get_unanswered(self, inquiry_id: str) -> list[Question]:
        """Get all pending/investigating questions for an inquiry."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT question_id, parent_id, text, question_type,
                       expected_info_gain, relevance_to_goal, novelty_score,
                       status, answer, evidence, confidence_in_answer,
                       hypothesis_id, iteration_generated
                FROM inquiry_questions
                WHERE inquiry_id = ? AND status IN ('pending', 'investigating')
                ORDER BY iteration_generated, question_id
                """,
                (inquiry_id,),
            )
            rows = await cursor.fetchall()

        return [self._row_to_question(row) for row in rows]

    @staticmethod
    def _row_to_question(row: tuple) -> Question:
        return Question(
            question_id=row[0],
            parent_id=row[1],
            text=row[2],
            question_type=row[3],
            expected_info_gain=row[4],
            relevance_to_goal=row[5],
            novelty_score=row[6],
            status=row[7],
            answer=row[8],
            evidence=json.loads(row[9]) if row[9] else [],
            confidence_in_answer=row[10],
            hypothesis_id=row[11],
            iteration_generated=row[12],
        )
