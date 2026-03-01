"""Tests for QuestionStore — CRUD, tree ordering, multi-inquiry isolation."""

from __future__ import annotations

import pytest

from qe.services.inquiry.schemas import Question
from qe.substrate.question_store import QuestionStore


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test_questions.db")
    s = QuestionStore(db_path)
    await s.initialize()
    return s


class TestQuestionStoreInit:
    @pytest.mark.asyncio
    async def test_initialize_creates_table(self, store):
        # Just verify no error on init
        assert store is not None

    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(self, store):
        await store.initialize()  # Should not error


class TestQuestionStoreCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        q = Question(text="What is X?", question_type="factual")
        await store.save_question("inq_1", q)

        questions = await store.get_questions_for_inquiry("inq_1")
        assert len(questions) == 1
        assert questions[0].text == "What is X?"
        assert questions[0].question_id == q.question_id

    @pytest.mark.asyncio
    async def test_save_multiple(self, store):
        for i in range(5):
            q = Question(text=f"Question {i}", iteration_generated=i)
            await store.save_question("inq_1", q)

        questions = await store.get_questions_for_inquiry("inq_1")
        assert len(questions) == 5

    @pytest.mark.asyncio
    async def test_update_status(self, store):
        q = Question(text="Pending question")
        await store.save_question("inq_1", q)

        await store.update_question_status(
            q.question_id, "answered", "The answer", 0.9
        )

        questions = await store.get_questions_for_inquiry("inq_1")
        assert questions[0].status == "answered"
        assert questions[0].answer == "The answer"
        assert questions[0].confidence_in_answer == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_get_unanswered(self, store):
        q1 = Question(text="Pending 1", status="pending")
        q2 = Question(text="Answered", status="answered")
        q3 = Question(text="Investigating", status="investigating")
        await store.save_question("inq_1", q1)
        await store.save_question("inq_1", q2)
        await store.save_question("inq_1", q3)

        # Note: status is set on save, need to update after
        await store.update_question_status(q2.question_id, "answered", "A", 0.8)

        unanswered = await store.get_unanswered("inq_1")
        # q1 (pending) + q3 (investigating) — q2 was updated to answered
        assert len(unanswered) == 2
        texts = {q.text for q in unanswered}
        assert "Pending 1" in texts
        assert "Investigating" in texts


class TestQuestionStoreTree:
    @pytest.mark.asyncio
    async def test_tree_ordering(self, store):
        root = Question(text="Root Q", parent_id=None)
        child1 = Question(text="Child 1", parent_id=root.question_id)
        child2 = Question(text="Child 2", parent_id=root.question_id)
        grandchild = Question(text="Grandchild", parent_id=child1.question_id)

        await store.save_question("inq_1", root)
        await store.save_question("inq_1", child1)
        await store.save_question("inq_1", child2)
        await store.save_question("inq_1", grandchild)

        tree = await store.get_question_tree("inq_1")
        assert len(tree) == 4
        # Root should be first
        assert tree[0].text == "Root Q"
        # Grandchild should come after its parent
        child1_idx = next(i for i, q in enumerate(tree) if q.text == "Child 1")
        gc_idx = next(i for i, q in enumerate(tree) if q.text == "Grandchild")
        assert gc_idx > child1_idx


class TestQuestionStoreIsolation:
    @pytest.mark.asyncio
    async def test_multi_inquiry_isolation(self, store):
        q1 = Question(text="Inquiry 1 Q")
        q2 = Question(text="Inquiry 2 Q")
        await store.save_question("inq_1", q1)
        await store.save_question("inq_2", q2)

        inq1_qs = await store.get_questions_for_inquiry("inq_1")
        inq2_qs = await store.get_questions_for_inquiry("inq_2")

        assert len(inq1_qs) == 1
        assert inq1_qs[0].text == "Inquiry 1 Q"
        assert len(inq2_qs) == 1
        assert inq2_qs[0].text == "Inquiry 2 Q"

    @pytest.mark.asyncio
    async def test_empty_inquiry(self, store):
        questions = await store.get_questions_for_inquiry("nonexistent")
        assert questions == []
