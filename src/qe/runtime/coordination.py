"""Fan-out voting coordination protocol for multi-agent consensus.

Provides ``CoordinationProtocol`` which publishes vote requests on the bus,
collects responses, and tallies confidence-weighted results into a
``ConsensusResult``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from qe.models.envelope import Envelope

log = logging.getLogger(__name__)


@dataclass
class Vote:
    """A single vote cast by an agent."""

    agent_id: str
    choice: str
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class ConsensusResult:
    """Outcome of a voting round."""

    vote_id: str
    winning_choice: str
    vote_count: int
    total_votes: int
    confidence: float
    unanimous: bool
    votes: list[Vote] = field(default_factory=list)


@dataclass
class _VoteSession:
    """Internal tracking for an in-progress vote."""

    vote_id: str
    question: str
    options: list[str]
    goal_id: str
    min_voters: int
    votes: list[Vote] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)


class CoordinationProtocol:
    """Fan-out voting for multi-agent consensus."""

    def __init__(self, bus: Any) -> None:
        self.bus = bus
        self._sessions: dict[str, _VoteSession] = {}

    async def start(self) -> None:
        """Subscribe to vote response topic."""
        self.bus.subscribe("coordination.vote_response", self._handle_vote_response)
        log.info("coordination.started")

    async def stop(self) -> None:
        """Unsubscribe from vote response topic."""
        self.bus.unsubscribe("coordination.vote_response", self._handle_vote_response)
        log.info("coordination.stopped")

    async def request_vote(
        self,
        question: str,
        options: list[str],
        goal_id: str = "",
        timeout_seconds: float = 10.0,
        min_voters: int = 1,
    ) -> ConsensusResult:
        """Publish a vote request and wait for responses.

        Returns the best-effort ``ConsensusResult`` after either
        ``min_voters`` have responded or ``timeout`` seconds have elapsed.
        """
        vote_id = f"vote_{uuid.uuid4().hex[:12]}"
        session = _VoteSession(
            vote_id=vote_id,
            question=question,
            options=options,
            goal_id=goal_id,
            min_voters=min_voters,
        )
        self._sessions[vote_id] = session

        # Publish vote request
        self.bus.publish(
            Envelope(
                topic="coordination.vote_request",
                source_service_id="coordination",
                payload={
                    "vote_id": vote_id,
                    "question": question,
                    "options": options,
                    "goal_id": goal_id,
                    "timeout_seconds": timeout_seconds,
                    "min_voters": min_voters,
                },
            )
        )

        # Wait for enough responses or timeout
        try:
            await asyncio.wait_for(session.event.wait(), timeout=timeout_seconds)
        except TimeoutError:
            log.debug(
                "coordination.vote_timeout vote_id=%s collected=%d/%d",
                vote_id,
                len(session.votes),
                min_voters,
            )

        result = self._tally(session)

        # Publish consensus result
        self.bus.publish(
            Envelope(
                topic="coordination.consensus",
                source_service_id="coordination",
                payload={
                    "vote_id": vote_id,
                    "winning_choice": result.winning_choice,
                    "vote_count": result.vote_count,
                    "total_votes": result.total_votes,
                    "confidence": result.confidence,
                    "unanimous": result.unanimous,
                },
            )
        )

        del self._sessions[vote_id]
        return result

    async def _handle_vote_response(self, envelope: Envelope) -> None:
        """Collect a vote, signal when threshold is met."""
        payload = envelope.payload
        vote_id = payload.get("vote_id", "")
        session = self._sessions.get(vote_id)
        if session is None:
            return

        choice = payload.get("choice", "")
        if choice not in session.options:
            log.warning(
                "coordination.invalid_choice vote_id=%s choice=%s options=%s",
                vote_id,
                choice,
                session.options,
            )
            return

        vote = Vote(
            agent_id=payload.get("agent_id", ""),
            choice=choice,
            confidence=payload.get("confidence", 1.0),
            reasoning=payload.get("reasoning", ""),
        )
        session.votes.append(vote)

        if len(session.votes) >= session.min_voters:
            session.event.set()

    def _tally(self, session: _VoteSession) -> ConsensusResult:
        """Confidence-weighted majority vote."""
        if not session.votes:
            return ConsensusResult(
                vote_id=session.vote_id,
                winning_choice="",
                vote_count=0,
                total_votes=0,
                confidence=0.0,
                unanimous=False,
                votes=[],
            )

        # Sum confidence per choice
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for vote in session.votes:
            scores[vote.choice] = scores.get(vote.choice, 0.0) + vote.confidence
            counts[vote.choice] = counts.get(vote.choice, 0) + 1

        winning_choice = max(scores, key=lambda c: scores[c])
        total_confidence = sum(scores.values())
        winning_confidence = (
            scores[winning_choice] / total_confidence if total_confidence > 0 else 0.0
        )

        unique_choices = {v.choice for v in session.votes}
        unanimous = len(unique_choices) == 1

        return ConsensusResult(
            vote_id=session.vote_id,
            winning_choice=winning_choice,
            vote_count=counts[winning_choice],
            total_votes=len(session.votes),
            confidence=winning_confidence,
            unanimous=unanimous,
            votes=list(session.votes),
        )
