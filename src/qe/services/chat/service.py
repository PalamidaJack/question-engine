"""Chat service: direct-conversation agent with cognitive augmentation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm
from dotenv import load_dotenv

from qe.models.envelope import Envelope
from qe.runtime.budget import BudgetTracker
from qe.services.chat.events import (
    CompleteEvent,
    ErrorEvent,
    LLMCompleteEvent,
    ToolCompleteEvent,
)
from qe.services.chat.schemas import ChatIntent, ChatResponsePayload
from qe.services.inquiry.schemas import InquiryConfig
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

_MAX_HISTORY = 50
_HISTORY_TRIM_TO = 30
_MAX_CONTEXT_MESSAGES = 16
_COMPACTION_THRESHOLD = 24

_CHAT_INQUIRY_CONFIG = InquiryConfig(
    max_iterations=5,
    confidence_threshold=0.6,
    questions_per_iteration=2,
    inquiry_timeout_seconds=120.0,
)

# ── Chat tool schemas ───────────────────────────────────────────────────────

_CHAT_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_beliefs",
            "description": (
                "Search the knowledge base (belief ledger) for claims matching a query. "
                "Use this when the user asks about what is known, or you need to look up facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant claims.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List all entities in the knowledge base. Use this when the user asks "
                "to see entities, or you need to know what subjects are tracked."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_details",
            "description": (
                "Get details about a specific entity, including its claims. "
                "Use this when the user asks about a particular entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity name to look up.",
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_observation",
            "description": (
                "Submit new information to the knowledge pipeline for processing. "
                "The system will extract claims, validate them, and add them to the belief ledger. "
                "Use this when the user provides new information or observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The observation text to submit.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retract_claim",
            "description": (
                "Retract (soft-delete) a claim from the belief ledger by its ID. "
                "Use this when the user asks to retract or remove a specific claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "The claim ID to retract (e.g. 'clm_abc123').",
                    },
                },
                "required": ["claim_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_budget_status",
            "description": (
                "Get the current budget status including spend, limit, and remaining percentage. "
                "Use this when the user asks about costs or budget."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": (
                "Activate the cognitive research engine to deeply investigate a question. "
                "This runs a multi-phase inquiry loop with hypothesis generation, "
                "dialectic challenges, and evidence synthesis. Use this for complex questions "
                "that need thorough investigation beyond a simple belief lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question to investigate.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swarm_research",
            "description": (
                "Deploy a swarm of cognitive agents to research a question in parallel. "
                "Each agent investigates independently, then results are merged. "
                "Optionally runs a competitive tournament for cross-examination. "
                "Use this for complex questions that benefit from multiple perspectives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question for the swarm to investigate.",
                    },
                    "num_agents": {
                        "type": "integer",
                        "description": "Number of agents to spawn (2-5). Default 3.",
                        "minimum": 2,
                        "maximum": 5,
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_and_execute",
            "description": (
                "Decompose a complex goal into a DAG of subtasks, dispatch them to agents, "
                "and synthesize the results. Use this for goals that require multiple steps "
                "or coordinated investigation across different aspects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The goal to decompose and execute.",
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reason_about",
            "description": (
                "Apply epistemic reasoning and dialectic analysis to a claim. "
                "Assesses uncertainty, detects surprising elements, surfaces hidden assumptions, "
                "generates counterarguments, and rotates perspectives. "
                "Use this when the user wants critical analysis of a claim or finding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim or finding to reason about.",
                    },
                },
                "required": ["claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystallize_insights",
            "description": (
                "Extract and crystallize novel insights from a finding. "
                "Evaluates novelty, extracts causal mechanisms, scores actionability, "
                "and finds cross-domain connections. Returns structured insight or 'not novel'. "
                "Use this to distill research findings into actionable knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finding": {
                        "type": "string",
                        "description": "The finding or result to crystallize.",
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "The domain context (e.g. 'science', 'business'). "
                            "Default 'general'."
                        ),
                    },
                },
                "required": ["finding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_knowledge",
            "description": (
                "Trigger background knowledge consolidation: scans recent episodes, "
                "detects patterns, promotes beliefs, reviews hypotheses, "
                "and resolves contradictions. "
                "Use this when the user wants to consolidate and organize accumulated knowledge."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_agent",
            "description": (
                "Delegate a subtask to an external A2A-compatible agent."
                " Provide agent_url and task_description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_url": {"type": "string"},
                    "task_description": {"type": "string"},
                },
                "required": ["agent_url", "task_description"],
            },
        },
    },
]


# ── Artifact store for large tool results (Phase 2) ───────────────────────

_ARTIFACT_THRESHOLD = 800  # chars


class _ArtifactStore:
    """Per-session store for large tool results to keep context lean."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def store(self, content: str, tool_name: str) -> str:
        """Store content if over threshold, return summary + handle."""
        if len(content) <= _ARTIFACT_THRESHOLD:
            return content
        handle = f"artifact_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        self._store[handle] = content
        summary = content[:300] + "..."
        return (
            f"[Result stored as {handle} ({len(content)} chars)]\n"
            f"Summary: {summary}\n"
            f"Use load_artifact(handle=\"{handle}\") to retrieve the full content."
        )

    def load(self, handle: str) -> str | None:
        return self._store.get(handle)


class ChatSession:
    """Per-session conversation state."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: list[dict[str, str]] = []
        self.context_summary: str | None = None
        self.created_at = datetime.now(UTC)
        self.last_active = datetime.now(UTC)

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self.last_active = datetime.now(UTC)
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_HISTORY_TRIM_TO:]


class ChatService:
    """Direct-conversation agent with cognitive augmentation.

    The user talks to the LLM agent directly. The agent decides what
    to do — calling tools for knowledge management, observation
    submission, and deep research when needed.
    """

    def __init__(
        self,
        substrate: Substrate,
        bus: Any,
        budget_tracker: BudgetTracker | None = None,
        model: str = "gpt-4o-mini",
        fast_model: str | None = None,
        inquiry_engine: Any | None = None,
        tool_registry: Any | None = None,
        tool_gate: Any | None = None,
        episodic_memory: Any | None = None,
        cognitive_pool: Any | None = None,
        competitive_arena: Any | None = None,
        planner: Any | None = None,
        dispatcher: Any | None = None,
        goal_store: Any | None = None,
        epistemic_reasoner: Any | None = None,
        dialectic_engine: Any | None = None,
        insight_crystallizer: Any | None = None,
        knowledge_loop: Any | None = None,
        procedural_memory: Any | None = None,
        access_mode: str = "balanced",
        guardrails: Any | None = None,
        sanitizer: Any | None = None,
        router: Any | None = None,
        recovery: Any | None = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus
        self.budget_tracker = budget_tracker
        self.model = model
        self._fast_model = fast_model
        self._inquiry_engine = inquiry_engine
        self._tool_registry = tool_registry
        self._tool_gate = tool_gate
        self._episodic_memory = episodic_memory
        self._cognitive_pool = cognitive_pool
        self._competitive_arena = competitive_arena
        self._planner = planner
        self._dispatcher = dispatcher
        self._goal_store = goal_store
        self._epistemic_reasoner = epistemic_reasoner
        self._dialectic_engine = dialectic_engine
        self._insight_crystallizer = insight_crystallizer
        self._knowledge_loop = knowledge_loop
        self._procedural_memory = procedural_memory
        self._guardrails = guardrails
        self._sanitizer = sanitizer
        self._router = router
        self._recovery = recovery
        self._sessions: dict[str, ChatSession] = {}
        self._access_mode = "balanced"
        self.set_access_mode(access_mode)
        # Phase 2: Stable prompt prefix cache
        self._stable_system_prefix: str | None = None
        # Phase 2: Artifact store per service (shared across sessions)
        self._artifact_store: _ArtifactStore = _ArtifactStore()
        self._last_trace: Any = None

    _MAX_SESSIONS = 1000

    # ── Session management (unchanged) ──────────────────────────────────

    def get_or_create_session(self, session_id: str | None = None) -> ChatSession:
        if len(self._sessions) > self._MAX_SESSIONS // 2:
            self.cleanup_stale_sessions()
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        if len(self._sessions) >= self._MAX_SESSIONS:
            oldest_sid = min(
                self._sessions,
                key=lambda s: self._sessions[s].last_active,
            )
            del self._sessions[oldest_sid]
        sid = session_id or str(uuid.uuid4())
        session = ChatSession(sid)
        self._sessions[sid] = session
        return session

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count removed."""
        now = datetime.now(UTC)
        stale = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_active).total_seconds() > max_age_hours * 3600
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def set_access_mode(self, mode: str) -> None:
        """Set chat tool access mode."""
        if mode not in {"strict", "balanced", "full"}:
            mode = "balanced"
        self._access_mode = mode

    def _tool_capabilities_for_mode(self) -> set[str]:
        """Resolve tool capabilities from current access mode."""
        base = {"chat", "web_search", "web_fetch", "mcp"}
        if self._access_mode == "strict":
            return base
        if self._access_mode == "balanced":
            return base | {"file_read", "file_write", "browser_control"}
        return base | {"file_read", "file_write", "code_execute", "browser_control"}

    # ── Main entry point ────────────────────────────────────────────────

    async def handle_message(
        self, session_id: str, user_message: str,
        progress_queue: asyncio.Queue[dict] | None = None,
        interjection_queue: asyncio.Queue[str] | None = None,
    ) -> ChatResponsePayload:
        """Main entry point: receive user message, return structured response."""
        _handle_start = time.monotonic()
        session = self.get_or_create_session(session_id)
        message_id = str(uuid.uuid4())

        # Phase 1: Input sanitization at boundary
        if self._sanitizer:
            try:
                san_result = self._sanitizer.sanitize(user_message)
                if san_result.risk_score >= 0.9:
                    return ChatResponsePayload(
                        message_id=message_id,
                        reply_text=(
                            "Your message was blocked by input sanitization. "
                            "Please rephrase your request."
                        ),
                        intent=ChatIntent.CONVERSATION,
                        error="input_rejected",
                    )
                if san_result.risk_score >= 0.7:
                    user_message = self._sanitizer.wrap_untrusted(user_message)
            except Exception:
                log.debug("Sanitizer failed, proceeding", exc_info=True)

        session.add_user_message(user_message)

        if self.budget_tracker and self.budget_tracker.remaining_pct() <= 0:
            return ChatResponsePayload(
                message_id=message_id,
                reply_text="Budget exhausted. Unable to process your request.",
                intent=ChatIntent.CONVERSATION,
                error="budget_exhausted",
            )

        # Phase 1: Guardrails input check
        if self._guardrails:
            try:
                guardrail_results = await self._guardrails.run_input(
                    user_message, {"request_id": message_id, "origin": "chat"},
                )
                for gr in guardrail_results:
                    if not gr.passed and gr.severity == "block":
                        return ChatResponsePayload(
                            message_id=message_id,
                            reply_text=(
                                f"Message blocked by safety guardrail: {gr.message}"
                            ),
                            intent=ChatIntent.CONVERSATION,
                            error="guardrail_blocked",
                        )
            except Exception:
                log.debug("Guardrails input check failed", exc_info=True)

        try:
            messages = await self._build_messages(session)
            tool_schemas = self._get_chat_tools()
            reply_text, tool_audit, cumulative_cost = await asyncio.wait_for(
                self._chat_tool_loop(
                    messages, tool_schemas, max_iterations=20,
                    progress_queue=progress_queue,
                    interjection_queue=interjection_queue,
                ),
                timeout=300.0,
            )

            # Phase 1: Guardrails output check
            if self._guardrails:
                try:
                    output_results = await self._guardrails.run_output(
                        reply_text, {"request_id": message_id, "origin": "chat"},
                    )
                    for gr in output_results:
                        if not gr.passed and gr.severity == "block":
                            reply_text = (
                                "My response was filtered by a safety guardrail. "
                                "Please try a different question."
                            )
                            break
                except Exception:
                    log.debug("Guardrails output check failed", exc_info=True)

            response = self._build_response(
                message_id, reply_text, tool_audit
            )
            await self._extract_session_patterns(tool_audit, cumulative_cost)
        except TimeoutError:
            log.warning(
                "Agent loop timed out for session %s", session_id,
            )
            response = ChatResponsePayload(
                message_id=message_id,
                reply_text=(
                    "Sorry, my response took too long. "
                    "This can happen with complex research queries. "
                    "Please try again with a more specific question."
                ),
                intent=ChatIntent.CONVERSATION,
                error="timeout",
            )
        except Exception as e:
            log.exception("Agent loop error for session %s", session_id)
            response = ChatResponsePayload(
                message_id=message_id,
                reply_text=f"I encountered an error: {e}",
                intent=ChatIntent.CONVERSATION,
                error=str(e),
            )

        session.add_assistant_message(response.reply_text)

        # Phase 4: Record chat response latency
        try:
            from qe.runtime.metrics import get_metrics
            _elapsed_ms = (time.monotonic() - _handle_start) * 1000
            get_metrics().histogram("chat_response_latency_ms").observe(_elapsed_ms)
        except Exception:
            pass

        return response

    # ── System prompt & context assembly ────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Static identity and behavioral instructions."""
        local_tool_note = ""
        if self._access_mode == "strict":
            local_tool_note = (
                "- Local filesystem and code execution tools are disabled in this session "
                "(strict security mode).\n"
            )
        elif self._access_mode == "balanced":
            local_tool_note = (
                "- You can read/write local files in a constrained workspace sandbox. "
                "Raw code execution tools are disabled in this mode.\n"
            )
        else:
            local_tool_note = (
                "- You can read/write local files and execute code when needed for user tasks.\n"
            )
        return (
            "You are the Question Engine assistant — an AI agent with a "
            "full cognitive architecture: belief ledger, episodic memory, "
            "multi-phase research engine, agent swarms, goal orchestration, "
            "epistemic reasoning, dialectic analysis, insight crystallization, "
            "and knowledge consolidation.\n\n"
            f"## About you\n"
            f"- You are powered by model: {self.model}\n"
            "- You are NOT a generic chatbot — you are a knowledge "
            "management and cognitive research agent.\n"
            "- When asked what you are or what model you run, "
            "identify yourself as the Question Engine assistant "
            f"running on {self.model}.\n\n"
            "## What you can do\n"
            "- Answer questions using the knowledge base "
            "(query_beliefs)\n"
            "- Accept USER-PROVIDED observations and submit them "
            "to the research pipeline (submit_observation)\n"
            "- List and inspect entities and claims "
            "(list_entities, get_entity_details)\n"
            "- Retract claims that are wrong (retract_claim)\n"
            "- Check system budget (get_budget_status)\n"
            "- Search the web for current information "
            "(web_search, web_fetch)\n"
            f"{local_tool_note}"
            "- Perform deep multi-phase research on complex "
            "questions (deep_research) — 5-iteration cognitive loop "
            "with hypothesis testing and evidence synthesis\n"
            "- Deploy a swarm of parallel cognitive agents to research "
            "a question from multiple perspectives, with optional "
            "competitive tournament scoring (swarm_research)\n"
            "- Decompose complex goals into subtask DAGs, dispatch "
            "to agents, and synthesize results (plan_and_execute)\n"
            "- Apply epistemic reasoning: uncertainty assessment, "
            "surprise detection, dialectic challenges, assumption "
            "surfacing, counterarguments, and perspective rotation "
            "(reason_about)\n"
            "- Crystallize research findings into novel insights "
            "with mechanism extraction, actionability scoring, and "
            "cross-domain connections (crystallize_insights)\n"
            "- Trigger knowledge consolidation: pattern detection, "
            "belief promotion, hypothesis review, contradiction "
            "resolution (consolidate_knowledge)\n\n"
            "## How to behave\n"
            "- Be concise and helpful.\n"
            "- NEVER lie or exaggerate about what you did. If you "
            "fetched one page, say you fetched one page. If you "
            "only read a summary, say so. Never claim to have "
            "read or processed more than you actually did.\n"
            "- When the user asks you to do something, DO IT. "
            "Do not refuse, warn, or suggest alternatives unless "
            "it is truly impossible. If the user says 'read all "
            "of them', read as many as you can.\n"
            "- Do NOT proactively warn about budget, costs, or "
            "resource usage. The user manages their own budget. "
            "Only mention budget if the user explicitly asks.\n"
            "- ONLY use submit_observation for information the "
            "USER explicitly provides (facts, news, things they "
            "tell you). NEVER submit your own analyses, "
            "summaries, or responses as observations.\n"
            "- When the user asks what is known about a topic, "
            "use query_beliefs first. If the results are "
            "insufficient, use deep_research.\n"
            "- When the user asks to retract, list, or inspect "
            "claims/entities, use the appropriate tool.\n"
            "- For complex, multi-faceted questions, prefer "
            "swarm_research or plan_and_execute over deep_research.\n"
            "- When the user asks for critical analysis, use "
            "reason_about to apply epistemic and dialectic reasoning.\n"
            "- For simple greetings and conversation, respond "
            "directly without tools.\n"
            "- Always suggest 2-3 short follow-up prompts at the "
            "end of your response.\n"
            "- Format follow-up suggestions on separate lines "
            "prefixed with '> '.\n"
        )

    async def _build_knowledge_context(self) -> str:
        """Query substrate for current knowledge state."""
        parts: list[str] = []
        try:
            claims = await self.substrate.get_claims()
            entities = await self.substrate.entity_resolver.list_entities()
            parts.append(
                f"Knowledge base: {len(claims)} claims, {len(entities)} entities."
            )

            if claims:
                sorted_claims = sorted(
                    claims,
                    key=lambda c: getattr(c, "confidence", 0.0),
                    reverse=True,
                )
                top = sorted_claims[:10]
                lines = []
                for c in top:
                    subj = getattr(c, "subject_entity_id", "?")
                    pred = getattr(c, "predicate", "?")
                    obj = getattr(c, "object_value", "?")
                    conf = getattr(c, "confidence", 0.0)
                    lines.append(f"  - [{conf:.0%}] {subj} {pred} {obj}")
                parts.append("Top claims:\n" + "\n".join(lines))
        except Exception:
            log.debug("Failed to build knowledge context", exc_info=True)

        if self.budget_tracker:
            try:
                spent = self.budget_tracker.total_spend()
                limit = self.budget_tracker.monthly_limit_usd
                remaining = self.budget_tracker.remaining_pct()
                parts.append(
                    f"Budget: ${spent:.4f} spent of ${limit:.2f} "
                    f"({remaining * 100:.1f}% remaining)."
                )
            except Exception:
                pass

        return "\n".join(parts)

    async def _build_memory_context(self, session: ChatSession) -> str:
        """Query episodic memory for recent relevant episodes."""
        if not self._episodic_memory:
            return ""
        try:
            last_user = ""
            for msg in reversed(session.history):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if not last_user:
                return ""
            episodes = await self._episodic_memory.recall(
                last_user, top_k=5, time_window_hours=72.0
            )
            if not episodes:
                return ""
            lines = ["Recent memory:"]
            for ep in episodes:
                summary = getattr(ep, "summary", "") or str(
                    getattr(ep, "content", {})
                )
                lines.append(f"  - [{ep.episode_type}] {summary[:120]}")
            return "\n".join(lines)
        except Exception:
            log.debug("Failed to build memory context", exc_info=True)
            return ""

    async def _compact_history(self, session: ChatSession) -> None:
        """Summarize older messages into a context summary, keep recent ones."""
        if len(session.history) <= _COMPACTION_THRESHOLD:
            return
        compact_model = self._fast_model or self.model
        older = session.history[:-_MAX_CONTEXT_MESSAGES]
        recent = session.history[-_MAX_CONTEXT_MESSAGES:]
        try:
            prompt = (
                "Summarize the following conversation history concisely. "
                "Preserve: the user's original request, key decisions made, "
                "tools used, entities discussed, and current state.\n\n"
            )
            for msg in older:
                prompt += f"[{msg['role']}]: {msg['content'][:200]}\n"
            resp = await asyncio.wait_for(
                litellm.acompletion(
                    model=compact_model,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30.0,
            )
            summary = getattr(resp.choices[0].message, "content", "") or ""
            if summary:
                session.context_summary = summary
                session.history = recent
                log.info(
                    "chat.compacted session=%s older=%d summary_len=%d",
                    session.session_id, len(older), len(summary),
                )
                return
        except Exception:
            log.debug("Context compaction failed, falling back to FIFO trim", exc_info=True)
        # Fallback: plain FIFO trim
        session.history = recent

    async def _build_messages(self, session: ChatSession) -> list[dict]:
        """Assemble full message list for the LLM call."""
        await self._compact_history(session)

        from qe.runtime.feature_flags import get_flag_store
        flags = get_flag_store()

        if flags.is_enabled("stable_prompt_prefix"):
            # Phase 2.1: Stable prefix + trailing dynamic context
            if self._stable_system_prefix is None:
                self._stable_system_prefix = self._build_system_prompt()
            messages: list[dict] = [
                {"role": "system", "content": self._stable_system_prefix},
            ]
            messages.extend(session.history[-_MAX_CONTEXT_MESSAGES:])

            # Dynamic context as trailing system message
            dynamic_parts: list[str] = []
            if session.context_summary:
                dynamic_parts.append(
                    f"## Conversation context (summarized)\n{session.context_summary}"
                )
            knowledge_ctx = await self._build_knowledge_context()
            if knowledge_ctx:
                dynamic_parts.append(f"## Current state\n{knowledge_ctx}")
            memory_ctx = await self._build_memory_context(session)
            if memory_ctx:
                dynamic_parts.append(f"## {memory_ctx}")
            # Phase 2.6: Proactive memory recall
            if flags.is_enabled("proactive_recall"):
                proactive = await self._proactive_recall(session)
                if proactive:
                    dynamic_parts.append(f"## Suggested approaches\n{proactive}")
            if dynamic_parts:
                messages.append(
                    {"role": "system", "content": "\n\n".join(dynamic_parts)}
                )
            return messages

        # Legacy: single system message with everything
        system_parts = [self._build_system_prompt()]

        if session.context_summary:
            system_parts.append(
                f"\n## Conversation context (summarized)\n{session.context_summary}"
            )

        knowledge_ctx = await self._build_knowledge_context()
        if knowledge_ctx:
            system_parts.append(f"\n## Current state\n{knowledge_ctx}")

        memory_ctx = await self._build_memory_context(session)
        if memory_ctx:
            system_parts.append(f"\n## {memory_ctx}")

        messages = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]
        messages.extend(session.history[-_MAX_CONTEXT_MESSAGES:])
        return messages

    async def _proactive_recall(self, session: ChatSession) -> str:
        """Phase 2.6: Query procedural memory for relevant tool sequences."""
        if not self._procedural_memory:
            return ""
        try:
            last_user = ""
            for msg in reversed(session.history):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if not last_user:
                return ""
            suggestions = await self._procedural_memory.suggest_sequence(
                domain="chat", context=last_user, top_k=2,
            )
            if not suggestions:
                return ""
            lines = []
            for s in suggestions:
                tools = getattr(s, "tool_names", []) or []
                if tools:
                    lines.append(f"- Previously effective: {' → '.join(tools)}")
            return "\n".join(lines)
        except Exception:
            log.debug("Proactive recall failed", exc_info=True)
            return ""

    # ── Tool definitions ────────────────────────────────────────────────

    def _get_chat_tools(self) -> list[dict]:
        """Return combined tool schemas: chat-specific + selected registry tools."""
        from qe.runtime.feature_flags import get_flag_store
        flags = get_flag_store()

        tools = list(_CHAT_TOOL_SCHEMAS)

        # Phase 2.4: Add load_artifact tool if flag enabled
        if flags.is_enabled("artifact_handles"):
            tools.append({
                "type": "function",
                "function": {
                    "name": "load_artifact",
                    "description": (
                        "Load the full content of a previously stored artifact. "
                        "Use when a tool result was truncated and stored as an artifact handle."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "handle": {
                                "type": "string",
                                "description": "The artifact handle (e.g. artifact_abc123).",
                            },
                        },
                        "required": ["handle"],
                    },
                },
            })

        if self._tool_registry:
            try:
                if flags.is_enabled("tool_masking"):
                    # Phase 2.2: Return ALL tools regardless of mode — ToolGate
                    # blocks at execution time. Keeps tool schemas stable for KV-cache.
                    registry_tools = self._tool_registry.get_tool_schemas(
                        capabilities={"chat", "web_search", "web_fetch", "mcp",
                                       "file_read", "file_write", "code_execute",
                                       "browser_control"},
                        mode="relevant",
                    )
                else:
                    registry_tools = self._tool_registry.get_tool_schemas(
                        capabilities=self._tool_capabilities_for_mode(),
                        mode="relevant",
                    )
                tools.extend(registry_tools)
            except Exception:
                log.debug("Failed to get registry tool schemas", exc_info=True)

        return tools

    # ── Tool handlers ───────────────────────────────────────────────────

    async def _tool_query_beliefs(self, query: str) -> str:
        """Search the belief ledger for matching claims."""
        results = await self.substrate.hybrid_search(query)
        if not results:
            return "No matching claims found."
        lines = []
        for r in results[:10]:
            claim_id = getattr(r, "claim_id", "?")
            subj = getattr(r, "subject_entity_id", "?")
            pred = getattr(r, "predicate", "?")
            obj = getattr(r, "object_value", "?")
            conf = getattr(r, "confidence", 0.0)
            lines.append(
                f"[{claim_id}] ({conf:.0%}) {subj} {pred} {obj}"
            )
        return f"Found {len(results)} claims. Top results:\n" + "\n".join(lines)

    async def _tool_list_entities(self) -> str:
        """List all entities in the knowledge base."""
        entities = await self.substrate.entity_resolver.list_entities()
        if not entities:
            return "No entities in the knowledge base yet."
        lines = []
        for e in entities[:30]:
            name = e.get("canonical_name", str(e))
            lines.append(f"  - {name}")
        return f"Found {len(entities)} entities:\n" + "\n".join(lines)

    async def _tool_get_entity_details(self, entity: str) -> str:
        """Resolve an entity and return its claims."""
        canonical = await self.substrate.entity_resolver.resolve(entity)
        claims = await self.substrate.get_claims(subject_entity_id=canonical)
        if not claims:
            return f"Entity '{canonical}' found but has no claims."
        lines = [f"Entity: {canonical} ({len(claims)} claims)"]
        for c in claims[:15]:
            claim_id = getattr(c, "claim_id", "?")
            pred = getattr(c, "predicate", "?")
            obj = getattr(c, "object_value", "?")
            conf = getattr(c, "confidence", 0.0)
            lines.append(f"  [{claim_id}] ({conf:.0%}) {pred} {obj}")
        return "\n".join(lines)

    async def _tool_submit_observation(self, text: str) -> str:
        """Publish an observation to the research pipeline."""
        envelope = Envelope(
            topic="observations.structured",
            source_service_id="chat",
            payload={"text": text},
        )
        self.bus.publish(envelope)
        return (
            f"Observation submitted. Tracking ID: {envelope.envelope_id}\n"
            f"Pipeline: observation → researcher → validator → committed"
        )

    async def _tool_retract_claim(self, claim_id: str) -> str:
        """Retract a claim from the belief ledger."""
        retracted = await self.substrate.retract_claim(claim_id)
        if retracted:
            return f"Claim '{claim_id}' has been retracted."
        return f"Claim '{claim_id}' not found."

    async def _tool_get_budget(self) -> str:
        """Return current budget status."""
        if not self.budget_tracker:
            return "Budget tracking is not active."
        spent = self.budget_tracker.total_spend()
        limit = self.budget_tracker.monthly_limit_usd
        remaining = self.budget_tracker.remaining_pct()
        return (
            f"Budget: ${spent:.4f} spent of ${limit:.2f} limit "
            f"({remaining * 100:.1f}% remaining)"
        )

    async def _tool_deep_research(self, question: str) -> str:
        """Run the InquiryEngine's multi-phase cognitive loop."""
        if not self._inquiry_engine:
            return "Deep research engine is not available."
        try:
            result = await asyncio.wait_for(
                self._inquiry_engine.run_inquiry(
                    goal_id=f"chat_{uuid.uuid4().hex[:12]}",
                    goal_description=question,
                    config=_CHAT_INQUIRY_CONFIG,
                ),
                timeout=_CHAT_INQUIRY_CONFIG.inquiry_timeout_seconds,
            )
        except TimeoutError:
            return (
                "Deep research timed out after "
                f"{_CHAT_INQUIRY_CONFIG.inquiry_timeout_seconds:.0f}s. "
                "The question may be too broad or the research "
                "engine encountered issues. Try a more specific question."
            )
        summary = result.findings_summary or (
            "Investigation completed but no definitive findings."
        )
        parts = [summary]
        if result.insights:
            parts.append("\nKey insights:")
            for ins in result.insights[:5]:
                headline = ins.get("headline", "")
                if headline:
                    parts.append(f"  - {headline}")
        parts.append(
            f"\n(Iterations: {result.iterations_completed}, "
            f"Questions: {result.total_questions_generated}, "
            f"Status: {result.status})"
        )
        return "\n".join(parts)

    async def _tool_swarm_research(
        self, question: str, num_agents: int = 3,
    ) -> str:
        """Deploy a swarm of cognitive agents for parallel research."""
        if not self._cognitive_pool:
            return "Swarm research is not available (cognitive pool not initialized)."
        num_agents = max(2, min(num_agents, 5))
        goal_id = f"swarm_{uuid.uuid4().hex[:12]}"
        agents = []
        try:
            for _ in range(num_agents):
                agent = await self._cognitive_pool.spawn_agent()
                agents.append(agent)
            agent_ids = [a.agent_id for a in agents]
            results = await asyncio.wait_for(
                self._cognitive_pool.run_parallel_inquiry(
                    goal_id=goal_id,
                    goal_description=question,
                    agent_ids=agent_ids,
                ),
                timeout=180.0,
            )
            if self._competitive_arena and len(results) >= 2:
                arena_result = await self._competitive_arena.run_tournament(
                    goal_id=goal_id,
                    goal_description=question,
                    results=results,
                    agent_ids=agent_ids,
                )
                parts = [
                    f"Swarm research complete ({num_agents} agents, tournament scored).",
                    f"\nWinner: {arena_result.winner_agent_id}"
                    if hasattr(arena_result, "winner_agent_id")
                    else "",
                ]
                if hasattr(arena_result, "summary") and arena_result.summary:
                    parts.append(f"\nTournament summary:\n{arena_result.summary}")
            else:
                merged = await self._cognitive_pool.merge_results(results)
                parts = [
                    f"Swarm research complete ({num_agents} agents, results merged).",
                ]
                summary = getattr(merged, "findings_summary", "") or ""
                if summary:
                    parts.append(f"\n{summary}")
                if getattr(merged, "insights", None):
                    parts.append("\nKey insights:")
                    for ins in merged.insights[:5]:
                        if isinstance(ins, dict):
                            headline = ins.get("headline", "")
                        else:
                            headline = getattr(ins, "headline", "")
                        if headline:
                            parts.append(f"  - {headline}")
        except TimeoutError:
            parts = ["Swarm research timed out after 180s. Try a more focused question."]
        except Exception as e:
            log.exception("Swarm research failed")
            parts = [f"Swarm research encountered an error: {e}"]
        finally:
            for a in agents:
                try:
                    await self._cognitive_pool.retire_agent(a.agent_id)
                except Exception:
                    pass
        return "\n".join(p for p in parts if p)

    async def _tool_plan_and_execute(
        self,
        goal: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> str:
        """Decompose a goal into subtasks and execute them."""
        if not self._planner or not self._dispatcher:
            return (
                "Plan-and-execute is not available "
                "(planner/dispatcher not initialized)."
            )

        _PE_TIMEOUT = 600.0
        _start = time.monotonic()

        try:
            state = await self._planner.decompose(goal)
            goal_id = state.goal_id
            subtask_count = (
                len(state.decomposition.subtasks)
                if state.decomposition else 0
            )

            log.info(
                "plan_and_execute.start goal_id=%s subtasks=%d",
                goal_id, subtask_count,
            )

            # Emit decomposition event
            if progress_queue is not None:
                try:
                    progress_queue.put_nowait({
                        "type": "chat_progress",
                        "phase": "plan_decomposed",
                        "goal_id": goal_id,
                        "subtask_count": subtask_count,
                        "strategy": (
                            state.decomposition.strategy[:200]
                            if state.decomposition else ""
                        ),
                    })
                except Exception:
                    pass

            done = asyncio.Event()
            result_holder: dict = {}
            _completed_count = [0]

            async def on_synthesized(
                envelope: Envelope, _gid=goal_id,
            ) -> None:
                if envelope.payload.get("goal_id") == _gid:
                    result_holder["data"] = envelope.payload
                    done.set()

            async def on_subtask_done(
                envelope: Envelope,
                _gid=goal_id,
                _total=subtask_count,
                _pq=progress_queue,
                _count=_completed_count,
            ) -> None:
                if envelope.payload.get("goal_id") != _gid:
                    return
                _count[0] += 1
                sid = envelope.payload.get("subtask_id", "?")
                log.info(
                    "plan_and_execute.subtask_done goal_id=%s "
                    "subtask=%s progress=%d/%d",
                    _gid, sid, _count[0], _total,
                )
                if _pq is not None:
                    try:
                        _pq.put_nowait({
                            "type": "chat_progress",
                            "phase": "plan_subtask_done",
                            "goal_id": _gid,
                            "subtask_id": sid,
                            "completed": _count[0],
                            "total": _total,
                        })
                    except Exception:
                        pass

            self.bus.subscribe("goals.synthesized", on_synthesized)
            self.bus.subscribe("tasks.completed", on_subtask_done)
            self.bus.subscribe("tasks.failed", on_subtask_done)
            try:
                await self._dispatcher.submit_goal(state)
                await asyncio.wait_for(
                    done.wait(), timeout=_PE_TIMEOUT,
                )
            finally:
                self.bus.unsubscribe(
                    "goals.synthesized", on_synthesized,
                )
                self.bus.unsubscribe(
                    "tasks.completed", on_subtask_done,
                )
                self.bus.unsubscribe(
                    "tasks.failed", on_subtask_done,
                )

            elapsed = time.monotonic() - _start
            data = result_holder.get("data", {})
            parts = [f"Goal '{goal}' completed."]
            if data.get("synthesis"):
                parts.append(f"\n{data['synthesis']}")
            parts.append(
                f"\n(Goal ID: {goal_id}, Subtasks: "
                f"{subtask_count}, Time: {elapsed:.1f}s)"
            )
            return "\n".join(parts)

        except TimeoutError:
            elapsed = time.monotonic() - _start
            completed = _completed_count[0] if '_completed_count' in dir() else 0
            return (
                f"Goal execution timed out after {int(elapsed)}s. "
                f"Completed {completed}/{subtask_count} subtasks. "
                f"The goal '{goal}' may still be processing "
                f"in the background."
            )
        except Exception as e:
            log.exception("Plan-and-execute failed")
            return f"Plan-and-execute error: {e}"

    async def _tool_reason_about(self, claim: str) -> str:
        """Apply epistemic reasoning and dialectic analysis to a claim."""
        if not self._epistemic_reasoner or not self._dialectic_engine:
            return "Epistemic reasoning is not available."
        goal_id = f"reason_{uuid.uuid4().hex[:12]}"
        parts = []
        try:
            uncertainty = await self._epistemic_reasoner.assess_uncertainty(
                goal_id=goal_id, finding=claim,
            )
            parts.append(f"Confidence: {uncertainty.confidence_level}")
            parts.append(f"Evidence quality: {uncertainty.evidence_quality}")
            if uncertainty.potential_biases:
                parts.append("Potential biases: " + "; ".join(uncertainty.potential_biases))
            if uncertainty.information_gaps:
                parts.append("Information gaps: " + "; ".join(uncertainty.information_gaps))
            if uncertainty.could_be_wrong_because:
                parts.append(
                    "Could be wrong because: "
                    + "; ".join(uncertainty.could_be_wrong_because)
                )
        except Exception:
            log.debug("Uncertainty assessment failed", exc_info=True)

        try:
            surprise = await self._epistemic_reasoner.detect_surprise(
                goal_id=goal_id, entity_id="", new_finding=claim,
            )
            if surprise:
                parts.append(
                    f"\nSurprise detected (magnitude {surprise.surprise_magnitude:.1f}): "
                    f"{surprise.finding}"
                )
                if surprise.implications:
                    parts.append("Implications: " + "; ".join(surprise.implications))
        except Exception:
            log.debug("Surprise detection failed", exc_info=True)

        try:
            dialectic = await self._dialectic_engine.full_dialectic(
                goal_id=goal_id, conclusion=claim,
            )
            parts.append(
                f"\nDialectic analysis "
                f"(revised confidence: {dialectic.revised_confidence:.0%}):"
            )
            if dialectic.counterarguments:
                parts.append("Counterarguments:")
                for ca in dialectic.counterarguments[:3]:
                    arg = getattr(ca, "argument", str(ca))
                    parts.append(f"  - {arg}")
            if dialectic.assumptions_challenged:
                parts.append("Assumptions challenged:")
                for ac in dialectic.assumptions_challenged[:3]:
                    assumption = getattr(ac, "assumption", str(ac))
                    parts.append(f"  - {assumption}")
            if dialectic.synthesis:
                parts.append(f"\nSynthesis: {dialectic.synthesis}")
        except Exception:
            log.debug("Dialectic analysis failed", exc_info=True)

        if not parts:
            return "Epistemic reasoning produced no results for this claim."
        return "\n".join(parts)

    async def _tool_crystallize_insights(
        self, finding: str, domain: str = "general",
    ) -> str:
        """Crystallize insights from a finding."""
        if not self._insight_crystallizer:
            return "Insight crystallization is not available."
        goal_id = f"crystal_{uuid.uuid4().hex[:12]}"
        try:
            insight = await self._insight_crystallizer.crystallize(
                goal_id=goal_id,
                finding=finding,
                domain=domain,
            )
        except Exception as e:
            log.exception("Insight crystallization failed")
            return f"Crystallization error: {e}"

        if insight is None:
            return "Finding did not pass the novelty gate — not novel enough to crystallize."

        parts = [f"Insight: {insight.headline}"]
        parts.append(f"Confidence: {insight.confidence:.0%}")
        if hasattr(insight, "mechanism") and insight.mechanism:
            mech = insight.mechanism
            explanation = getattr(mech, "explanation", str(mech))
            parts.append(f"Mechanism: {explanation}")
        if hasattr(insight, "novelty") and insight.novelty:
            nov = insight.novelty
            score = getattr(nov, "novelty_score", None)
            if score is not None:
                parts.append(f"Novelty score: {score:.2f}")
        parts.append(f"Actionability: {insight.actionability_score:.2f}")
        if insight.actionability_description:
            parts.append(f"  {insight.actionability_description}")
        if insight.cross_domain_connections:
            parts.append("Cross-domain connections:")
            for conn in insight.cross_domain_connections[:5]:
                parts.append(f"  - {conn}")
        return "\n".join(parts)

    async def _tool_consolidate_knowledge(self) -> str:
        """Trigger knowledge consolidation cycle."""
        if not self._knowledge_loop:
            return "Knowledge consolidation is not available."
        try:
            await self._knowledge_loop.trigger_consolidation()
            status = self._knowledge_loop.status()
            last = status.get("last_cycle_result")
            if last:
                parts = ["Knowledge consolidation completed."]
                parts.append(f"Episodes scanned: {last.get('episodes_scanned', 0)}")
                parts.append(f"Patterns detected: {last.get('patterns_detected', 0)}")
                parts.append(f"Beliefs promoted: {last.get('beliefs_promoted', 0)}")
                parts.append(f"Hypotheses reviewed: {last.get('hypotheses_reviewed', 0)}")
                parts.append(f"Contradictions found: {last.get('contradictions_found', 0)}")
                return "\n".join(parts)
            return "Knowledge consolidation triggered (no cycle results yet)."
        except Exception as e:
            log.exception("Knowledge consolidation failed")
            return f"Consolidation error: {e}"

    async def _tool_delegate_to_agent(self, agent_url: str, task_description: str) -> str:
        """Delegate a subtask to an external A2A-compatible agent using A2AClient."""
        try:
            from qe.runtime.a2a_client import A2AClient

            client = A2AClient(agent_url)
            resp = await client.send_task(description=task_description)
            task_id = resp.get("task_id") or resp.get("id") or ""
            return f"Delegated to {agent_url} as task {task_id}"
        except Exception as e:
            log.exception("delegate_to_agent_failed")
            return f"Delegate error: {e}"

    # ── Model selection ─────────────────────────────────────────────────

    def _classify_task(self, messages: list[dict]) -> str | None:
        """Phase 3.1: Keyword-based zero-latency task classification."""
        # Find last user message
        user_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "").lower()
                break
        if not user_text:
            return None

        _KEYWORDS: dict[str, list[str]] = {
            "extraction": ["extract", "list", "get", "show", "find", "fetch"],
            "summarization": ["summarize", "summary", "tldr", "brief", "overview"],
            "analysis": ["analyze", "compare", "evaluate", "assess"],
            "reasoning": ["reason", "explain", "why", "how", "think", "logic"],
            "critical": ["verify", "validate", "prove", "confirm", "check"],
            "creative": ["write", "create", "generate", "compose", "draft"],
            "code": ["code", "implement", "function", "debug", "program"],
        }
        best_hint = None
        best_score = 0
        for hint, keywords in _KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in user_text)
            if score > best_score:
                best_score = score
                best_hint = hint
        return best_hint if best_score > 0 else None

    def _select_model_for_iteration(
        self, iteration: int, has_tool_results: bool, max_iterations: int,
        task_hint: str | None = None,
    ) -> str:
        """Choose balanced or fast model based on iteration context."""
        from qe.runtime.feature_flags import get_flag_store
        flags = get_flag_store()

        # Phase 3.1: Task-aware routing via AutoRouter
        if flags.is_enabled("task_aware_routing") and self._router:
            try:
                from qe.models.envelope import Envelope as _Env
                env = _Env(
                    topic="chat.llm_select",
                    source_service_id="chat",
                    payload={
                        "task_hint": task_hint or "reasoning",
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "has_tool_results": has_tool_results,
                    },
                )
                return self._router.select(env)
            except Exception:
                log.debug("Task-aware routing failed, falling back", exc_info=True)

        if not self._fast_model:
            return self.model
        # First iteration (initial reasoning) → balanced
        if iteration == 0:
            return self.model
        # Near max iterations → balanced for best final answer
        if iteration >= max_iterations - 1:
            return self.model
        # Mid-chain after tool results → fast
        if has_tool_results:
            return self._fast_model
        return self.model

    async def _call_llm_with_recovery(
        self, kwargs: dict[str, Any], iteration: int, max_retries: int = 2,
    ) -> Any:
        """Phase 3.4: LLM call with retry and model escalation."""
        from qe.runtime.feature_flags import get_flag_store
        flags = get_flag_store()

        if not flags.is_enabled("chat_llm_recovery"):
            return await asyncio.wait_for(
                litellm.acompletion(**kwargs), timeout=60.0,
            )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    litellm.acompletion(**kwargs), timeout=60.0,
                )
                if self._router:
                    try:
                        self._router.record_success(kwargs["model"])
                    except Exception:
                        pass
                return result
            except TimeoutError as e:
                last_error = e
                if self._router:
                    try:
                        self._router.record_error(kwargs["model"])
                    except Exception:
                        pass
                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 4))
                    continue
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if self._router:
                    try:
                        self._router.record_error(kwargs["model"])
                    except Exception:
                        pass
                # Context length error → escalate model
                if "context" in err_str or "token" in err_str:
                    if self._fast_model and kwargs["model"] == self._fast_model:
                        kwargs["model"] = self.model
                        continue
                # Rate limit → retry with backoff
                if "rate" in err_str or "429" in err_str:
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 ** attempt, 4))
                        continue
                if attempt >= max_retries:
                    break
        raise last_error  # type: ignore[misc]

    # ── Agent loop ──────────────────────────────────────────────────────

    async def _chat_tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        max_iterations: int = 8,
        progress_queue: asyncio.Queue[dict] | None = None,
        interjection_queue: asyncio.Queue[str] | None = None,
    ) -> tuple[str, list[dict], float]:
        """Simple ReAct loop: call LLM, execute tools, repeat.

        Returns (reply_text, tool_audit_log, cumulative_cost_usd).
        Populates self._last_trace with a ChatTrace when complete.
        """
        from qe.services.chat.trace import (
            ChatTrace,
            LLMCallTrace,
            ToolCallTrace,
        )

        tool_audit: list[dict] = []
        working_messages = list(messages)

        _loop_start = time.monotonic()
        _cumulative_tokens: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        _cumulative_cost: float = 0.0
        _total_tool_calls = 0
        _trace = ChatTrace()
        _current_iteration = 0

        def _finalize_trace(outcome: str) -> None:
            _trace.total_iterations = _current_iteration + 1
            _trace.total_tool_calls = _total_tool_calls
            _trace.total_tokens = _cumulative_tokens["total"]
            _trace.total_cost_usd = _cumulative_cost
            _trace.total_duration_ms = (
                (time.monotonic() - _loop_start) * 1000
            )
            _trace.outcome = outcome
            self._last_trace = _trace

        def _emit(event: dict) -> None:
            if progress_queue is None:
                return
            phase = event.get("phase", "")
            iteration = event.get("iteration", 0)
            # Build typed event models for SSE consumers
            typed: dict = {}
            if phase == "llm_complete":
                typed = LLMCompleteEvent(
                    iteration=iteration,
                    model=event.get("model", ""),
                    call_tokens=event.get("call_tokens", {}),
                    call_cost_usd=event.get("call_cost_usd", 0.0),
                    has_tool_calls=event.get("has_tool_calls", False),
                ).model_dump()
            elif phase == "tool_complete":
                typed = ToolCompleteEvent(
                    iteration=iteration,
                    tool_name=event.get("tool_name", ""),
                    result_preview=event.get("result_preview", ""),
                    duration_ms=event.get("duration_ms", 0.0),
                ).model_dump()
            elif phase == "complete":
                typed = CompleteEvent(
                    iteration=iteration,
                    summary=event.get("summary", ""),
                ).model_dump()
            elif phase == "error":
                typed = ErrorEvent(
                    iteration=iteration,
                    message=event.get("message", ""),
                ).model_dump()
            # Merge typed fields into raw event (typed fields win)
            event.update(typed)
            event.setdefault("type", phase or "chat_progress")
            event["elapsed_ms"] = int((time.monotonic() - _loop_start) * 1000)
            event["cumulative_tokens"] = dict(_cumulative_tokens)
            event["cumulative_cost_usd"] = round(_cumulative_cost, 6)
            event["timestamp"] = datetime.now(UTC).isoformat()
            try:
                progress_queue.put_nowait(event)
            except Exception:
                pass

        from qe.runtime.feature_flags import get_flag_store
        _flags = get_flag_store()
        _task_hint = self._classify_task(working_messages)
        # Capture original user request for recitation (Phase 2.3)
        _original_request = ""
        for _msg in reversed(working_messages):
            if isinstance(_msg, dict) and _msg.get("role") == "user":
                _original_request = _msg.get("content", "")[:500]
                break

        for _iteration in range(max_iterations):
            _current_iteration = _iteration
            has_tool_results = _total_tool_calls > 0
            selected_model = self._select_model_for_iteration(
                _iteration, has_tool_results, max_iterations,
                task_hint=_task_hint,
            )
            kwargs: dict[str, Any] = {
                "model": selected_model,
                "messages": working_messages,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            _emit({
                "phase": "llm_start",
                "iteration": _iteration,
                "max_iterations": max_iterations,
                "model": selected_model,
            })

            try:
                response = await self._call_llm_with_recovery(
                    kwargs, _iteration,
                )
            except TimeoutError:
                log.warning("chat_tool_loop.llm_timeout iteration=%d", _iteration)
                _emit({
                    "phase": "complete",
                    "iteration": _iteration,
                    "max_iterations": max_iterations,
                    "total_iterations": _iteration + 1,
                    "total_tool_calls": _total_tool_calls,
                    "final_cost_usd": round(_cumulative_cost, 6),
                    "final_tokens": dict(_cumulative_tokens),
                    "total_elapsed_ms": int((time.monotonic() - _loop_start) * 1000),
                })
                msg = "Sorry, the model took too long to respond. Please try again."
                _finalize_trace("timeout")
                return msg, tool_audit, _cumulative_cost
            except Exception as e:
                log.exception("chat_tool_loop.llm_error iteration=%d", _iteration)
                _emit({
                    "phase": "complete",
                    "iteration": _iteration,
                    "max_iterations": max_iterations,
                    "total_iterations": _iteration + 1,
                    "total_tool_calls": _total_tool_calls,
                    "final_cost_usd": round(_cumulative_cost, 6),
                    "final_tokens": dict(_cumulative_tokens),
                    "total_elapsed_ms": int((time.monotonic() - _loop_start) * 1000),
                })
                _finalize_trace("error")
                return f"LLM call failed: {e}", tool_audit, _cumulative_cost

            # Extract token usage and cost from response
            call_tokens = {"prompt": 0, "completion": 0, "total": 0}
            call_cost = 0.0
            if hasattr(response, "usage") and response.usage:
                call_tokens["prompt"] = getattr(response.usage, "prompt_tokens", 0) or 0
                call_tokens["completion"] = getattr(response.usage, "completion_tokens", 0) or 0
                call_tokens["total"] = getattr(response.usage, "total_tokens", 0) or 0
                _cumulative_tokens["prompt"] += call_tokens["prompt"]
                _cumulative_tokens["completion"] += call_tokens["completion"]
                _cumulative_tokens["total"] += call_tokens["total"]
            try:
                call_cost = litellm.completion_cost(completion_response=response)
                _cumulative_cost += call_cost
            except Exception:
                pass

            choice = response.choices[0]
            message = choice.message
            has_tool_calls = bool(getattr(message, "tool_calls", None))
            tool_count = len(message.tool_calls) if has_tool_calls else 0

            _emit({
                "phase": "llm_complete",
                "iteration": _iteration,
                "max_iterations": max_iterations,
                "model": selected_model,
                "call_tokens": call_tokens,
                "call_cost_usd": round(call_cost, 6),
                "has_tool_calls": has_tool_calls,
                "tool_count": tool_count,
            })

            # Phase 4.3: Record LLM call trace
            _llm_duration = int(
                (time.monotonic() - _loop_start) * 1000
            ) - sum(t.duration_ms for t in _trace.tool_calls)
            _trace.llm_calls.append(LLMCallTrace(
                iteration=_iteration,
                model=selected_model,
                prompt_tokens=call_tokens["prompt"],
                completion_tokens=call_tokens["completion"],
                cost_usd=call_cost,
                duration_ms=max(_llm_duration, 0),
                has_tool_calls=has_tool_calls,
            ))

            # No tool calls → return the text reply
            if not has_tool_calls:
                reply_text = getattr(message, "content", "") or ""
                self._record_cost(response)
                _emit({
                    "phase": "complete",
                    "iteration": _iteration,
                    "max_iterations": max_iterations,
                    "total_iterations": _iteration + 1,
                    "total_tool_calls": _total_tool_calls,
                    "final_cost_usd": round(_cumulative_cost, 6),
                    "final_tokens": dict(_cumulative_tokens),
                    "total_elapsed_ms": int((time.monotonic() - _loop_start) * 1000),
                })
                _finalize_trace("success")
                return reply_text, tool_audit, _cumulative_cost

            # Append the assistant message with tool calls
            working_messages.append(message.model_dump())

            # Execute tool calls (parallel or sequential)
            _use_parallel = (
                _flags.is_enabled("parallel_tool_calls")
                and tool_count > 1
            )
            if _use_parallel:
                tool_results = await self._execute_tools_parallel(
                    message.tool_calls, _iteration, max_iterations,
                    _flags, progress_queue, _emit,
                )
            else:
                tool_results = []
                for idx, tool_call in enumerate(message.tool_calls):
                    r = await self._execute_single_tool(
                        tool_call, idx, tool_count,
                        _iteration, max_iterations,
                        _flags, progress_queue, _emit,
                    )
                    tool_results.append(r)

            for r in tool_results:
                _total_tool_calls += 1
                tool_audit.append(r["audit_entry"])
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "content": r["result_str"],
                })
                # Phase 4.3: Record tool call trace
                _trace.tool_calls.append(ToolCallTrace(
                    iteration=_iteration,
                    tool_name=r["tool_name"],
                    duration_ms=r["duration_ms"],
                    blocked=r["blocked"],
                    error=r["error"],
                    cache_hit=r.get("cache_hit", False),
                    result_size=len(r["result_str"]),
                ))

            self._record_cost(response)

            # Inject agent state so the LLM knows what it's already done
            tools_used = [e["tool"] for e in tool_audit]
            blocked_count = sum(1 for e in tool_audit if e.get("blocked"))

            # Phase 2.5: Pattern breaking — vary agent state format
            if _flags.is_enabled("pattern_breaking"):
                _AGENT_STATE_FORMATS = [
                    (
                        f"[AGENT STATE — Iteration {_iteration + 1}/{max_iterations}]\n"
                        f"Tools used: {', '.join(tools_used) or 'none'}\n"
                        f"Total calls: {_total_tool_calls} ({blocked_count} blocked)\n"
                        f"Cost: ${_cumulative_cost:.4f} | Tokens: {_cumulative_tokens['total']}"
                    ),
                    (
                        f"--- Progress: step {_iteration + 1} of {max_iterations} ---\n"
                        f"Executed: {_total_tool_calls} tools | "
                        f"Budget used: ${_cumulative_cost:.4f}"
                    ),
                    (
                        f"[Step {_iteration + 1}] "
                        f"{_total_tool_calls} tool calls completed. "
                        f"Tokens: {_cumulative_tokens['total']}. "
                        f"Remaining iterations: {max_iterations - _iteration - 1}."
                    ),
                    (
                        f"Status update (iteration {_iteration + 1}/{max_iterations}): "
                        f"tools={_total_tool_calls}, "
                        f"cost=${_cumulative_cost:.4f}, "
                        f"blocked={blocked_count}"
                    ),
                ]
                state_content = random.choice(_AGENT_STATE_FORMATS)
            else:
                state_content = (
                    f"[AGENT STATE — Iteration {_iteration + 1}/{max_iterations}]\n"
                    f"Tools used: {', '.join(tools_used) or 'none'}\n"
                    f"Total calls: {_total_tool_calls} ({blocked_count} blocked)\n"
                    f"Cost: ${_cumulative_cost:.4f} | Tokens: {_cumulative_tokens['total']}"
                )

            working_messages.append({
                "role": "system",
                "content": state_content,
            })

            # Phase 2.3: Recitation pattern — re-state original request periodically
            if (_flags.is_enabled("recitation_pattern")
                    and _original_request
                    and (_iteration + 1) % 4 == 0):
                working_messages.append({
                    "role": "system",
                    "content": (
                        f"[TASK RECITATION]\n"
                        f"Original user request: {_original_request}\n"
                        f"You have completed {_total_tool_calls} tool calls. "
                        f"Focus on directly answering the above request."
                    ),
                })

            # Check for user interjections between iterations
            if interjection_queue is not None:
                try:
                    interjection = interjection_queue.get_nowait()
                    working_messages.append({
                        "role": "system",
                        "content": (
                            f"USER INTERJECTION: {interjection}\n"
                            "Acknowledge this and adjust your approach."
                        ),
                    })
                    _emit({
                        "phase": "interjection_received",
                        "iteration": _iteration,
                        "interjection": interjection[:200],
                    })
                except asyncio.QueueEmpty:
                    pass

        # Max iterations reached — return whatever text we have
        last_text = ""
        for msg in reversed(working_messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_text = msg.get("content", "") or ""
                if last_text:
                    break
            elif hasattr(msg, "role") and msg.role == "assistant":
                last_text = getattr(msg, "content", "") or ""
                if last_text:
                    break
        if not last_text:
            last_text = "I ran out of steps processing your request. Please try again."
        _emit({
            "phase": "complete",
            "iteration": max_iterations - 1,
            "max_iterations": max_iterations,
            "total_iterations": max_iterations,
            "total_tool_calls": _total_tool_calls,
            "final_cost_usd": round(_cumulative_cost, 6),
            "final_tokens": dict(_cumulative_tokens),
            "total_elapsed_ms": int((time.monotonic() - _loop_start) * 1000),
        })
        _finalize_trace("max_iterations")
        return last_text, tool_audit, _cumulative_cost

    async def _execute_single_tool(
        self,
        tool_call: Any,
        idx: int,
        tool_count: int,
        _iteration: int,
        max_iterations: int,
        _flags: Any,
        progress_queue: asyncio.Queue | None,
        _emit: Any,
    ) -> dict:
        """Execute a single tool call and return structured result.

        Used by both sequential and parallel execution paths.
        Returns dict with keys: tool_call_id, tool_name, result_str,
        blocked, error, cache_hit, duration_ms, audit_entry.
        """
        fn = tool_call.function
        tool_name = fn.name
        try:
            params = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError:
            params = {}

        params_preview = json.dumps(params)[:120] if params else ""
        _emit({
            "phase": "tool_start",
            "iteration": _iteration,
            "max_iterations": max_iterations,
            "tool_name": tool_name,
            "tool_index": idx,
            "tool_count": tool_count,
            "params_preview": params_preview,
        })

        tool_start = time.monotonic()
        blocked = False
        error = False
        cache_hit = False
        result_str = ""

        # Phase 2.4: Handle load_artifact tool
        if tool_name == "load_artifact" and _flags.is_enabled(
            "artifact_handles"
        ):
            handle = params.get("handle", "")
            content = self._artifact_store.load(handle)
            result_str = content if content else (
                f"Artifact '{handle}' not found."
            )
            audit_entry = {
                "tool": tool_name, "params": params,
                "result": result_str[:500], "blocked": False,
            }
        else:
            _gate_ok, _gate_reason = self._check_tool_gate(
                tool_name, params,
            )
            if not _gate_ok:
                result_str = (
                    f"Tool '{tool_name}' blocked: {_gate_reason}"
                )
                blocked = True
                audit_entry = {
                    "tool": tool_name, "params": params,
                    "result": result_str, "blocked": True,
                }
            else:
                _LONG_TIMEOUT_TOOLS = {
                    "plan_and_execute",
                    "deep_research",
                    "swarm_research",
                }
                tool_timeout = (
                    600.0 if tool_name in _LONG_TIMEOUT_TOOLS
                    else 120.0
                )

                # Phase 3.5: Check sub-agent cache
                if _flags.is_enabled("subagent_cache"):
                    try:
                        from qe.runtime.tool_result_cache import (
                            get_tool_result_cache,
                        )
                        cached = get_tool_result_cache().get(
                            tool_name, params,
                        )
                        if cached is not None:
                            result_str = cached
                            cache_hit = True
                    except Exception:
                        pass

                if not cache_hit:
                    try:
                        result_str = await asyncio.wait_for(
                            self._execute_chat_tool(
                                tool_name, params,
                                progress_queue=progress_queue,
                            ),
                            timeout=tool_timeout,
                        )
                        if _flags.is_enabled("subagent_cache"):
                            try:
                                from qe.runtime.tool_result_cache import (
                                    get_tool_result_cache,
                                )
                                get_tool_result_cache().put(
                                    tool_name, params, result_str,
                                )
                            except Exception:
                                pass
                    except TimeoutError:
                        log.warning(
                            "chat_tool_loop.tool_timeout tool=%s "
                            "timeout=%.0fs iteration=%d",
                            tool_name, tool_timeout, _iteration,
                        )
                        result_str = (
                            f"Tool '{tool_name}' timed out after "
                            f"{int(tool_timeout)}s."
                        )
                        error = True
                    except Exception as e:
                        log.exception(
                            "chat_tool_loop.tool_error tool=%s",
                            tool_name,
                        )
                        result_str = f"Tool '{tool_name}' failed: {e}"
                        error = True

                # Phase 2.4: Artifact handle for large results
                if (
                    _flags.is_enabled("artifact_handles")
                    and not blocked
                    and not error
                ):
                    result_str = self._artifact_store.store(
                        result_str, tool_name,
                    )

                audit_entry = {
                    "tool": tool_name, "params": params,
                    "result": result_str[:500], "blocked": False,
                }

        tool_duration_ms = int((time.monotonic() - tool_start) * 1000)
        _emit({
            "phase": "tool_complete",
            "iteration": _iteration,
            "max_iterations": max_iterations,
            "tool_name": tool_name,
            "tool_index": idx,
            "tool_count": tool_count,
            "duration_ms": tool_duration_ms,
            "result_preview": result_str[:120] if result_str else "",
            "blocked": blocked,
            "error": error,
        })

        # Phase 4.4: Per-tool metrics
        try:
            from qe.runtime.metrics import get_metrics as _gm
            _m = _gm()
            _m.counter("chat_tool_calls_total").inc()
            _m.histogram("chat_tool_latency_ms").observe(
                float(tool_duration_ms),
            )
            if error:
                _m.counter("chat_tool_errors_total").inc()
        except Exception:
            pass

        return {
            "tool_call_id": tool_call.id,
            "tool_name": tool_name,
            "result_str": result_str,
            "blocked": blocked,
            "error": error,
            "cache_hit": cache_hit,
            "duration_ms": tool_duration_ms,
            "audit_entry": audit_entry,
        }

    async def _execute_tools_parallel(
        self,
        tool_calls: list,
        _iteration: int,
        max_iterations: int,
        _flags: Any,
        progress_queue: asyncio.Queue | None,
        _emit: Any,
    ) -> list[dict]:
        """Execute multiple tool calls concurrently.

        Returns list of result dicts (same shape as _execute_single_tool).
        One failure does not cancel others.
        """
        tool_count = len(tool_calls)
        coros = [
            self._execute_single_tool(
                tc, idx, tool_count, _iteration, max_iterations,
                _flags, progress_queue, _emit,
            )
            for idx, tc in enumerate(tool_calls)
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Convert exceptions to error results
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                log.exception(
                    "parallel_tool_exec.exception tool=%s",
                    tool_calls[i].function.name,
                )
                processed.append({
                    "tool_call_id": tool_calls[i].id,
                    "tool_name": tool_calls[i].function.name,
                    "result_str": f"Tool failed: {r}",
                    "blocked": False,
                    "error": True,
                    "cache_hit": False,
                    "duration_ms": 0,
                    "audit_entry": {
                        "tool": tool_calls[i].function.name,
                        "params": {},
                        "result": f"Tool failed: {r}"[:500],
                        "blocked": False,
                    },
                })
            else:
                processed.append(r)
        return processed

    def _check_tool_gate(
        self, tool_name: str, params: dict
    ) -> tuple[bool, str]:
        """Check the tool gate. Returns (allowed, reason)."""
        if not self._tool_gate:
            return True, ""
        try:
            from qe.runtime.tool_gate import GateDecision

            result = self._tool_gate.validate(
                tool_name=tool_name,
                params=params,
                capabilities=self._tool_capabilities_for_mode(),
                goal_id="chat",
            )
            if result.decision == GateDecision.DENY:
                return False, result.reason
            return True, ""
        except Exception:
            log.debug("Tool gate check failed", exc_info=True)
            return True, ""

    async def _execute_chat_tool(
        self,
        tool_name: str,
        params: dict,
        progress_queue: asyncio.Queue | None = None,
    ) -> str:
        """Dispatch to local handler or tool registry."""
        handlers: dict[str, Any] = {
            "query_beliefs": lambda p: self._tool_query_beliefs(p["query"]),
            "list_entities": lambda _p: self._tool_list_entities(),
            "get_entity_details": lambda p: self._tool_get_entity_details(
                p["entity"],
            ),
            "submit_observation": lambda p: self._tool_submit_observation(
                p["text"],
            ),
            "retract_claim": lambda p: self._tool_retract_claim(
                p["claim_id"],
            ),
            "get_budget_status": lambda _p: self._tool_get_budget(),
            "deep_research": lambda p: self._tool_deep_research(
                p["question"],
            ),
            "swarm_research": lambda p: self._tool_swarm_research(
                p["question"], p.get("num_agents", 3),
            ),
            "plan_and_execute": lambda p: self._tool_plan_and_execute(
                p["goal"], progress_queue=progress_queue,
            ),
            "reason_about": lambda p: self._tool_reason_about(p["claim"]),
            "crystallize_insights": lambda p: self._tool_crystallize_insights(
                p["finding"], p.get("domain", "general"),
            ),
            "consolidate_knowledge": (
                lambda _p: self._tool_consolidate_knowledge()
            ),
            "delegate_to_agent": lambda p: self._tool_delegate_to_agent(
                p["agent_url"], p["task_description"]
            ),
        }

        if tool_name in handlers:
            try:
                return await handlers[tool_name](params)
            except Exception as e:
                log.exception("Chat tool %s failed", tool_name)
                return f"Tool error: {e}"

        # Fall back to tool registry
        if self._tool_registry:
            try:
                result = await self._tool_registry.execute(tool_name, params)
                return str(result)
            except Exception as e:
                log.exception("Registry tool %s failed", tool_name)
                return f"Tool error: {e}"

        return f"Unknown tool: {tool_name}"

    # ── Response builder ────────────────────────────────────────────────

    def _build_response(
        self,
        message_id: str,
        reply_text: str,
        tool_audit: list[dict],
    ) -> ChatResponsePayload:
        """Construct ChatResponsePayload from agent output."""
        tracking_id = None
        claims: list[dict] = []
        cognitive_used = False

        for entry in tool_audit:
            tool = entry.get("tool", "")
            if tool == "submit_observation" and not entry.get("blocked"):
                # Extract tracking ID from result
                result = entry.get("result", "")
                if "Tracking ID:" in result:
                    tracking_id = result.split("Tracking ID:")[1].strip().split("\n")[0].strip()
            if tool == "query_beliefs" and not entry.get("blocked"):
                # Mark that we have claim results
                pass
            if tool in {
                "deep_research", "swarm_research", "plan_and_execute",
                "reason_about", "crystallize_insights", "consolidate_knowledge",
            } and not entry.get("blocked"):
                cognitive_used = True

        return ChatResponsePayload(
            message_id=message_id,
            reply_text=reply_text,
            intent=ChatIntent.CONVERSATION,
            claims=claims,
            tracking_envelope_id=tracking_id,
            tool_calls_made=[
                {"tool": e["tool"], "blocked": e.get("blocked", False)}
                for e in tool_audit
            ],
            cognitive_process_used=cognitive_used,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    async def _extract_session_patterns(
        self, tool_audit: list[dict], cost_usd: float,
    ) -> None:
        """Record tool sequence outcome in procedural memory."""
        if not self._procedural_memory:
            return
        tool_names = [
            e["tool"] for e in tool_audit if not e.get("blocked")
        ]
        if not tool_names:
            return
        has_error = any(
            "error" in (e.get("result", "") or "").lower()
            or "failed" in (e.get("result", "") or "").lower()
            or "timed out" in (e.get("result", "") or "").lower()
            for e in tool_audit
            if not e.get("blocked")
        )
        try:
            await self._procedural_memory.record_sequence_outcome(
                sequence_id=None,
                tool_names=tool_names,
                success=not has_error,
                cost_usd=cost_usd,
                domain="chat",
            )
        except Exception:
            log.debug("Failed to record session patterns", exc_info=True)

    def _record_cost(self, response: Any) -> None:
        """Record LLM cost to budget tracker using actual response."""
        if self.budget_tracker is None:
            return
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        self.budget_tracker.record_cost(self.model, cost, service_id="chat")
