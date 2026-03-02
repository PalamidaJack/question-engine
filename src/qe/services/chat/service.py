"""Chat service: direct-conversation agent with cognitive augmentation."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm
from dotenv import load_dotenv

from qe.models.envelope import Envelope
from qe.runtime.budget import BudgetTracker
from qe.services.chat.schemas import ChatIntent, ChatResponsePayload
from qe.services.inquiry.schemas import InquiryConfig
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

_MAX_HISTORY = 50
_HISTORY_TRIM_TO = 30
_MAX_CONTEXT_MESSAGES = 16

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
]


class ChatSession:
    """Per-session conversation state."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: list[dict[str, str]] = []
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
    ) -> None:
        self.substrate = substrate
        self.bus = bus
        self.budget_tracker = budget_tracker
        self.model = model
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
        self._sessions: dict[str, ChatSession] = {}

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

    # ── Main entry point ────────────────────────────────────────────────

    async def handle_message(
        self, session_id: str, user_message: str
    ) -> ChatResponsePayload:
        """Main entry point: receive user message, return structured response."""
        session = self.get_or_create_session(session_id)
        session.add_user_message(user_message)
        message_id = str(uuid.uuid4())

        if self.budget_tracker and self.budget_tracker.remaining_pct() <= 0:
            return ChatResponsePayload(
                message_id=message_id,
                reply_text="Budget exhausted. Unable to process your request.",
                intent=ChatIntent.CONVERSATION,
                error="budget_exhausted",
            )

        try:
            messages = await self._build_messages(session)
            tool_schemas = self._get_chat_tools()
            reply_text, tool_audit = await asyncio.wait_for(
                self._chat_tool_loop(
                    messages, tool_schemas, max_iterations=8
                ),
                timeout=300.0,
            )
            response = self._build_response(
                message_id, reply_text, tool_audit
            )
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
        return response

    # ── System prompt & context assembly ────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Static identity and behavioral instructions."""
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

    async def _build_messages(self, session: ChatSession) -> list[dict]:
        """Assemble full message list for the LLM call."""
        system_parts = [self._build_system_prompt()]

        knowledge_ctx = await self._build_knowledge_context()
        if knowledge_ctx:
            system_parts.append(f"\n## Current state\n{knowledge_ctx}")

        memory_ctx = await self._build_memory_context(session)
        if memory_ctx:
            system_parts.append(f"\n## {memory_ctx}")

        messages: list[dict] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]
        messages.extend(session.history[-_MAX_CONTEXT_MESSAGES:])
        return messages

    # ── Tool definitions ────────────────────────────────────────────────

    def _get_chat_tools(self) -> list[dict]:
        """Return combined tool schemas: chat-specific + selected registry tools."""
        tools = list(_CHAT_TOOL_SCHEMAS)

        if self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_tool_schemas(
                    capabilities={"web_search", "web_fetch"},
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

    async def _tool_plan_and_execute(self, goal: str) -> str:
        """Decompose a goal into subtasks and execute them."""
        if not self._planner or not self._dispatcher:
            return "Plan-and-execute is not available (planner/dispatcher not initialized)."
        try:
            state = await self._planner.decompose(goal)
            goal_id = state.goal_id

            done = asyncio.Event()
            result_holder: dict = {}

            async def on_synthesized(envelope: Envelope) -> None:
                if envelope.payload.get("goal_id") == goal_id:
                    result_holder["data"] = envelope.payload
                    done.set()

            self.bus.subscribe("goals.synthesized", on_synthesized)
            try:
                await self._dispatcher.submit_goal(state)
                await asyncio.wait_for(done.wait(), timeout=180.0)
            finally:
                self.bus.unsubscribe("goals.synthesized", on_synthesized)

            data = result_holder.get("data", {})
            parts = [f"Goal '{goal}' completed."]
            if data.get("synthesis"):
                parts.append(f"\n{data['synthesis']}")
            subtask_count = len(state.decomposition.subtasks) if state.decomposition else 0
            parts.append(f"\n(Goal ID: {goal_id}, Subtasks: {subtask_count})")
            return "\n".join(parts)

        except TimeoutError:
            return (
                f"Goal execution timed out after 180s. "
                f"The goal '{goal}' may still be processing in the background."
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

    # ── Agent loop ──────────────────────────────────────────────────────

    async def _chat_tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        max_iterations: int = 8,
    ) -> tuple[str, list[dict]]:
        """Simple ReAct loop: call LLM, execute tools, repeat.

        Returns (reply_text, tool_audit_log).
        """
        tool_audit: list[dict] = []
        working_messages = list(messages)

        for _iteration in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": working_messages,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(**kwargs), timeout=60.0
                )
            except TimeoutError:
                log.warning("chat_tool_loop.llm_timeout iteration=%d", _iteration)
                return "Sorry, the model took too long to respond. Please try again.", tool_audit
            except Exception as e:
                log.exception("chat_tool_loop.llm_error iteration=%d", _iteration)
                return f"LLM call failed: {e}", tool_audit
            choice = response.choices[0]
            message = choice.message

            # No tool calls → return the text reply
            if not getattr(message, "tool_calls", None):
                reply_text = getattr(message, "content", "") or ""
                self._record_cost(working_messages)
                return reply_text, tool_audit

            # Append the assistant message with tool calls
            working_messages.append(message.model_dump())

            # Execute each tool call
            for tool_call in message.tool_calls:
                fn = tool_call.function
                tool_name = fn.name
                try:
                    params = json.loads(fn.arguments) if fn.arguments else {}
                except json.JSONDecodeError:
                    params = {}

                # Tool gate check
                gate_ok, gate_reason = self._check_tool_gate(tool_name, params)
                if not gate_ok:
                    result_str = f"Tool '{tool_name}' blocked: {gate_reason}"
                    tool_audit.append({
                        "tool": tool_name,
                        "params": params,
                        "result": result_str,
                        "blocked": True,
                    })
                else:
                    try:
                        result_str = await asyncio.wait_for(
                            self._execute_chat_tool(tool_name, params),
                            timeout=120.0,
                        )
                    except TimeoutError:
                        log.warning(
                            "chat_tool_loop.tool_timeout tool=%s iteration=%d",
                            tool_name, _iteration,
                        )
                        result_str = f"Tool '{tool_name}' timed out after 120s."
                    except Exception as e:
                        log.exception(
                            "chat_tool_loop.tool_error tool=%s", tool_name,
                        )
                        result_str = f"Tool '{tool_name}' failed: {e}"
                    tool_audit.append({
                        "tool": tool_name,
                        "params": params,
                        "result": result_str[:500],
                        "blocked": False,
                    })

                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

            self._record_cost(working_messages)

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
        return last_text, tool_audit

    # Capabilities the chat agent declares for tool gate validation.
    _CHAT_CAPABILITIES: set[str] = {
        "chat", "web_search", "web_fetch",
    }

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
                capabilities=self._CHAT_CAPABILITIES,
                goal_id="chat",
            )
            if result.decision == GateDecision.DENY:
                return False, result.reason
            return True, ""
        except Exception:
            log.debug("Tool gate check failed", exc_info=True)
            return True, ""

    async def _execute_chat_tool(
        self, tool_name: str, params: dict
    ) -> str:
        """Dispatch to local handler or tool registry."""
        handlers: dict[str, Any] = {
            "query_beliefs": lambda p: self._tool_query_beliefs(p["query"]),
            "list_entities": lambda _p: self._tool_list_entities(),
            "get_entity_details": lambda p: self._tool_get_entity_details(p["entity"]),
            "submit_observation": lambda p: self._tool_submit_observation(p["text"]),
            "retract_claim": lambda p: self._tool_retract_claim(p["claim_id"]),
            "get_budget_status": lambda _p: self._tool_get_budget(),
            "deep_research": lambda p: self._tool_deep_research(p["question"]),
            "swarm_research": lambda p: self._tool_swarm_research(
                p["question"], p.get("num_agents", 3),
            ),
            "plan_and_execute": lambda p: self._tool_plan_and_execute(p["goal"]),
            "reason_about": lambda p: self._tool_reason_about(p["claim"]),
            "crystallize_insights": lambda p: self._tool_crystallize_insights(
                p["finding"], p.get("domain", "general"),
            ),
            "consolidate_knowledge": lambda _p: self._tool_consolidate_knowledge(),
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

    def _record_cost(self, messages: list[dict]) -> None:
        """Record LLM cost to budget tracker."""
        if self.budget_tracker is None:
            return
        try:
            cost = litellm.completion_cost(
                model=self.model, messages=messages, completion=""
            )
            self.budget_tracker.record_cost(self.model, cost)
        except Exception:
            pass
