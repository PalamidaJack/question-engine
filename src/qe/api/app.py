"""FastAPI application for Question Engine OS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qe.api.endpoints.goals import register_goal_routes
from qe.api.endpoints.memory import register_memory_routes
from qe.api.middleware import AuthMiddleware, RateLimitMiddleware, RequestTimingMiddleware
from qe.api.profiling import InquiryProfilingStore
from qe.api.setup import (
    CHANNELS,
    PROVIDERS,
    get_configured_channels,
    get_configured_providers,
    get_current_tiers,
    get_settings,
    is_setup_complete,
    save_settings,
    save_setup,
)
from qe.api.ws import ConnectionManager
from qe.audit import get_audit_log
from qe.bus import get_bus
from qe.bus.bus_metrics import get_bus_metrics
from qe.bus.event_log import EventLog
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.optimization.prompt_registry import PromptRegistry, register_all_baselines
from qe.runtime.context_curator import ContextCurator
from qe.runtime.episodic_memory import EpisodicMemory
from qe.runtime.epistemic_reasoner import EpistemicReasoner
from qe.runtime.feature_flags import get_flag_store
from qe.runtime.logging_config import configure_from_config, update_log_level
from qe.runtime.metacognitor import Metacognitor
from qe.runtime.metrics import get_metrics
from qe.runtime.persistence_engine import PersistenceEngine
from qe.runtime.procedural_memory import ProceduralMemory
from qe.runtime.readiness import get_readiness
from qe.runtime.service import BaseService
from qe.services.chat import ChatService
from qe.services.dispatcher import Dispatcher
from qe.services.doctor import DoctorService
from qe.services.executor import ExecutorService
from qe.services.inquiry.dialectic import DialecticEngine
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.hypothesis import HypothesisManager
from qe.services.inquiry.insight import InsightCrystallizer
from qe.services.inquiry.question_generator import QuestionGenerator
from qe.services.planner import PlannerService
from qe.services.query import answer_question
from qe.services.recovery import RecoveryOrchestrator
from qe.services.verification import VerificationGate, VerificationService
from qe.substrate import Substrate
from qe.substrate.bayesian_belief import BayesianBeliefStore
from qe.substrate.failure_kb import FailureKnowledgeBase
from qe.substrate.goal_store import GoalStore
from qe.substrate.memory_store import MemoryStore
from qe.substrate.question_store import QuestionStore

load_dotenv()

log = logging.getLogger(__name__)

ws_manager = ConnectionManager()

# Global references set during lifespan
_supervisor: Supervisor | None = None
_substrate: Substrate | None = None
_supervisor_task: asyncio.Task | None = None
_event_log: EventLog | None = None
_chat_service: ChatService | None = None
_planner: PlannerService | None = None
_dispatcher: Dispatcher | None = None
_executor: ExecutorService | None = None
_goal_store: GoalStore | None = None
_doctor: DoctorService | None = None
_verification_gate: VerificationGate | None = None
_memory_store: MemoryStore | None = None
_notification_router = None
_active_adapters: list = []
_extra_routes_registered = False
_inquiry_engine: InquiryEngine | None = None
_cognitive_pool = None
_competitive_arena = None
_strategy_evolver = None
_prompt_registry: PromptRegistry | None = None
_prompt_mutator = None
_knowledge_loop = None
_inquiry_bridge = None
_elastic_scaler = None
_episodic_memory = None
_synthesizer = None
_tool_registry = None
_tool_gate = None
_workspace_manager = None
_last_inquiry_profile: dict[str, Any] = {}
_inquiry_profiling_store = InquiryProfilingStore()

INBOX_DIR = Path("data/runtime_inbox")


def _configure_kilocode() -> None:
    """If KILOCODE_API_KEY is set, configure litellm env vars for openai/ prefix routing."""
    import litellm

    kilo_key = os.environ.get("KILOCODE_API_KEY", "")
    if not kilo_key:
        return
    kilo_base = os.environ.get(
        "KILOCODE_API_BASE", "https://kilo.ai/api/openrouter"
    )
    # litellm uses OPENAI_API_KEY / OPENAI_API_BASE for the "openai/" prefix
    os.environ.setdefault("OPENAI_API_KEY", kilo_key)
    os.environ.setdefault("OPENAI_API_BASE", kilo_base)

    # Register Kilo Code models so litellm recognises them
    _kilo_models = {
        "openai/anthropic/claude-sonnet-4": {
            "max_tokens": 8192,
            "max_input_tokens": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "litellm_provider": "openai",
        },
        "openai/google/gemini-2.0-flash": {
            "max_tokens": 8192,
            "max_input_tokens": 1048576,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000004,
            "litellm_provider": "openai",
        },
    }
    litellm.register_model(_kilo_models)
    log.info(
        "kilocode.configured base=%s models=%s",
        kilo_base,
        list(_kilo_models.keys()),
    )


async def _init_channels() -> None:
    """Conditionally start channel adapters based on env vars."""
    global _notification_router, _active_adapters

    from qe.channels import (
        EmailAdapter,
        NotificationRouter,
        SlackAdapter,
        TelegramAdapter,
        WebhookAdapter,
    )

    _notification_router = NotificationRouter()
    activated: list[str] = []
    bus = get_bus()

    def _publish_channel_message(msg: dict) -> None:
        """Route an adapter message to the correct bus topic based on command."""
        command = msg.get("command", "goal")
        topic_map = {
            "ask": "queries.asked",
            "status": "system.health.check",
        }
        topic = topic_map.get(command, "channel.message_received")
        bus.publish(
            Envelope(
                topic=topic,
                source_service_id=f"channel_{msg.get('channel', 'unknown')}",
                payload={
                    "channel": msg.get("channel", "unknown"),
                    "user_id": msg.get("user_id", ""),
                    "text": msg.get("sanitized_text", msg.get("text", "")),
                    "command": command,
                },
            )
        )

    # Telegram
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if tg_token:
        adapter = TelegramAdapter(
            bot_token=tg_token, message_callback=_publish_channel_message
        )
        try:
            await adapter.start()
            _notification_router.register_channel("telegram", adapter)
            _active_adapters.append(adapter)
            activated.append("telegram")
        except Exception:
            log.exception("channel.telegram_start_failed")

    # Slack
    slack_bot = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_app = os.environ.get("SLACK_APP_TOKEN", "")
    if slack_bot and slack_app:
        adapter = SlackAdapter(
            bot_token=slack_bot,
            app_token=slack_app,
            message_callback=_publish_channel_message,
        )
        try:
            await adapter.start()
            _notification_router.register_channel("slack", adapter)
            _active_adapters.append(adapter)
            activated.append("slack")
        except Exception:
            log.exception("channel.slack_start_failed")

    # Email
    email_host = os.environ.get("EMAIL_IMAP_HOST", "")
    email_user = os.environ.get("EMAIL_USERNAME", "")
    email_pass = os.environ.get("EMAIL_PASSWORD", "")
    if email_host and email_user and email_pass:
        adapter = EmailAdapter(
            imap_host=email_host,
            username=email_user,
            password=email_pass,
            message_callback=_publish_channel_message,
        )
        try:
            await adapter.start()
            _notification_router.register_channel("email", adapter)
            _active_adapters.append(adapter)
            activated.append("email")
        except Exception:
            log.exception("channel.email_start_failed")

    # Webhook (no start needed — driven by HTTP endpoint)
    webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
    webhook_adapter = WebhookAdapter(secret=webhook_secret)
    await webhook_adapter.start()
    _notification_router.register_channel("webhook", webhook_adapter)
    _active_adapters.append(webhook_adapter)
    activated.append("webhook")

    # ── Bus subscriptions: channel → goals, goals → notifications ──

    async def _on_channel_message(envelope: Envelope) -> None:
        """Forward inbound channel messages to the planner as goals."""
        text = envelope.payload.get("text", "").strip()
        if not text:
            return

        user_id = envelope.payload.get("user_id", "")
        channel = envelope.payload.get("channel", "unknown")

        log.info(
            "channel.goal_from_message channel=%s user=%s text=%s",
            channel,
            user_id,
            text[:80],
        )

        # v2 Inquiry path
        if get_flag_store().is_enabled("inquiry_mode") and _inquiry_engine is not None:
            try:
                goal_id = f"goal_{uuid.uuid4().hex[:12]}"

                # Select strategy via Thompson sampling
                config = None
                if _strategy_evolver is not None:
                    from qe.runtime.strategy_models import strategy_to_inquiry_config
                    strategy = _strategy_evolver.select_strategy()
                    config = strategy_to_inquiry_config(strategy)

                result = await _inquiry_engine.run_inquiry(
                    goal_id=goal_id, goal_description=text,
                    config=config,
                )
                # Notify via channel if router available
                if _notification_router is not None:
                    await _notification_router.notify(
                        user_id=user_id,
                        event_type="inquiry_result",
                        message=f"Inquiry complete: {result.findings_summary[:500]}",
                        urgency="high",
                    )
                return
            except Exception:
                log.exception("channel.inquiry_failed channel=%s", channel)

        # v1 Pipeline fallback
        if not _planner or not _dispatcher:
            log.warning("channel.message_ignored reason=engine_not_ready")
            return

        try:
            state = await _planner.decompose(text)
            state.metadata["origin_user_id"] = user_id
            state.metadata["origin_channel"] = channel
            await _dispatcher.submit_goal(state)

            bus.publish(
                Envelope(
                    topic="goals.submitted",
                    source_service_id="channel_bridge",
                    correlation_id=state.goal_id,
                    payload={
                        "goal_id": state.goal_id,
                        "description": text,
                        "channel": channel,
                        "user_id": user_id,
                    },
                )
            )
        except Exception:
            log.exception(
                "channel.goal_creation_failed channel=%s user=%s",
                channel,
                user_id,
            )

    bus.subscribe("channel.message_received", _on_channel_message)

    async def _on_goal_completed(envelope: Envelope) -> None:
        """Deliver goal results to users via notification channels."""
        if _notification_router is None:
            return

        goal_id = envelope.payload.get("goal_id", "")
        if not goal_id:
            return

        # Build a summary from the goal's subtask results
        summary = f"Goal completed with {envelope.payload.get('subtask_count', 0)} subtasks."
        target_user = "broadcast"
        if _goal_store:
            state = await _goal_store.load_goal(goal_id)
            if state:
                target_user = state.metadata.get("origin_user_id", "broadcast") or "broadcast"
                if state.subtask_results:
                    parts = []
                    for _st_id, res in state.subtask_results.items():
                        output_text = res.output.get("result", "")
                        if output_text:
                            parts.append(str(output_text)[:500])
                    if parts:
                        summary = "\n---\n".join(parts)

        await _notification_router.notify(
            user_id=target_user,
            event_type="goal_result",
            message=f"Goal {goal_id} completed:\n{summary}",
            urgency="high",
        )

    async def _on_goal_failed(envelope: Envelope) -> None:
        """Notify channels when a goal fails."""
        if _notification_router is None:
            return

        goal_id = envelope.payload.get("goal_id", "")
        reason = envelope.payload.get("reason", "unknown")

        target_user = "broadcast"
        if _goal_store:
            state = await _goal_store.load_goal(goal_id)
            if state:
                target_user = state.metadata.get("origin_user_id", "broadcast") or "broadcast"

        await _notification_router.notify(
            user_id=target_user,
            event_type="goal_failed",
            message=f"Goal {goal_id} failed: {reason}",
            urgency="high",
        )

    bus.subscribe("goals.completed", _on_goal_completed)
    bus.subscribe("goals.failed", _on_goal_failed)

    async def _on_drift_detected(envelope: Envelope) -> None:
        """Notify the user when a subtask output drifts from the goal."""
        if _notification_router is None:
            return

        goal_id = envelope.payload.get("goal_id", "")
        subtask_id = envelope.payload.get("subtask_id", "")
        similarity = envelope.payload.get("similarity", 0.0)

        target_user = "broadcast"
        if _goal_store:
            state = await _goal_store.load_goal(goal_id)
            if state:
                target_user = state.metadata.get("origin_user_id", "broadcast") or "broadcast"

        await _notification_router.notify(
            user_id=target_user,
            event_type="goal_drift",
            message=(
                f"Drift detected on goal {goal_id}: "
                f"subtask {subtask_id} similarity={similarity:.3f}"
            ),
            urgency="normal",
        )

    bus.subscribe("goals.drift_detected", _on_drift_detected)

    async def _on_query_asked(envelope: Envelope) -> None:
        """Handle /ask command: answer from belief ledger and notify the user."""
        if not _substrate or not _notification_router:
            return

        text = envelope.payload.get("text", "").strip()
        user_id = envelope.payload.get("user_id", "broadcast")
        if not text:
            return

        try:
            result = await answer_question(text, _substrate)
            answer = result.get("answer", "No answer found.")
            await _notification_router.notify(
                user_id=user_id or "broadcast",
                event_type="query_answer",
                message=f"Q: {text}\n\nA: {answer}",
                urgency="normal",
            )
        except Exception:
            log.exception("query.answer_failed question=%s", text[:80])
            await _notification_router.notify(
                user_id=user_id or "broadcast",
                event_type="query_error",
                message=f"Failed to answer: {text[:200]}",
                urgency="normal",
            )

    async def _on_health_check(envelope: Envelope) -> None:
        """Handle /status command: return system status to the user."""
        if not _notification_router:
            return

        user_id = envelope.payload.get("user_id", "broadcast")

        parts = ["System Status:"]
        parts.append(f"  Supervisor: {'running' if _supervisor else 'stopped'}")
        parts.append(f"  Substrate: {'ready' if _substrate else 'not ready'}")
        parts.append(f"  Channels: {len(_active_adapters)} active")

        if _doctor and _doctor.last_report:
            parts.append(f"  Health: {_doctor.last_report.get('status', 'unknown')}")

        await _notification_router.notify(
            user_id=user_id or "broadcast",
            event_type="system_status",
            message="\n".join(parts),
            urgency="normal",
        )

    bus.subscribe("queries.asked", _on_query_asked)
    bus.subscribe("system.health.check", _on_health_check)

    log.info("channels.initialized active=%s", activated)


def _genome_paths() -> list[Path]:
    genomes_dir = Path("genomes")
    if not genomes_dir.exists():
        return []
    return sorted(genomes_dir.glob("*.toml"))


async def _inbox_relay_loop() -> None:
    """Relay cross-process submissions into the in-memory bus."""
    bus = get_bus()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    while True:
        for item in sorted(INBOX_DIR.glob("*.json")):  # noqa: ASYNC240
            try:
                payload = json.loads(item.read_text(encoding="utf-8"))
                env = Envelope.model_validate(payload)
                bus.publish(env)
                item.unlink(missing_ok=True)
            except Exception:
                log.warning("inbox.relay_failed file=%s", item.name, exc_info=True)
        await asyncio.sleep(0.5)


def _bus_to_ws_bridge() -> None:
    """Subscribe to all bus topics and forward events to WebSocket clients."""
    from qe.bus.protocol import TOPICS

    bus = get_bus()

    async def _forward(envelope: Envelope) -> None:
        event = {
            "type": "bus_event",
            "envelope_id": envelope.envelope_id,
            "topic": envelope.topic,
            "source_service_id": envelope.source_service_id,
            "timestamp": envelope.timestamp.isoformat(),
            "payload": envelope.payload,
        }
        await ws_manager.broadcast(json.dumps(event))

    for topic in TOPICS:
        bus.subscribe(topic, _forward)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the QE engine on app startup, shut down on teardown."""
    global _supervisor, _substrate, _supervisor_task, _event_log, _chat_service
    global _planner, _dispatcher, _executor, _goal_store, _doctor, _verification_gate
    global _memory_store, _extra_routes_registered, _inquiry_engine
    global _cognitive_pool, _competitive_arena, _strategy_evolver, _prompt_mutator, _knowledge_loop
    global _inquiry_bridge, _synthesizer
    global _elastic_scaler, _episodic_memory
    global _tool_registry, _tool_gate, _workspace_manager

    configure_from_config(get_settings())

    bus = get_bus()

    # Initialize durable event log
    readiness = get_readiness()

    _event_log = EventLog()
    await _event_log.initialize()
    bus.set_event_log(_event_log)
    readiness.mark_ready("event_log_ready")

    _substrate = Substrate()
    await _substrate.initialize()
    _memory_store = MemoryStore(_substrate.belief_ledger._db_path)
    _substrate.set_memory_store(_memory_store)

    if not _extra_routes_registered:
        register_memory_routes(
            app=app,
            memory_store=_memory_store,
        )

    readiness.mark_ready("substrate_ready")

    relay_task: asyncio.Task | None = None

    if is_setup_complete():
        _configure_kilocode()

        _supervisor = Supervisor(
            bus=bus, substrate=_substrate, config_path=Path("config.toml")
        )

        _bus_to_ws_bridge()

        relay_task = asyncio.create_task(_inbox_relay_loop())
        _supervisor_task = asyncio.create_task(
            _supervisor.start(_genome_paths())
        )
        balanced_model = get_current_tiers().get("balanced", "gpt-4o")
        fast_model = get_current_tiers().get("fast", "gpt-4o-mini")
        _db_path = _substrate.belief_ledger._db_path

        # Tool Infrastructure
        from qe.runtime.tool_bootstrap import create_default_gate, create_default_registry
        from qe.runtime.tool_gate import SecurityPolicy
        from qe.runtime.workspace import WorkspaceManager

        _tool_registry = create_default_registry()
        _tool_gate = create_default_gate(policies=[
            SecurityPolicy(
                name="default",
                max_calls_per_goal=50,
                blocked_domains=["localhost", "127.0.0.1"],
            ),
        ])
        _workspace_manager = WorkspaceManager(base_dir="data/workspaces")
        BaseService.set_tool_registry(_tool_registry)
        BaseService.set_tool_gate(_tool_gate)

        # Phase 1 Memory
        _episodic_memory = EpisodicMemory(db_path=_db_path)
        await _episodic_memory.initialize()
        _bayesian_belief = BayesianBeliefStore(db_path=_db_path)
        _context_curator = ContextCurator()
        BaseService.set_episodic_memory(_episodic_memory)
        BaseService.set_bayesian_belief(_bayesian_belief)
        BaseService.set_context_curator(_context_curator)

        # Prompt Evolution Registry (disabled by default)
        _prompt_registry = PromptRegistry(db_path=_db_path, bus=bus, enabled=False)
        await _prompt_registry.initialize()
        register_all_baselines(_prompt_registry)

        # Prompt Mutator (auto-generates variants when prompt_evolution flag enabled)
        from qe.optimization.prompt_mutator import PromptMutator

        _prompt_mutator = PromptMutator(
            registry=_prompt_registry,
            bus=bus,
            model=fast_model,
        )
        _prompt_mutator.start()

        # Phase 2 Cognitive
        _metacognitor = Metacognitor(
            episodic_memory=_episodic_memory, model=fast_model,
            prompt_registry=_prompt_registry,
        )
        _epistemic_reasoner = EpistemicReasoner(
            episodic_memory=_episodic_memory, belief_store=_bayesian_belief, model=fast_model,
            prompt_registry=_prompt_registry,
        )
        _dialectic_engine = DialecticEngine(
            episodic_memory=_episodic_memory, model=balanced_model,
            prompt_registry=_prompt_registry,
        )
        _persistence_engine = PersistenceEngine(
            episodic_memory=_episodic_memory, model=fast_model,
            prompt_registry=_prompt_registry,
        )
        _insight_crystallizer = InsightCrystallizer(
            episodic_memory=_episodic_memory, belief_store=_bayesian_belief, model=balanced_model,
            prompt_registry=_prompt_registry,
        )
        BaseService.set_metacognitor(_metacognitor)
        BaseService.set_epistemic_reasoner(_epistemic_reasoner)
        BaseService.set_dialectic_engine(_dialectic_engine)
        BaseService.set_persistence_engine(_persistence_engine)
        BaseService.set_insight_crystallizer(_insight_crystallizer)

        readiness.mark_ready("cognitive_layer_ready")

        # Phase 3 Inquiry Engine
        _question_generator = QuestionGenerator(
            model=fast_model, prompt_registry=_prompt_registry,
        )
        _hypothesis_manager = HypothesisManager(
            belief_store=_bayesian_belief, model=balanced_model,
            prompt_registry=_prompt_registry,
        )
        _question_store = QuestionStore(db_path=_substrate.belief_ledger._db_path)
        await _question_store.initialize()
        _procedural_memory = ProceduralMemory(db_path=_substrate.belief_ledger._db_path)
        await _procedural_memory.initialize()
        _inquiry_engine = InquiryEngine(
            episodic_memory=_episodic_memory,
            context_curator=_context_curator,
            metacognitor=_metacognitor,
            epistemic_reasoner=_epistemic_reasoner,
            dialectic_engine=_dialectic_engine,
            persistence_engine=_persistence_engine,
            insight_crystallizer=_insight_crystallizer,
            question_generator=_question_generator,
            hypothesis_manager=_hypothesis_manager,
            question_store=_question_store,
            procedural_memory=_procedural_memory,
            tool_registry=_tool_registry,
            bus=bus,
        )
        flag_store = get_flag_store()
        flag_store.define(
            "inquiry_mode",
            enabled=False,
            description="Route POST /api/goals to InquiryEngine instead of v1 pipeline",
        )
        flag_store.define(
            "prompt_evolution",
            enabled=False,
            description="Enable Thompson sampling over prompt variants",
        )
        flag_store.define(
            "goal_orchestration",
            enabled=False,
            description="Enable goal orchestration with tool loops and synthesis",
        )
        readiness.mark_ready("inquiry_engine_ready")

        # Phase 4: Strategy Loop + Elastic Scaling
        from qe.runtime.cognitive_agent_pool import CognitiveAgentPool
        from qe.runtime.strategy_evolver import ElasticScaler, StrategyEvolver

        def _engine_factory() -> InquiryEngine:
            """Factory closure capturing all cognitive components."""
            return InquiryEngine(
                episodic_memory=_episodic_memory,
                context_curator=_context_curator,
                metacognitor=_metacognitor,
                epistemic_reasoner=_epistemic_reasoner,
                dialectic_engine=_dialectic_engine,
                persistence_engine=_persistence_engine,
                insight_crystallizer=_insight_crystallizer,
                question_generator=QuestionGenerator(
                    model=fast_model, prompt_registry=_prompt_registry,
                ),
                hypothesis_manager=HypothesisManager(
                    belief_store=_bayesian_belief, model=balanced_model,
                    prompt_registry=_prompt_registry,
                ),
                question_store=_question_store,
                procedural_memory=_procedural_memory,
                bus=bus,
            )

        from qe.models.arena import ArenaConfig
        from qe.runtime.competitive_arena import CompetitiveArena

        _competitive_arena = CompetitiveArena(bus=bus, config=ArenaConfig())
        _cognitive_pool = CognitiveAgentPool(
            bus=bus,
            max_agents=5,
            engine_factory=_engine_factory,
            arena=_competitive_arena,
        )
        _elastic_scaler = ElasticScaler(
            agent_pool=_cognitive_pool,
            budget_tracker=_supervisor.budget_tracker,
        )
        _strategy_evolver = StrategyEvolver(
            agent_pool=_cognitive_pool,
            procedural_memory=_procedural_memory,
            bus=bus,
            elastic_scaler=_elastic_scaler,
            budget_tracker=_supervisor.budget_tracker,
        )
        flag_store.define(
            "multi_agent_mode",
            enabled=False,
            description="Use CognitiveAgentPool for parallel multi-agent inquiry",
        )
        flag_store.define(
            "competitive_arena",
            enabled=False,
            description="Enable agent-vs-agent competitive verification in multi-agent mode",
        )

        # Auto-populate pool with diverse strategies (when multi-agent enabled)
        if flag_store.is_enabled("multi_agent_mode") and _cognitive_pool is not None:
            from qe.runtime.strategy_models import DEFAULT_STRATEGIES
            strategies = list(DEFAULT_STRATEGIES.values())
            for strat in strategies[:_cognitive_pool._max_agents]:
                try:
                    await _cognitive_pool.spawn_agent(
                        specialization=strat.name,
                        model_tier=strat.preferred_model_tier,
                        strategy=strat,
                    )
                except RuntimeError:
                    break  # Pool at capacity

        await _strategy_evolver.start()
        readiness.mark_ready("strategy_loop_ready")

        # Knowledge Loop — background consolidation (episodic → semantic)
        from qe.runtime.knowledge_loop import KnowledgeLoop

        _knowledge_loop = KnowledgeLoop(
            episodic_memory=_episodic_memory,
            belief_store=_bayesian_belief,
            procedural_memory=_procedural_memory,
            bus=bus,
            model=fast_model,
        )
        _knowledge_loop.start()
        flag_store.define(
            "knowledge_consolidation",
            enabled=False,
            description="Enable background knowledge consolidation loop",
        )
        readiness.mark_ready("knowledge_loop_ready")

        # Inquiry Bridge — cross-loop glue
        from qe.runtime.inquiry_bridge import InquiryBridge

        _inquiry_bridge = InquiryBridge(
            bus=bus,
            episodic_memory=_episodic_memory,
            strategy_evolver=_strategy_evolver,
            knowledge_loop=_knowledge_loop,
        )
        _inquiry_bridge.start()

        _chat_service = ChatService(
            substrate=_substrate,
            bus=bus,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
        )
        _goal_store = GoalStore(_substrate.belief_ledger._db_path)
        _planner = PlannerService(
            bus=bus,
            substrate=_substrate,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
        )
        _dispatcher = Dispatcher(
            bus=bus,
            goal_store=_goal_store,
            agent_pool=_supervisor.agent_pool,
            working_memory=_supervisor.working_memory,
        )

        if not _extra_routes_registered:
            register_goal_routes(
                app=app,
                planner=_planner,
                dispatcher=_dispatcher,
                goal_store=_goal_store,
            )
            _extra_routes_registered = True

        # Start the executor service that processes dispatched tasks
        _executor = ExecutorService(
            bus=bus,
            substrate=_substrate,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
            agent_id="executor_default",
            tool_registry=_tool_registry,
            tool_gate=_tool_gate,
            workspace_manager=_workspace_manager,
        )
        await _executor.start()

        # Phase 4: Wire verification gate between executor and dispatcher
        _db_path = _substrate.belief_ledger._db_path
        _failure_kb = FailureKnowledgeBase(_db_path)
        _verification_svc = VerificationService(substrate=_substrate)
        _recovery = RecoveryOrchestrator(failure_kb=_failure_kb, bus=bus)
        _verification_gate = VerificationGate(
            bus, _verification_svc, _recovery, _failure_kb
        )
        await _verification_gate.start()

        # Dispatcher receives verified/recovered results instead of raw results
        async def _on_verified_result(envelope: Envelope) -> None:
            from qe.models.goal import SubtaskResult as _SR

            result = _SR.model_validate(envelope.payload)
            await _dispatcher.handle_subtask_completed(result.goal_id, result)

        bus.subscribe("tasks.verified", _on_verified_result)
        bus.subscribe("tasks.recovered", _on_verified_result)

        # Goal Synthesizer: aggregates subtask results on goal completion
        from qe.services.synthesizer import GoalSynthesizer

        _synthesizer = GoalSynthesizer(
            bus=bus,
            goal_store=_goal_store,
            dialectic_engine=_dialectic_engine,
            model=balanced_model,
            budget_tracker=_supervisor.budget_tracker,
        )
        await _synthesizer.start()

        # Register the default executor as an agent in the pool
        from qe.runtime.agent_pool import AgentRecord

        _supervisor.agent_pool.register(
            AgentRecord(
                agent_id="executor_default",
                service_id="executor",
                capabilities={"web_search", "code_exec"},
                task_types={
                    "research",
                    "analysis",
                    "fact_check",
                    "synthesis",
                    "document_generation",
                    "web_search",
                    "code_execution",
                },
                model_tier="balanced",
                max_concurrency=5,
            )
        )

        # Wire event log into substrate for MAGMA temporal/causal queries
        _substrate.set_event_log(_event_log)
        # Start Doctor health monitoring service
        _doctor = DoctorService(
            bus=bus,
            substrate=_substrate,
            supervisor=_supervisor,
            event_log=_event_log,
            budget_tracker=_supervisor.budget_tracker,
        )
        await _doctor.start()
        # Reconcile in-flight goals from previous run
        await _dispatcher.reconcile()

        readiness.mark_ready("services_subscribed")

        # Start channel adapters (Telegram, Slack, Email, Webhook)
        await _init_channels()
        readiness.mark_ready("channels_ready")

        readiness.mark_ready("supervisor_ready")
        log.info("QE API server started (engine running)")
    else:
        log.info("QE API server started (setup required — no API keys configured)")

    # Initialize EngramCache before yielding
    from qe.runtime.engram_cache import get_engram_cache

    _engram_cache = get_engram_cache()
    _cache_stats = _engram_cache.stats()
    log.info(
        "engram_cache.initialized exact=%d template=%d",
        _cache_stats.get("exact_entries", 0),
        _cache_stats.get("template_entries", 0),
    )

    try:
        yield
    finally:
        # Shutdown — EngramCache cleanup
        try:
            from qe.runtime.engram_cache import get_engram_cache as _get_cache

            _cache = _get_cache()
            cleared = _cache.clear()
            log.info("engram_cache.cleared count=%d", cleared)
        except Exception:
            log.debug("shutdown.engram_cache_clear_failed")

        # Shutdown — Inquiry Bridge
        try:
            if _inquiry_bridge is not None:
                await _inquiry_bridge.stop()
        except Exception:
            log.debug("shutdown.inquiry_bridge_stop_failed")
        _inquiry_bridge = None

        # Shutdown — Knowledge Loop
        try:
            if _knowledge_loop is not None:
                await _knowledge_loop.stop()
        except Exception:
            log.debug("shutdown.knowledge_loop_stop_failed")
        _knowledge_loop = None

        # Shutdown — Phase 4 strategy loop
        try:
            if _strategy_evolver is not None:
                await _strategy_evolver.stop()
        except Exception:
            log.debug("shutdown.strategy_evolver_stop_failed")

        _cognitive_pool = None
        _competitive_arena = None
        _strategy_evolver = None
        _elastic_scaler = None
        _episodic_memory = None

        # Shutdown — prompt mutator
        try:
            if _prompt_mutator is not None:
                await _prompt_mutator.stop()
        except Exception:
            log.debug("shutdown.prompt_mutator_stop_failed")
        _prompt_mutator = None

        # Shutdown — persist prompt registry
        try:
            if _prompt_registry is not None:
                await _prompt_registry.persist()
        except Exception:
            log.debug("shutdown.prompt_registry_persist_failed")
        _prompt_registry = None

        # Shutdown — clear Phase 3 refs
        _inquiry_engine = None

        # Shutdown — clear cognitive layer + tool refs
        try:
            BaseService._shared_episodic_memory = None
            BaseService._shared_bayesian_belief = None
            BaseService._shared_context_curator = None
            BaseService._shared_metacognitor = None
            BaseService._shared_epistemic_reasoner = None
            BaseService._shared_dialectic_engine = None
            BaseService._shared_persistence_engine = None
            BaseService._shared_insight_crystallizer = None
            BaseService._shared_tool_registry = None
            BaseService._shared_tool_gate = None
        except Exception:
            log.debug("shutdown.clear_shared_refs_failed")

        for adapter in _active_adapters:
            try:
                await adapter.stop()
            except Exception:
                log.exception("channel.stop_failed adapter=%s", adapter.channel_name)

        # Shutdown — Synthesizer
        try:
            if _synthesizer is not None:
                await _synthesizer.stop()
        except Exception:
            log.debug("shutdown.synthesizer_stop_failed")
        _synthesizer = None

        try:
            if _verification_gate:
                await _verification_gate.stop()
        except Exception:
            log.debug("shutdown.verification_gate_stop_failed")

        try:
            if _executor:
                await _executor.stop()
        except Exception:
            log.debug("shutdown.executor_stop_failed")

        try:
            if _doctor:
                await _doctor.stop()
        except Exception:
            log.debug("shutdown.doctor_stop_failed")

        try:
            if _supervisor:
                await _supervisor.stop()
        except Exception:
            log.debug("shutdown.supervisor_stop_failed")

        if relay_task:
            relay_task.cancel()
        if _supervisor_task:
            _supervisor_task.cancel()
        log.info("QE API server stopped")


app = FastAPI(
    title="Question Engine OS",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = (
    [o.strip() for o in os.environ["QE_CORS_ORIGINS"].split(",")]
    if os.environ.get("QE_CORS_ORIGINS")
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RateLimitMiddleware, rpm=120, burst=20)
app.add_middleware(AuthMiddleware)
try:
    from qe.api.endpoints.ingest import router as ingest_router

    app.include_router(ingest_router)
except RuntimeError as exc:
    # FastAPI raises RuntimeError when python-multipart is missing.
    log.warning("Ingest routes disabled: %s", exc)

# Serve dashboard
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Dashboard ───────────────────────────────────────────────────────────────


@app.get("/")
async def dashboard():
    """Serve the dashboard UI."""
    index = _static_dir / "index.html"
    if not index.exists():
        return JSONResponse(
            {"status": "ok", "message": "Question Engine is running. No dashboard UI found."},
        )
    return FileResponse(str(index))


# ── Setup Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/setup/status")
async def setup_status():
    """Return setup status: whether complete, configured providers, tier mapping, channels."""
    return {
        "complete": is_setup_complete(),
        "providers": get_configured_providers(),
        "tiers": get_current_tiers(),
        "channels": get_configured_channels(),
    }


@app.get("/api/setup/providers")
async def setup_providers():
    """Return the static list of supported providers."""
    return {
        "providers": [
            {
                "name": p["name"],
                "env_var": p["env_var"],
                "example_models": p["example_models"],
                "tier_defaults": p["tier_defaults"],
            }
            for p in PROVIDERS
        ],
    }


@app.post("/api/setup/save")
async def setup_save(body: dict[str, Any]):
    """Save provider API keys, tier assignments, and channel config.

    Expects:
        {
            "providers": {"OPENAI_API_KEY": "sk-...", ...},
            "tiers": {
                "fast": {"provider": "OpenAI", "model": "gpt-4o-mini"},
                "balanced": {"provider": "OpenAI", "model": "gpt-4o"},
                "powerful": {"provider": "Anthropic", "model": "claude-sonnet-4-20250514"}
            },
            "channels": {"TELEGRAM_BOT_TOKEN": "123:ABC...", ...}
        }
    """
    # Block setup changes after initial setup is complete
    if is_setup_complete():
        return JSONResponse(
            {
                "error": "Setup already complete. Use POST /api/setup/reconfigure to update.",
            },
            status_code=403,
        )

    providers = body.get("providers", {})
    tiers = body.get("tiers", {})
    channels = body.get("channels")

    if not providers and not tiers:
        return JSONResponse(
            {"error": "providers or tiers required"}, status_code=400
        )

    save_setup(providers=providers, tier_config=tiers, channels=channels)
    return {"status": "saved", "complete": is_setup_complete()}


@app.get("/api/setup/channels")
async def setup_channels():
    """Return the static list of available communication channels."""
    return {
        "channels": [
            {
                "id": ch["id"],
                "name": ch["name"],
                "description": ch["description"],
                "always_on": ch.get("always_on", False),
                "env_vars": [
                    {"key": ev["key"], "label": ev["label"], "type": ev["type"]}
                    for ev in ch.get("env_vars", [])
                ],
            }
            for ch in CHANNELS
        ],
    }


@app.post("/api/setup/reconfigure")
async def setup_reconfigure(body: dict[str, Any]):
    """Reconfigure providers, tiers, and channels after initial setup.

    Same payload shape as /api/setup/save but works after setup is complete.
    """
    providers = body.get("providers", {})
    tiers = body.get("tiers", {})
    channels = body.get("channels")

    if not providers and not tiers and not channels:
        return JSONResponse(
            {"error": "providers, tiers, or channels required"}, status_code=400
        )

    save_setup(providers=providers, tier_config=tiers, channels=channels)
    return {
        "status": "saved",
        "complete": is_setup_complete(),
        "note": "Restart required for channel changes to take effect.",
    }


# ── Webhooks ─────────────────────────────────────────────────────────────


@app.post("/api/webhooks/inbound")
async def inbound_webhook(request: Request):
    """Receive inbound webhook payloads from external systems."""
    if _notification_router is None:
        return JSONResponse({"error": "Channels not initialized"}, status_code=503)

    body = await request.json()
    headers = dict(request.headers)

    # Find the webhook adapter from active adapters
    from qe.channels.webhook import WebhookAdapter

    webhook = None
    for adapter in _active_adapters:
        if isinstance(adapter, WebhookAdapter):
            webhook = adapter
            break

    if webhook is None:
        return JSONResponse({"error": "Webhook adapter not available"}, status_code=503)

    result = await webhook.process_webhook(body, headers)
    if result is None:
        return JSONResponse({"error": "Invalid signature or rejected"}, status_code=403)

    # Route based on command field in the original payload
    command = body.get("command", "goal")
    topic_map = {
        "ask": "queries.asked",
        "status": "system.health.check",
    }
    topic = topic_map.get(command, "channel.message_received")

    get_bus().publish(
        Envelope(
            topic=topic,
            source_service_id="webhook",
            payload={
                "channel": "webhook",
                "user_id": result.get("user_id", ""),
                "text": result.get("sanitized_text", ""),
                "command": command,
            },
        )
    )

    return {"status": "received", "user_id": result.get("user_id", "")}


# ── Settings ─────────────────────────────────────────────────────────────


@app.get("/api/settings")
async def get_settings_endpoint():
    """Return runtime settings from config.toml."""
    return get_settings()


@app.post("/api/settings")
async def save_settings_endpoint(body: dict[str, Any]):
    """Save runtime settings and apply budget changes at runtime."""
    save_settings(body)

    # Apply budget limits at runtime if supervisor is running
    budget_vals = body.get("budget")
    if budget_vals and isinstance(budget_vals, dict) and _supervisor:
        _supervisor.budget_tracker.update_limits(
            monthly_limit_usd=budget_vals.get("monthly_limit_usd"),
            alert_at_pct=budget_vals.get("alert_at_pct"),
        )

    # Apply log level at runtime without restart
    runtime_vals = body.get("runtime")
    if runtime_vals and isinstance(runtime_vals, dict):
        if "log_level" in runtime_vals:
            update_log_level(runtime_vals["log_level"])

    get_audit_log().record("settings.update", resource="config", detail=body)
    return {"status": "saved"}


@app.post("/api/optimize/{genome_id}")
async def optimize_genome(genome_id: str):
    """Run DSPy prompt optimization on a genome using calibration data."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Find the genome path
    genome_path = None
    for p in _genome_paths():
        import tomllib
        with p.open("rb") as f:
            g = tomllib.load(f)
        if g.get("service_id") == genome_id:
            genome_path = p
            break

    if not genome_path:
        return JSONResponse(
            {"error": f"Genome '{genome_id}' not found"}, status_code=404
        )

    from qe.optimization.prompt_tuner import PromptTuner
    from qe.runtime.calibration import CalibrationTracker

    # Get or create calibration tracker
    cal = CalibrationTracker(
        db_path=_substrate.belief_ledger._db_path if _substrate else None
    )

    tuner = PromptTuner(cal, _substrate)
    result = await tuner.optimize_genome(genome_id, genome_path)
    return result.to_dict()


@app.post("/api/services/{service_id}/reset-circuit")
async def reset_circuit_breaker(service_id: str):
    """Reset a circuit-broken service so it can resume processing."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Check service exists
    found = any(
        s.blueprint.service_id == service_id
        for s in _supervisor.registry.all_services()
    )
    if not found:
        return JSONResponse(
            {"error": f"Service '{service_id}' not found"}, status_code=404
        )

    _supervisor._circuits.pop(service_id, None)
    _supervisor._pub_history.pop(service_id, None)
    get_audit_log().record(
        "circuit.reset", resource=f"service/{service_id}"
    )
    return {"status": "reset", "service_id": service_id}


# ── Dead Letter Queue ──────────────────────────────────────────────────────


@app.get("/api/dlq")
async def list_dlq(limit: int = 100):
    """List dead-letter queue entries."""
    bus = get_bus()
    return {
        "entries": bus.dlq_list(limit=limit),
        "count": bus.dlq_size(),
    }


@app.post("/api/dlq/{envelope_id}/replay")
async def replay_dlq(envelope_id: str):
    """Replay a dead-lettered envelope back into the bus."""
    bus = get_bus()
    ok = await bus.dlq_replay(envelope_id)
    if not ok:
        return JSONResponse(
            {"error": f"Envelope '{envelope_id}' not found in DLQ"},
            status_code=404,
        )
    return {"status": "replayed", "envelope_id": envelope_id}


@app.delete("/api/dlq")
async def purge_dlq():
    """Purge all entries from the dead-letter queue."""
    bus = get_bus()
    count = await bus.dlq_purge()
    return {"status": "purged", "count": count}


# ── Audit Trail ────────────────────────────────────────────────────────────


@app.get("/api/audit")
async def list_audit(
    action: str | None = None,
    actor: str | None = None,
    limit: int = 100,
):
    """Query the admin audit trail."""
    entries = get_audit_log().query(action=action, actor=actor, limit=limit)
    return {"entries": entries, "count": len(entries)}


# ── Metrics ───────────────────────────────────────────────────────────────


@app.get("/api/metrics")
async def metrics_snapshot():
    """Return full metrics snapshot: counters, histograms, gauges, SLOs."""
    return get_metrics().snapshot()


# ── Profiling ─────────────────────────────────────────────────────────────


@app.get("/api/profiling/inquiry")
async def profiling_inquiry():
    """Return profiling data for inquiry runs and system resources."""
    import resource
    import sys

    from qe.runtime.engram_cache import get_engram_cache as _get_cache

    rusage = resource.getrusage(resource.RUSAGE_SELF)

    # Engram cache stats
    try:
        cache_stats = _get_cache().stats()
    except Exception:
        cache_stats = {}

    # Episodic memory hot store size
    episodic_hot_size = 0
    if BaseService._shared_episodic_memory is not None:
        try:
            episodic_hot_size = len(BaseService._shared_episodic_memory._hot_store)
        except Exception:
            pass

    # Belief store status
    belief_status = "unavailable"
    if BaseService._shared_bayesian_belief is not None:
        belief_status = "available"

    # Build response with profiling store data
    store_data = _inquiry_profiling_store.to_dict()

    return {
        "phase_timings": _last_inquiry_profile,
        "last_inquiry": store_data["last_inquiry"],
        "history_count": store_data["history_count"],
        "percentiles": store_data["percentiles"],
        "process": {
            "rss_bytes": rusage.ru_maxrss,
            "python_version": sys.version,
        },
        "engram_cache": cache_stats,
        "components": {
            "episodic_hot_store_size": episodic_hot_size,
            "belief_store_status": belief_status,
        },
    }


# ── Prompt Evolution ───────────────────────────────────────────────────────


@app.get("/api/prompts/stats")
async def prompt_stats():
    """Return prompt evolution registry status and per-slot stats."""
    if _prompt_registry is None:
        return {"enabled": False, "slots": 0}
    return _prompt_registry.status()


@app.get("/api/knowledge/status")
async def knowledge_loop_status():
    """Return knowledge loop status."""
    if _knowledge_loop is None:
        return {"running": False}
    return _knowledge_loop.status()


@app.get("/api/bridge/status")
async def inquiry_bridge_status():
    """Return inquiry bridge status."""
    if _inquiry_bridge is None:
        return {"running": False}
    return _inquiry_bridge.status()


@app.get("/api/prompts/mutator/status")
async def prompt_mutator_status():
    """Return prompt mutator status."""
    if _prompt_mutator is None:
        return {"running": False}
    return _prompt_mutator.status()


@app.get("/api/prompts/slots/{slot_key}")
async def prompt_slot_detail(slot_key: str):
    """Return detailed stats for a specific prompt slot."""
    if _prompt_registry is None:
        return {"error": "prompt registry not initialized"}
    stats = _prompt_registry.get_slot_stats(slot_key)
    if not stats:
        return {"error": f"slot '{slot_key}' not found", "variants": []}
    return {"slot_key": slot_key, "variants": stats}


# ── Operational Observability ──────────────────────────────────────────────


@app.get("/api/pool/status")
async def pool_status():
    """Return cognitive agent pool status."""
    if _cognitive_pool is None:
        return JSONResponse({"error": "Cognitive pool not initialized"}, status_code=503)
    return _cognitive_pool.pool_status()


@app.get("/api/arena/status")
async def arena_status():
    """Return competitive arena status and Elo rankings."""
    if _competitive_arena is None:
        return {"enabled": False, "rankings": []}
    return _competitive_arena.status()


@app.get("/api/strategy/snapshots")
async def strategy_snapshots():
    """Return strategy evolver snapshots and current strategy."""
    if _strategy_evolver is None:
        return JSONResponse({"error": "Strategy evolver not initialized"}, status_code=503)
    return {
        "current_strategy": _strategy_evolver._current_strategy,
        "scaling_profile": (
            _elastic_scaler.current_profile_name() if _elastic_scaler else None
        ),
        "snapshots": [s.model_dump() for s in _strategy_evolver.get_snapshots()],
    }


@app.get("/api/flags")
async def list_flags():
    """List all feature flags and stats."""
    store = get_flag_store()
    return {
        "flags": store.list_flags(),
        "stats": store.stats(),
    }


@app.get("/api/flags/evaluations")
async def flag_evaluations(limit: int = 100):
    """Return recent flag evaluation log."""
    store = get_flag_store()
    evaluations = store.evaluation_log(limit=limit)
    return {
        "evaluations": evaluations,
        "count": len(evaluations),
    }


@app.get("/api/flags/{flag_name}")
async def get_flag(flag_name: str):
    """Get a single feature flag state."""
    flag = get_flag_store().get(flag_name)
    if flag is None:
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    return flag.to_dict()


@app.post("/api/flags/{flag_name}/enable")
async def enable_flag(flag_name: str):
    """Enable a feature flag at runtime."""
    if not get_flag_store().enable(flag_name):
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    get_audit_log().record("flag.enabled", resource=f"flag/{flag_name}")
    return {"status": "enabled", "flag_name": flag_name}


@app.post("/api/flags/{flag_name}/disable")
async def disable_flag(flag_name: str):
    """Disable a feature flag at runtime."""
    if not get_flag_store().disable(flag_name):
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    get_audit_log().record("flag.disabled", resource=f"flag/{flag_name}")
    return {"status": "disabled", "flag_name": flag_name}


@app.get("/api/episodic/status")
async def episodic_status():
    """Return episodic memory status overview."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    status = _episodic_memory.status()
    warm = await _episodic_memory.warm_count()
    return {
        "hot_entries": status["hot_entries"],
        "max_hot": status["max_hot"],
        "warm_entries": warm,
    }


@app.get("/api/episodic/search")
async def episodic_search(
    query: str = "",
    top_k: int = 10,
    goal_id: str | None = None,
    episode_type: str | None = None,
    time_window_hours: float | None = None,
):
    """Search episodic memory by keyword + recency."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    if not query:
        return JSONResponse({"error": "query parameter is required"}, status_code=400)
    episodes = await _episodic_memory.recall(
        query, top_k=top_k, time_window_hours=time_window_hours,
        goal_id=goal_id, episode_type=episode_type,
    )
    return {
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


@app.get("/api/episodic/goal/{goal_id}")
async def episodic_goal(goal_id: str, top_k: int = 20):
    """Return episodes for a specific goal."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    episodes = await _episodic_memory.recall_for_goal(goal_id, top_k=top_k)
    return {
        "goal_id": goal_id,
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


@app.get("/api/episodic/latest")
async def episodic_latest(limit: int = 20):
    """Return most recent episodes from hot store."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    episodes = _episodic_memory.get_latest(limit=limit)
    return {
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


# ── Bus Stats ─────────────────────────────────────────────────────────────


@app.get("/api/bus/stats")
async def bus_stats():
    """Return per-topic bus metrics: publish counts, latency, errors."""
    return get_bus_metrics().snapshot()


# ── Topology ──────────────────────────────────────────────────────────────


@app.get("/api/topology")
async def topology():
    """Return service dependency graph from blueprint declarations."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    all_topics: set[str] = set()

    for svc in _supervisor.registry.all_services():
        bp = svc.blueprint
        subs = bp.capabilities.bus_topics_subscribe
        pubs = bp.capabilities.bus_topics_publish
        all_topics.update(subs)
        all_topics.update(pubs)
        services.append({
            "service_id": bp.service_id,
            "display_name": bp.display_name,
            "subscribes": subs,
            "publishes": pubs,
        })

    return {
        "services": services,
        "topics": sorted(all_topics),
        "service_count": len(services),
        "topic_count": len(all_topics),
    }


# ── Event Replay ──────────────────────────────────────────────────────────


@app.post("/api/events/replay")
async def replay_events(body: dict[str, Any]):
    """Bulk replay historical events from the event log back into the bus.

    Accepts: {"topic": "...", "since": "ISO8601", "limit": 100}
    """
    if not _event_log:
        return JSONResponse(
            {"error": "Event log not ready"}, status_code=503
        )

    topic = body.get("topic")
    since_str = body.get("since")
    limit = body.get("limit", 100)

    since = None
    if since_str:
        since = datetime.fromisoformat(since_str)

    events = await _event_log.replay(since=since, topic=topic, limit=limit)

    bus = get_bus()
    replayed = 0
    for event in events:
        env = Envelope(
            envelope_id=event["envelope_id"],
            schema_version=event.get("schema_version") or "1.0",
            topic=event["topic"],
            source_service_id=event["source_service_id"],
            correlation_id=event.get("correlation_id"),
            causation_id=event.get("causation_id"),
            timestamp=datetime.fromisoformat(event["timestamp"]),
            payload=event["payload"],
            ttl_seconds=event.get("ttl_seconds"),
        )
        bus.publish(env)
        replayed += 1

    get_audit_log().record(
        "events.replayed",
        detail={"topic": topic, "since": since_str, "count": replayed},
    )
    return {"status": "replayed", "count": replayed}


# ── REST Endpoints ──────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/api/health/ready")
async def health_ready():
    """Readiness probe: returns 200 when fully initialized, 503 during startup."""
    readiness = get_readiness()
    status_data = readiness.to_dict()
    if readiness.is_ready:
        return status_data
    return JSONResponse(status_data, status_code=503)


@app.get("/api/health/live")
async def health_live():
    """Live health report from the Doctor service."""
    if not _doctor:
        return JSONResponse({"error": "Doctor service not running"}, status_code=503)

    report = _doctor.last_report
    if report is None:
        # First check hasn't run yet — run on demand
        report = await _doctor.run_all_checks()

    return report.model_dump(mode="json")


@app.get("/api/status")
async def status():
    """Overall engine status: services, budget, circuit breakers."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    for svc in _supervisor.registry.all_services():
        sid = svc.blueprint.service_id
        services.append({
            "service_id": sid,
            "display_name": svc.blueprint.display_name,
            "status": "alive" if svc._running else "stopped",
            "turn_count": svc._turn_count,
            "circuit_broken": sid in _supervisor._circuits,
        })

    return {
        "services": services,
        "budget": {
            "total_spend": _supervisor.budget_tracker.total_spend(),
            "remaining_pct": _supervisor.budget_tracker.remaining_pct(),
            "limit_usd": _supervisor.budget_tracker.monthly_limit_usd,
            "by_model": _supervisor.budget_tracker.spend_by_model(),
        },
        "pool": _cognitive_pool.pool_status() if _cognitive_pool else None,
        "strategy": {
            "current_strategy": (
                _strategy_evolver._current_strategy if _strategy_evolver else None
            ),
            "scaling_profile": (
                _elastic_scaler.current_profile_name() if _elastic_scaler else None
            ),
            "snapshots": (
                [s.model_dump() for s in _strategy_evolver.get_snapshots()]
                if _strategy_evolver else []
            ),
        },
        "flags": get_flag_store().stats(),
        "arena": _competitive_arena.status() if _competitive_arena else None,
        "bridge": _inquiry_bridge.status() if _inquiry_bridge else None,
        "knowledge_loop": _knowledge_loop.status() if _knowledge_loop else None,
    }


@app.post("/api/submit")
async def submit(body: dict[str, Any]):
    """Submit an observation to the engine."""
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="api",
        payload={"text": text},
    )

    get_bus().publish(envelope)

    # Also write to inbox for cross-process relay
    INBOX_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    inbox_file = INBOX_DIR / f"{envelope.envelope_id}.json"
    inbox_file.write_text(envelope.model_dump_json(), encoding="utf-8")

    return {"envelope_id": envelope.envelope_id, "status": "submitted"}


@app.post("/api/ask")
async def ask(body: dict[str, Any]):
    """Ask a question against the belief ledger."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    question = body.get("question", "")
    if not question:
        return JSONResponse(
            {"error": "question is required"}, status_code=400
        )

    result = await answer_question(question, _substrate)
    return result


@app.get("/api/claims")
async def list_claims(
    subject: str | None = None,
    include_superseded: bool = False,
):
    """List claims from the belief ledger."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claims = await _substrate.get_claims(
        subject_entity_id=subject,
        include_superseded=include_superseded,
    )
    return {
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    """Get a specific claim by ID."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claim = await _substrate.get_claim_by_id(claim_id)
    if not claim:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return claim.model_dump(mode="json")


@app.delete("/api/claims/{claim_id}")
async def retract_claim(claim_id: str):
    """Soft-retract a claim (mark as superseded by 'retracted')."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    retracted = await _substrate.retract_claim(claim_id)
    if not retracted:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return {"status": "retracted", "claim_id": claim_id}


@app.get("/api/entities")
async def list_entities():
    """List all entities with claim counts."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    entities = await _substrate.entity_resolver.list_entities()
    # Enrich with claim counts
    for ent in entities:
        claims = await _substrate.get_claims(
            subject_entity_id=ent["canonical_name"]
        )
        ent["claim_count"] = len(claims)
    return {"entities": entities, "count": len(entities)}


@app.get("/api/entities/{name}")
async def get_entity(name: str):
    """Get claims for a specific entity."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    canonical = await _substrate.entity_resolver.resolve(name)
    claims = await _substrate.get_claims(subject_entity_id=canonical)
    return {
        "canonical_name": canonical,
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@app.post("/api/entities/{name}/alias")
async def add_entity_alias(name: str, body: dict[str, Any]):
    """Add an alias for an entity."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    alias = body.get("alias", "")
    if not alias:
        return JSONResponse({"error": "alias is required"}, status_code=400)

    await _substrate.entity_resolver.add_alias(name, alias)
    return {"status": "alias_added", "canonical_name": name, "alias": alias}


@app.get("/api/events")
async def list_events(
    topic: str | None = None,
    limit: int = 100,
):
    """Query the durable event log."""
    if not _event_log:
        return JSONResponse({"error": "Event log not ready"}, status_code=503)

    events = await _event_log.replay(topic=topic, limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/api/hil/pending")
async def hil_pending():
    """List pending HIL approval requests."""
    pending_dir = Path("data/hil_queue/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    items = []
    for file in sorted(pending_dir.glob("*.json")):  # noqa: ASYNC240
        payload = json.loads(file.read_text(encoding="utf-8"))
        items.append(payload)
    return {"pending": items, "count": len(items)}


@app.post("/api/hil/{envelope_id}/approve")
async def hil_approve(envelope_id: str):
    """Approve a pending HIL request."""
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps({
            "decision": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    get_audit_log().record(
        "hil.approve", resource=f"envelope/{envelope_id}"
    )
    return {"status": "approved", "envelope_id": envelope_id}


@app.post("/api/hil/{envelope_id}/reject")
async def hil_reject(envelope_id: str, body: dict[str, Any] | None = None):
    """Reject a pending HIL request."""
    reason = (body or {}).get("reason", "rejected")
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps({
            "decision": "rejected",
            "reason": reason,
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    get_audit_log().record(
        "hil.reject",
        resource=f"envelope/{envelope_id}",
        detail={"reason": reason},
    )
    return {"status": "rejected", "envelope_id": envelope_id}


# ── Goals ───────────────────────────────────────────────────────────────────


@app.post("/api/goals")
async def submit_goal(body: dict[str, Any]):
    """Submit a new goal for decomposition and execution.

    When the inquiry_mode feature flag is enabled, routes through the
    InquiryEngine (v2 7-phase loop) instead of the v1 pipeline.
    """
    description = body.get("description", "").strip()
    if not description:
        return JSONResponse(
            {"error": "description is required"}, status_code=400
        )

    # v2 Multi-agent path
    flag_store = get_flag_store()
    if flag_store.is_enabled("multi_agent_mode"):
        try:
            if _cognitive_pool is not None:
                goal_id = f"goal_{uuid.uuid4().hex[:12]}"

                # Select strategy for this inquiry
                config = None
                if _strategy_evolver is not None:
                    from qe.runtime.strategy_models import strategy_to_inquiry_config
                    strategy = _strategy_evolver.select_strategy()
                    config = strategy_to_inquiry_config(strategy)

                # Competitive arena path: tournament verification
                if (
                    flag_store.is_enabled("competitive_arena")
                    and _competitive_arena is not None
                ):
                    from qe.models.arena import ArenaResult

                    result = await _cognitive_pool.run_competitive_inquiry(
                        goal_id=goal_id,
                        goal_description=description,
                        config=config,
                    )
                    if isinstance(result, ArenaResult):
                        return {
                            "arena_id": result.arena_id,
                            "goal_id": result.goal_id,
                            "winner_id": result.winner_id,
                            "sycophancy_detected": result.sycophancy_detected,
                            "match_count": len(result.matches),
                            "total_cost_usd": result.total_cost_usd,
                            "mode": "competitive_arena",
                        }
                    # Fell through to InquiryResult (< 2 agents)
                    if result is not None:
                        return {
                            "goal_id": result.goal_id,
                            "inquiry_id": result.inquiry_id,
                            "status": result.status,
                            "findings_summary": result.findings_summary[:1000],
                            "mode": "multi_agent",
                        }

                # Standard multi-agent path: parallel + merge
                results = await _cognitive_pool.run_parallel_inquiry(
                    goal_id=goal_id,
                    goal_description=description,
                    config=config,
                )
                if results:
                    merged = await _cognitive_pool.merge_results(results)
                    return {
                        "goal_id": merged.goal_id,
                        "inquiry_id": merged.inquiry_id,
                        "status": merged.status,
                        "termination_reason": merged.termination_reason,
                        "iterations": merged.iterations_completed,
                        "questions_answered": merged.total_questions_answered,
                        "insights_count": len(merged.insights),
                        "findings_summary": merged.findings_summary[:1000],
                        "mode": "multi_agent",
                    }
        except Exception:
            log.debug("submit_goal.multi_agent_fallthrough")

    # v2 Inquiry path (single agent)
    if flag_store.is_enabled("inquiry_mode") and _inquiry_engine is not None:
        global _last_inquiry_profile
        goal_id = f"goal_{uuid.uuid4().hex[:12]}"

        # Select strategy via Thompson sampling
        config = None
        if _strategy_evolver is not None:
            from qe.runtime.strategy_models import strategy_to_inquiry_config
            strategy = _strategy_evolver.select_strategy()
            config = strategy_to_inquiry_config(strategy)

        result = await _inquiry_engine.run_inquiry(
            goal_id=goal_id,
            goal_description=description,
            config=config,
        )
        _last_inquiry_profile = result.phase_timings
        _inquiry_profiling_store.record(result.phase_timings, result.duration_seconds)

        # Update readiness with inquiry status
        readiness = get_readiness()
        readiness.last_inquiry_status = result.status
        readiness.last_inquiry_at = time.monotonic()
        readiness.last_inquiry_duration_s = result.duration_seconds

        return {
            "goal_id": result.goal_id,
            "inquiry_id": result.inquiry_id,
            "status": result.status,
            "termination_reason": result.termination_reason,
            "iterations": result.iterations_completed,
            "questions_answered": result.total_questions_answered,
            "insights_count": len(result.insights),
            "findings_summary": result.findings_summary[:1000],
        }

    # v1 Pipeline path
    if not _planner or not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    state = await _planner.decompose(description)
    await _dispatcher.submit_goal(state)

    get_bus().publish(
        Envelope(
            topic="goals.submitted",
            source_service_id="api",
            correlation_id=state.goal_id,
            payload={
                "goal_id": state.goal_id,
                "description": description,
            },
        )
    )

    return {
        "goal_id": state.goal_id,
        "status": state.status,
        "subtask_count": len(state.subtask_states),
        "strategy": (
            state.decomposition.strategy if state.decomposition else ""
        ),
    }


@app.get("/api/goals")
async def list_goals(status: str | None = None):
    """List all goals with status."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    goals = await _goal_store.list_goals(status=status)
    return {
        "goals": [
            {
                "goal_id": g.goal_id,
                "description": g.description,
                "status": g.status,
                "subtask_count": len(g.subtask_states),
                "created_at": g.created_at.isoformat(),
                "completed_at": (
                    g.completed_at.isoformat()
                    if g.completed_at
                    else None
                ),
            }
            for g in goals
        ],
        "count": len(goals),
    }


@app.get("/api/goals/{goal_id}")
async def get_goal(goal_id: str):
    """Get goal detail with DAG and subtask states."""
    if not _dispatcher or not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Try in-memory first, then store
    state = _dispatcher.get_goal_state(goal_id)
    if not state:
        state = await _goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    return {
        "goal_id": state.goal_id,
        "description": state.description,
        "status": state.status,
        "subtask_states": state.subtask_states,
        "created_at": state.created_at.isoformat(),
        "completed_at": (
            state.completed_at.isoformat()
            if state.completed_at
            else None
        ),
        "decomposition": (
            state.decomposition.model_dump(mode="json")
            if state.decomposition
            else None
        ),
    }


@app.post("/api/goals/{goal_id}/pause")
async def pause_goal(goal_id: str):
    """Pause execution of a goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.pause_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not running"}, status_code=404
        )
    return {"status": "paused", "goal_id": goal_id}


@app.post("/api/goals/{goal_id}/resume")
async def resume_goal(goal_id: str):
    """Resume a paused goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.resume_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not paused"}, status_code=404
        )
    return {"status": "resumed", "goal_id": goal_id}


@app.post("/api/goals/{goal_id}/assign")
async def assign_goal_to_project(goal_id: str, body: dict[str, Any]):
    """Assign a goal to a project."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project_id = body.get("project_id")
    if not project_id:
        return JSONResponse(
            {"error": "project_id is required"}, status_code=400
        )

    state = await _goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    state.project_id = project_id
    await _goal_store.save_goal(state)
    return {"status": "assigned", "goal_id": goal_id, "project_id": project_id}


@app.post("/api/goals/{goal_id}/cancel")
async def cancel_goal(goal_id: str):
    """Cancel a running goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.cancel_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found"}, status_code=404
        )
    return {"status": "cancelled", "goal_id": goal_id}


# ── Projects ─────────────────────────────────────────────────────────────────


@app.get("/api/projects")
async def list_projects(status: str | None = None):
    """List all projects with goal counts."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    projects = await _goal_store.list_projects(status=status)
    result = []
    for p in projects:
        goals = await _goal_store.get_project_goals(p.project_id)
        result.append({
            **p.model_dump(mode="json"),
            "goal_count": len(goals),
            "completed_goals": sum(1 for g in goals if g.status == "completed"),
        })
    return {"projects": result, "count": len(result)}


@app.post("/api/projects")
async def create_project(body: dict[str, Any]):
    """Create a new project."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    from qe.models.goal import Project

    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    project = Project(
        name=name,
        description=body.get("description", ""),
        owner=body.get("owner", ""),
        tags=body.get("tags", []),
    )
    await _goal_store.save_project(project)
    return project.model_dump(mode="json")


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project detail with goals and metrics."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project = await _goal_store.get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)

    goals = await _goal_store.get_project_goals(project_id)
    metrics = await _goal_store.get_project_metrics(project_id)

    return {
        **project.model_dump(mode="json"),
        "goals": [
            {
                "goal_id": g.goal_id,
                "description": g.description,
                "status": g.status,
                "created_at": g.created_at.isoformat(),
                "completed_at": (
                    g.completed_at.isoformat() if g.completed_at else None
                ),
            }
            for g in goals
        ],
        "metrics": metrics,
    }


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, body: dict[str, Any]):
    """Update a project's fields."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project = await _goal_store.get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)

    from datetime import UTC, datetime

    if "name" in body:
        project.name = body["name"]
    if "description" in body:
        project.description = body["description"]
    if "owner" in body:
        project.owner = body["owner"]
    if "status" in body:
        project.status = body["status"]
    if "tags" in body:
        project.tags = body["tags"]
    project.updated_at = datetime.now(UTC)

    await _goal_store.save_project(project)
    return project.model_dump(mode="json")


# ── Chat ────────────────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat_rest(body: dict[str, Any]):
    """REST endpoint for chat (non-streaming fallback)."""
    if not _chat_service:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    message = body.get("message", "").strip()
    session_id = body.get("session_id")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    if not session_id:
        session_id = str(uuid.uuid4())

    response = await _chat_service.handle_message(session_id, message)
    return {
        "session_id": session_id,
        **response.model_dump(mode="json"),
    }


# ── WebSocket ───────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            log.debug("WS received: %s", data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """Per-session chat WebSocket with pipeline progress events."""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    if not _chat_service:
        await websocket.send_json({
            "type": "error",
            "error": "Engine not started. Please complete setup first.",
        })
        await websocket.close()
        return

    await websocket.send_json({
        "type": "session_init",
        "session_id": session_id,
    })

    # Track envelopes for pipeline progress forwarding
    tracked_envelopes: set[str] = set()

    async def _pipeline_forwarder(envelope: Envelope) -> None:
        """Forward pipeline events for envelopes this session is tracking."""
        correlation = envelope.correlation_id or envelope.causation_id
        if correlation in tracked_envelopes:
            try:
                await websocket.send_json({
                    "type": "pipeline_event",
                    "topic": envelope.topic,
                    "envelope_id": envelope.envelope_id,
                    "correlation_id": correlation,
                    "payload": envelope.payload,
                    "timestamp": envelope.timestamp.isoformat(),
                })
            except Exception:
                pass

    pipeline_topics = [
        "claims.proposed",
        "claims.committed",
        "claims.contradiction_detected",
    ]
    bus = get_bus()
    for topic in pipeline_topics:
        bus.subscribe(topic, _pipeline_forwarder)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                user_text = data.get("content", "").strip()
                if not user_text:
                    continue

                await websocket.send_json({"type": "typing", "active": True})

                response = await _chat_service.handle_message(
                    session_id, user_text
                )

                if response.tracking_envelope_id:
                    tracked_envelopes.add(response.tracking_envelope_id)

                await websocket.send_json({"type": "typing", "active": False})
                await websocket.send_json({
                    "type": "chat_response",
                    **response.model_dump(mode="json"),
                })

            elif msg_type == "track_envelope":
                envelope_id = data.get("envelope_id")
                if envelope_id:
                    tracked_envelopes.add(envelope_id)

    except WebSocketDisconnect:
        pass
    finally:
        for topic in pipeline_topics:
            bus.unsubscribe(topic, _pipeline_forwarder)
