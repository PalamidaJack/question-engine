"""FastAPI application for Question Engine OS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from qe.api.endpoints.memory import register_memory_routes
from qe.api.endpoints.memory_ops import register_memory_ops_routes
from qe.api.middleware import AuthMiddleware, RateLimitMiddleware, RequestTimingMiddleware
from qe.api.profiling import InquiryProfilingStore
from qe.api.setup import (
    get_current_tiers,
    get_settings,
    is_setup_complete,
)
from qe.api.ws import ConnectionManager
from qe.bus import get_bus
from qe.bus.event_log import EventLog
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.optimization.prompt_registry import PromptRegistry, register_all_baselines
from qe.runtime.context_curator import ContextCurator
from qe.runtime.episodic_memory import EpisodicMemory
from qe.runtime.epistemic_reasoner import EpistemicReasoner
from qe.runtime.feature_flags import get_flag_store
from qe.runtime.logging_config import configure_from_config
from qe.runtime.metacognitor import Metacognitor
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
# NOTE: Globals used by endpoints (via get_app_globals()) or tests are kept here.
# Internal-only variables (used only within lifespan init/shutdown) are local to lifespan().
_supervisor: Supervisor | None = None
_substrate: Substrate | None = None
_event_log: EventLog | None = None
_chat_service: ChatService | None = None
_planner: PlannerService | None = None
_dispatcher: Dispatcher | None = None
_goal_store: GoalStore | None = None
_doctor: DoctorService | None = None
_notification_router = None
_active_adapters: list = []
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
_peer_registry = None
_scout_service = None
_scout_store = None
_harvest_service = None
_last_inquiry_profile: dict[str, Any] = {}
_inquiry_profiling_store = InquiryProfilingStore()
_extra_routes_registered = False

_mass_intelligence_store = None
_mass_intelligence_market_agent = None
_mass_intelligence_executor = None

# Shutdown-only globals (assigned in lifespan, read/cleared in _shutdown_services)
_mcp_bridge = None
_discovery_service = None
_synthesizer = None
_verification_gate = None
_executor = None

# Memory-tier globals (assigned in lifespan, read by memory_ops endpoints)
_context_curator = None
_procedural_memory = None

# Phase 5: New service globals
_profile_loader = None
_chat_store = None
_model_intelligence = None
_workflow_executor = None
_webhook_notifier = None

# Phase 1 enhancement globals
_unified_llm = None
_tool_metrics = None
_knowledge_graph = None

INBOX_DIR = Path("data/runtime_inbox")

_AGENT_ACCESS_MODES = {"strict", "balanced", "full"}


def _resolve_agent_access_mode(settings: dict[str, Any]) -> str:
    """Resolve agent access mode from persisted settings."""
    mode = settings.get("agent_access", {}).get("mode", "balanced")
    if mode not in _AGENT_ACCESS_MODES:
        return "balanced"
    return mode


def _workspace_root_for_mode(mode: str) -> Path:
    """Map agent access mode to workspace root path."""
    if mode == "full":
        return Path(os.environ.get("QE_FULL_ACCESS_ROOT", "/"))
    return Path("data/workspaces/chat")


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


async def _shutdown_services() -> None:
    """Gracefully shut down all active services and clear global references."""
    global _supervisor, _substrate, _event_log, _chat_service
    global _planner, _dispatcher, _goal_store, _doctor
    global _inquiry_engine
    global _cognitive_pool, _competitive_arena, _strategy_evolver, _prompt_mutator, _knowledge_loop
    global _inquiry_bridge
    global _elastic_scaler, _episodic_memory
    global _peer_registry
    global _scout_service, _scout_store, _harvest_service
    global _mass_intelligence_store, _mass_intelligence_market_agent, _mass_intelligence_executor
    global _prompt_registry
    global _active_adapters
    global _mcp_bridge, _discovery_service, _synthesizer, _verification_gate, _executor
    global _profile_loader, _chat_store, _model_intelligence
    global _workflow_executor, _webhook_notifier
    global _unified_llm, _tool_metrics, _knowledge_graph

    # Shutdown — EngramCache cleanup
    try:
        from qe.runtime.engram_cache import get_engram_cache as _get_cache
        _cache = _get_cache()
        cleared = _cache.clear()
        log.info("engram_cache.cleared count=%d", cleared)
    except Exception:
        log.debug("shutdown.engram_cache_clear_failed")

    # Helper to stop a service safely
    async def _stop_svc(svc, name):
        if svc:
            try:
                await svc.stop()
            except Exception:
                log.debug(f"shutdown.{name}_stop_failed")

    # Stop services in reverse dependency order
    if _mcp_bridge:
        try:
            await _mcp_bridge.stop()
        except Exception:
            log.debug("shutdown.mcp_bridge_stop_failed")

    await _stop_svc(_mass_intelligence_market_agent, "mass_intelligence_market_agent")

    if _mass_intelligence_store:
        try:
            await _mass_intelligence_store.close()
        except Exception:
            log.debug("shutdown.mass_intelligence_store_close_failed")

    await _stop_svc(_model_intelligence, "model_intelligence")
    await _stop_svc(_discovery_service, "discovery")
    await _stop_svc(_scout_service, "scout")
    await _stop_svc(_harvest_service, "harvest")

    if _chat_store:
        try:
            await _chat_store.close()
        except Exception:
            log.debug("shutdown.chat_store_close_failed")
    await _stop_svc(_inquiry_bridge, "inquiry_bridge")
    await _stop_svc(_knowledge_loop, "knowledge_loop")
    await _stop_svc(_strategy_evolver, "strategy_evolver")
    await _stop_svc(_prompt_mutator, "prompt_mutator")

    if _prompt_registry:
        try:
            await _prompt_registry.persist()
        except Exception:
            log.debug("shutdown.prompt_registry_persist_failed")

    # Clear shared refs
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

    await _stop_svc(_synthesizer, "synthesizer")
    await _stop_svc(_verification_gate, "verification_gate")
    await _stop_svc(_executor, "executor")
    await _stop_svc(_doctor, "doctor")
    await _stop_svc(_supervisor, "supervisor")

    # Clear globals
    _mass_intelligence_market_agent = None
    _mass_intelligence_executor = None
    _mass_intelligence_store = None
    _scout_service = None
    _scout_store = None
    _harvest_service = None
    _inquiry_bridge = None
    _knowledge_loop = None
    _cognitive_pool = None
    _competitive_arena = None
    _strategy_evolver = None
    _elastic_scaler = None
    _episodic_memory = None
    _prompt_mutator = None
    _prompt_registry = None
    _inquiry_engine = None
    _synthesizer = None
    _peer_registry = None
    _mcp_bridge = None
    _unified_llm = None
    _tool_metrics = None
    _knowledge_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the QE engine on app startup, shut down on teardown."""
    global _supervisor, _substrate, _event_log, _chat_service
    global _planner, _dispatcher, _goal_store, _doctor
    global _cognitive_pool, _competitive_arena, _strategy_evolver, _prompt_mutator, _knowledge_loop
    global _peer_registry
    global _inquiry_bridge
    global _elastic_scaler, _episodic_memory
    global _scout_service, _scout_store, _harvest_service
    global _mass_intelligence_store, _mass_intelligence_market_agent, _mass_intelligence_executor
    global _extra_routes_registered
    global _prompt_registry, _inquiry_engine
    global _mcp_bridge, _discovery_service, _synthesizer, _verification_gate, _executor
    global _context_curator, _procedural_memory
    global _profile_loader, _chat_store, _model_intelligence
    global _workflow_executor, _webhook_notifier

    settings = get_settings()
    configure_from_config(settings)

    # Initialize optional OpenTelemetry tracing if enabled in config.toml
    try:
        from qe.config import load_config
        from qe.runtime import otel as _otel

        _cfg = load_config(Path("config.toml"))
        if getattr(_cfg, "otel", None) and _cfg.otel.enabled:
            _otel.init_tracing(
                service_name=_cfg.otel.service_name,
                exporter=_cfg.otel.exporter,
                otlp_endpoint=_cfg.otel.otlp_endpoint,
            )
            log.info("otel.initialized exporter=%s", _cfg.otel.exporter)
    except Exception:
        log.debug("otel.init_failed (optional)")

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
        # Register extended memory operations (Phase 1)
        try:
            register_memory_ops_routes(app=app, memory_store=_memory_store)
        except Exception:
            log.exception("register_memory_ops_routes_failed")
        _extra_routes_registered = True

    readiness.mark_ready("substrate_ready")

    # Initialize Mass Intelligence services (always available, even before setup)
    from qe.services.mass_intelligence import MassIntelligenceExecutor, ModelMarketAgent
    from qe.substrate.model_market import ModelMarketStore

    _db_path = _substrate.belief_ledger._db_path
    _mass_intelligence_store = ModelMarketStore(db_path=_db_path)
    await _mass_intelligence_store.initialize()

    api_keys = {
        "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
        "groq": os.environ.get("GROQ_API_KEY", ""),
        "cerebras": os.environ.get("CEREBRAS_API_KEY", ""),
        "cohere": os.environ.get("COHERE_API_KEY", ""),
        "mistral": os.environ.get("MISTRAL_API_KEY", ""),
        "google": os.environ.get("GOOGLE_API_KEY", ""),
        "hyperbolic": os.environ.get("HYPERBOLIC_API_KEY", ""),
        "sambanova": os.environ.get("SAMBANOVA_API_KEY", ""),
        "scaleway": os.environ.get("SCALEWAY_API_KEY", ""),
        "cloudflare": os.environ.get("CLOUDFLARE_API_KEY", ""),
    }

    _mass_intelligence_market_agent = ModelMarketAgent(
        store=_mass_intelligence_store,
        poll_interval_seconds=900,
        api_keys=api_keys,
    )
    await _mass_intelligence_market_agent.start()

    _mass_intelligence_executor = MassIntelligenceExecutor(
        store=_mass_intelligence_store,
        market_agent=_mass_intelligence_market_agent,
        default_timeout_seconds=45.0,
        max_concurrent=20,
        api_keys=api_keys,
    )

    log.info("mass_intelligence.services_initialized")

    relay_task: asyncio.Task | None = None

    if is_setup_complete():
        _configure_kilocode()

        # Model Discovery — polls free providers for available models
        from qe.runtime.discovery.service import ModelDiscoveryService

        _discovery_service = ModelDiscoveryService(bus=bus)
        await _discovery_service.start()

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

        # Phase 1 Enhancements: ToolMetrics, UnifiedLLM, KnowledgeGraph
        from qe.runtime.llm import UnifiedLLM
        from qe.runtime.metrics import get_metrics
        from qe.runtime.tools import ToolMetrics
        from qe.substrate.knowledge_graph import KnowledgeGraph

        global _tool_metrics, _unified_llm, _knowledge_graph
        _tool_metrics = ToolMetrics()
        _unified_llm = UnifiedLLM(
            default_model=balanced_model,
            budget_tracker=_supervisor.budget_tracker,
            metrics=get_metrics(),
        )
        _knowledge_graph = KnowledgeGraph()

        # Tool Infrastructure
        from qe.runtime.tool_bootstrap import create_default_gate, create_default_registry
        from qe.runtime.tool_gate import SecurityPolicy
        from qe.runtime.workspace import WorkspaceManager
        from qe.tools.file_ops import set_elevated_root, set_workspace_root

        agent_access_mode = _resolve_agent_access_mode(settings)
        _tool_registry = create_default_registry(tool_metrics=_tool_metrics)
        _tool_gate = create_default_gate(policies=[
            SecurityPolicy(
                name="default",
                max_calls_per_goal=50,
                blocked_domains=["localhost", "127.0.0.1"],
            ),
        ])
        _workspace_manager = WorkspaceManager(base_dir="data/workspaces")
        workspace_root = _workspace_root_for_mode(agent_access_mode)
        workspace_root.mkdir(parents=True, exist_ok=True)
        set_workspace_root(workspace_root)
        # Set elevated root for sandbox_escape mode (project root)
        project_root = Path(
            os.environ.get(
                "QE_PROJECT_ROOT",
                str(Path(__file__).resolve().parent.parent.parent.parent),
            )
        )
        set_elevated_root(project_root)
        log.info(
            "agent_access.configured mode=%s workspace_root=%s elevated_root=%s",
            agent_access_mode,
            workspace_root,
            project_root,
        )
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
        from qe.runtime.peer_registry import PeerRegistry

        _peer_registry = PeerRegistry()
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

        # Guardrails pipeline (Phase 2)
        from qe.config import load_config
        from qe.runtime.guardrails import GuardrailsPipeline

        _guardrails_config = load_config(Path("config.toml")).guardrails
        _guardrails_pipeline = GuardrailsPipeline.default_pipeline(
            config=_guardrails_config, bus=bus,
        )
        # expose to module globals for endpoints
        globals().update({
            "_guardrails_pipeline": _guardrails_pipeline,
            "_guardrails_config": _guardrails_config,
        })

        # Inquiry Bridge — cross-loop glue
        from qe.runtime.inquiry_bridge import InquiryBridge

        _inquiry_bridge = InquiryBridge(
            bus=bus,
            episodic_memory=_episodic_memory,
            strategy_evolver=_strategy_evolver,
            knowledge_loop=_knowledge_loop,
        )
        _inquiry_bridge.start()

        # Innovation Scout — self-improving meta-agent (disabled by default)
        from qe.config import load_config
        from qe.services.scout import InnovationScoutService
        from qe.substrate.scout_store import ScoutStore as _ScoutStoreClass

        _scout_config = load_config(Path("config.toml")).scout
        _scout_store = _ScoutStoreClass(db_path=_db_path)
        await _scout_store.initialize()
        _scout_service = InnovationScoutService(
            bus=bus,
            scout_store=_scout_store,
            config=_scout_config,
            model=fast_model,
            balanced_model=balanced_model,
        )
        flag_store.define(
            "innovation_scout",
            enabled=False,
            description="Enable Innovation Scout background scouting and proposal generation",
        )
        await _scout_service.start()

        # Harvest Service — autonomous knowledge improvement via free models
        from qe.services.harvest import HarvestService

        _harvest_config = load_config(Path("config.toml")).harvest
        _harvest_service = HarvestService(
            bus=bus,
            substrate=_substrate,
            discovery=_discovery_service,
            mass_executor=_mass_intelligence_executor,
            episodic_memory=_episodic_memory,
            epistemic_reasoner=_epistemic_reasoner,
            procedural_memory=_procedural_memory,
            config=_harvest_config,
        )
        flag_store.define(
            "harvest_service",
            enabled=False,
            description="Enable Harvest Service for autonomous knowledge improvement",
        )
        await _harvest_service.start()

        # Phase 2-4 feature flags
        flag_store.define(
            "stable_prompt_prefix", enabled=False,
            description="Split system prompt into stable prefix + dynamic suffix for KV-cache opt",
        )
        flag_store.define(
            "tool_masking", enabled=False,
            description="Return all tools regardless of mode — ToolGate blocks at execution time",
        )
        flag_store.define(
            "recitation_pattern", enabled=False,
            description="Re-state original request every 4 iterations to prevent drift",
        )
        flag_store.define(
            "artifact_handles", enabled=False,
            description="Store large tool results as artifacts, inject handles into context",
        )
        flag_store.define(
            "pattern_breaking", enabled=False,
            description="Vary agent state format to prevent behavioral mimicry",
        )
        flag_store.define(
            "proactive_recall", enabled=False,
            description="Inject procedural memory suggestions into dynamic context",
        )
        flag_store.define(
            "task_aware_routing", enabled=False,
            description="Route LLM calls based on task classification",
        )
        flag_store.define(
            "parallel_tool_calls", enabled=False,
            description="Execute multiple tool calls concurrently",
        )
        flag_store.define(
            "chat_llm_recovery", enabled=False,
            description="Retry LLM calls with backoff and model escalation",
        )
        flag_store.define(
            "subagent_cache", enabled=False,
            description="Cache tool results to reduce redundant API calls",
        )
        flag_store.define(
            "llm_health_check", enabled=False,
            description="Include LLM connectivity in Doctor health checks",
        )

        # Phase 5: New system feature flags
        flag_store.define(
            "agent_profiles", enabled=True,
            description="Use profile-based system prompt assembly",
        )
        flag_store.define(
            "chat_persistence", enabled=True,
            description="Persist conversations to ChatStore",
        )
        flag_store.define(
            "model_intelligence", enabled=True,
            description="Enable Model Intelligence profiling service",
        )
        flag_store.define(
            "workflow_executor", enabled=False,
            description="Enable visual workflow execution engine",
        )
        flag_store.define(
            "multi_model_compare", enabled=False,
            description="Enable side-by-side multi-model comparison",
        )
        flag_store.define(
            "introspection_tools", enabled=True,
            description="Give the agent self-inspection tools",
        )

        # Phase 1 enhancement flags
        flag_store.define(
            "vector_embeddings", enabled=False,
            description="Vector embedding enhancements: re-embed, coverage metrics",
            category="knowledge",
            maturity="preview",
        )
        flag_store.define(
            "graph_knowledge_retrieval", enabled=False,
            description="Graph-based knowledge retrieval with entity traversal",
            category="knowledge",
            maturity="experimental",
        )
        flag_store.define(
            "tool_quality_metrics", enabled=False,
            description="Per-tool success rate, latency, and error tracking",
            category="observability",
            maturity="preview",
        )

        # Phase 2 enhancement flags
        flag_store.define(
            "token_budget_management", enabled=False,
            description="Section-level token budgets for context assembly",
            category="context",
            maturity="experimental",
        )
        flag_store.define(
            "adaptive_memory_weighting", enabled=False,
            description="Query-type-aware memory tier weighting",
            category="context",
            maturity="experimental",
        )
        flag_store.define(
            "workflow_checkpoints", enabled=False,
            description="Checkpoint/resume for workflow executions",
            category="orchestration",
            maturity="preview",
        )
        flag_store.define(
            "smart_failover", enabled=False,
            description="Multi-provider failover with circuit breaker",
            category="routing",
            maturity="experimental",
        )
        flag_store.define(
            "filesystem_artifacts", enabled=False,
            description="Store artifacts on filesystem instead of memory",
            category="system",
            maturity="preview",
        )
        flag_store.define(
            "bm25_hybrid_search", enabled=False,
            description="True BM25 scoring in hybrid search",
            category="knowledge",
            maturity="experimental",
        )
        flag_store.define(
            "fast_path_bypass", enabled=False,
            description="Skip tool loop for simple messages (greetings, acks)",
            category="performance",
            maturity="preview",
        )

        # Phase 3 enhancement flags
        flag_store.define(
            "cognitive_personas", enabled=False,
            description="Behavioral profiles for tool categories",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "orchestrator_handoff", enabled=False,
            description="Rule-based tool routing and handoff",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "specialist_profiles", enabled=False,
            description="Role-specific specialist agent profiles",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "verification_protocol", enabled=False,
            description="Verify cognitive tool outputs for quality",
            category="safety",
            maturity="experimental",
        )
        flag_store.define(
            "two_stage_review", enabled=False,
            description="Two-stage review of merged swarm results",
            category="safety",
            maturity="experimental",
        )
        flag_store.define(
            "structured_workflow", enabled=False,
            description="Design-Plan-Execute structured workflow",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "rpi_methodology", enabled=False,
            description="Research-Plan-Implement with confidence tags",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "auto_session_memory", enabled=False,
            description="Extract decisions/facts/preferences at session end",
            category="memory",
            maturity="experimental",
        )
        flag_store.define(
            "constraint_guardrails", enabled=False,
            description="Per-session constraint limits (tool calls, cost, tokens, domains)",
            category="safety",
            maturity="experimental",
        )

        # Phase 4 enhancement flags
        flag_store.define(
            "belief_clustering", enabled=False,
            description="Cluster similar claims and detect causal chains",
            category="knowledge",
            maturity="experimental",
        )
        flag_store.define(
            "contradiction_cascade", enabled=False,
            description="Blast radius analysis for claim retraction",
            category="knowledge",
            maturity="experimental",
        )
        flag_store.define(
            "self_learning_routing", enabled=False,
            description="EMA-tracked self-learning model routing",
            category="routing",
            maturity="experimental",
        )
        flag_store.define(
            "domain_swarms", enabled=False,
            description="Domain-specialized parallel agent swarms",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "subagent_context_isolation", enabled=False,
            description="Isolated context per swarm sub-agent",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "swarm_consensus", enabled=False,
            description="Voting and agreement protocols for swarm results",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "context_health_check", enabled=False,
            description="Detect context degradation (lost-in-middle, scatter)",
            category="context",
            maturity="experimental",
        )
        flag_store.define(
            "tiered_context_loading", enabled=False,
            description="L0/L1/L2 tiered context loading for token savings",
            category="context",
            maturity="experimental",
        )
        flag_store.define(
            "knowledge_filesystem", enabled=False,
            description="Hierarchical tree view over knowledge claims",
            category="knowledge",
            maturity="experimental",
        )
        flag_store.define(
            "task_scheduler", enabled=False,
            description="Cron-based task scheduling",
            category="system",
            maturity="experimental",
        )

        # Phase 5 enhancement flags
        flag_store.define(
            "artifact_system", enabled=False,
            description="4-tier artifact system (instructions/prompts/agents/skills)",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "composable_skills", enabled=False,
            description="DAG-based composable workflow skills",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "skill_chaining", enabled=False,
            description="Multi-skill chaining with data passing",
            category="agent",
            maturity="experimental",
        )
        flag_store.define(
            "skill_catalog", enabled=False,
            description="Catalog of external skills with install/uninstall",
            category="system",
            maturity="experimental",
        )
        flag_store.define(
            "progressive_tool_loading", enabled=False,
            description="Load tool schemas by detected intent",
            category="performance",
            maturity="experimental",
        )
        flag_store.define(
            "learning_loop", enabled=False,
            description="5-stage knowledge improvement cycle",
            category="intelligence",
            maturity="experimental",
        )
        flag_store.define(
            "bdi_tracking", enabled=False,
            description="Beliefs-Desires-Intentions mental state tracking",
            category="agent",
            maturity="experimental",
        )

        # Phase 6 enhancement flags
        flag_store.define(
            "mcp_server", enabled=False,
            description="Expose QE tools via MCP protocol",
            category="integration",
            maturity="experimental",
        )
        flag_store.define(
            "output_pipeline", enabled=False,
            description="Multi-channel output formatting",
            category="integration",
            maturity="experimental",
        )
        flag_store.define(
            "discord_integration", enabled=False,
            description="Discord channel adapter with /ask command",
            category="integration",
            maturity="experimental",
        )
        flag_store.define(
            "retrieval_trace_ui", enabled=False,
            description="Retrieval trace panel in chat UI",
            category="observability",
            maturity="experimental",
        )
        flag_store.define(
            "analyze_then_validate", enabled=False,
            description="Two-step analyze-then-validate retrieval",
            category="knowledge",
            maturity="experimental",
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

        # MCP Bridge: connect to external tool servers
        from qe.runtime.mcp_bridge import MCPBridge, MCPServerConfig

        _mcp_configs: list[MCPServerConfig] = []
        _mcp_config_file = "mcp_servers.json"
        if os.path.exists(_mcp_config_file):  # noqa: ASYNC240
            try:
                with open(_mcp_config_file) as _f:  # noqa: ASYNC230
                    _mcp_raw = json.load(_f)
                _mcp_configs = [MCPServerConfig(**c) for c in _mcp_raw]
            except Exception:
                log.warning("mcp_bridge.config_parse_failed", exc_info=True)
        # Inject project root into MCP filesystem server args
        _project_root = Path(os.environ.get("QE_PROJECT_ROOT", str(Path.cwd())))
        for _cfg in _mcp_configs:
            if _cfg.name == "filesystem" and str(_project_root) not in _cfg.args:
                _cfg.args.append(str(_project_root))
        _mcp_bridge = MCPBridge(configs=_mcp_configs, tool_registry=_tool_registry)
        if _mcp_configs:
            _mcp_tool_count = await _mcp_bridge.start()
            log.info("mcp_bridge.started servers=%d tools=%d", len(_mcp_configs), _mcp_tool_count)

        from qe.runtime.sanitizer import InputSanitizer

        _input_sanitizer = InputSanitizer()

        # Phase 5: Initialize new foundation services
        from qe.runtime.profiles import ProfileLoader
        from qe.substrate.chat_store import ChatStore

        _profile_loader = ProfileLoader(
            profiles_dir="profiles", active_profile="default",
        )
        _chat_store = ChatStore(db_path=_db_path.replace(
            ".db", "_chat.db",
        ) if isinstance(_db_path, str) else "data/chat_history.db")
        await _chat_store.initialize()

        from qe.services.model_intelligence import ModelIntelligenceService

        _model_intelligence = ModelIntelligenceService(
            bus=bus, discovery_service=_discovery_service,
        )
        await _model_intelligence.start()

        from qe.runtime.workflow_executor import WorkflowExecutor

        _workflow_executor = WorkflowExecutor(bus=bus)

        from qe.services.chat.runtime_context import RuntimeContext

        _runtime_context = RuntimeContext(
            workspace_root=workspace_root,
            project_root=project_root,
            mcp_bridge=_mcp_bridge,
            peer_registry=_peer_registry,
        )

        _chat_service = ChatService(
            substrate=_substrate,
            bus=bus,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
            fast_model=fast_model,
            inquiry_engine=_inquiry_engine,
            tool_registry=_tool_registry,
            tool_gate=_tool_gate,
            episodic_memory=_episodic_memory,
            cognitive_pool=_cognitive_pool,
            competitive_arena=_competitive_arena,
            planner=_planner,
            dispatcher=_dispatcher,
            goal_store=_goal_store,
            epistemic_reasoner=_epistemic_reasoner,
            dialectic_engine=_dialectic_engine,
            insight_crystallizer=_insight_crystallizer,
            knowledge_loop=_knowledge_loop,
            procedural_memory=_procedural_memory,
            access_mode=agent_access_mode,
            guardrails=_guardrails_pipeline,
            sanitizer=_input_sanitizer,
            router=globals().get("_auto_router"),
            recovery=_recovery if '_recovery' in dir() else None,
            profile_loader=_profile_loader,
            chat_store=_chat_store,
            model_intelligence=_model_intelligence,
            workflow_executor=_workflow_executor,
            runtime_context=_runtime_context,
        )

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
        # Store globals in state for modular routers
        app.state.notification_router = _notification_router
        app.state.active_adapters = _active_adapters
        app.state.scout_service = _scout_service
        app.state.scout_store = _scout_store
        app.state.harvest_service = _harvest_service
        app.state.cognitive_pool = _cognitive_pool
        app.state.strategy_evolver = _strategy_evolver
        app.state.elastic_scaler = _elastic_scaler
        app.state.competitive_arena = _competitive_arena
        app.state.inquiry_bridge = _inquiry_bridge
        app.state.knowledge_loop = _knowledge_loop
        app.state.guardrails_pipeline = _guardrails_pipeline
        app.state.guardrails_config = _guardrails_config
        app.state.bus = get_bus()
        app.state.ws_manager = ws_manager
        app.state.peer_registry = _peer_registry
        app.state.static_dir = _static_dir
        app.state.mass_intelligence_store = _mass_intelligence_store
        app.state.mass_intelligence_market_agent = _mass_intelligence_market_agent
        app.state.mass_intelligence_executor = _mass_intelligence_executor
        app.state.inquiry_profiling = _inquiry_profiling_store
        app.state.event_log = _event_log
        app.state.substrate = _substrate
        app.state.planner = _planner
        app.state.dispatcher = _dispatcher
        app.state.goal_store = _goal_store
        app.state.supervisor = _supervisor
        app.state.chat_service = _chat_service
        app.state.profile_loader = _profile_loader
        app.state.chat_store = _chat_store
        app.state.model_intelligence = _model_intelligence
        app.state.workflow_executor = _workflow_executor
        app.state.unified_llm = _unified_llm
        app.state.tool_metrics = _tool_metrics
        app.state.knowledge_graph = _knowledge_graph
        yield
    finally:
        await _shutdown_services()

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

# Initialize safe defaults for tests and any requests made before lifespan startup.
app.state.notification_router = None
app.state.active_adapters = []
app.state.scout_service = None
app.state.scout_store = None
app.state.harvest_service = None
app.state.cognitive_pool = None
app.state.strategy_evolver = None
app.state.elastic_scaler = None
app.state.competitive_arena = None
app.state.inquiry_bridge = None
app.state.knowledge_loop = None
app.state.guardrails_pipeline = None
app.state.guardrails_config = None
app.state.bus = get_bus()
app.state.ws_manager = ws_manager
app.state.peer_registry = None
app.state.static_dir = Path(__file__).parent / "static"
app.state.mass_intelligence_store = None
app.state.mass_intelligence_market_agent = None
app.state.mass_intelligence_executor = None
app.state.inquiry_profiling = _inquiry_profiling_store
app.state.event_log = None
app.state.substrate = None
app.state.planner = None
app.state.dispatcher = None
app.state.goal_store = None
app.state.supervisor = None
app.state.chat_service = None
app.state.profile_loader = None
app.state.chat_store = None
app.state.model_intelligence = None
app.state.workflow_executor = None
app.state.webhook_notifier = None

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


# ── Router Registration ─────────────────────────────────────────────────────

from qe.api.endpoints.a2a_router import playground_router  # noqa: E402
from qe.api.endpoints.a2a_router import router as a2a_router  # noqa: E402
from qe.api.endpoints.chat import router as chat_router  # noqa: E402
from qe.api.endpoints.communications import router as comms_router  # noqa: E402
from qe.api.endpoints.conversations import router as conv_router  # noqa: E402
from qe.api.endpoints.goals_v2 import projects_router  # noqa: E402
from qe.api.endpoints.goals_v2 import router as goals_v2_router  # noqa: E402
from qe.api.endpoints.guardrails import router as guardrails_router  # noqa: E402
from qe.api.endpoints.harvest import router as harvest_router  # noqa: E402
from qe.api.endpoints.knowledge import router as knowledge_router  # noqa: E402
from qe.api.endpoints.mass_intelligence import router as mass_intel_router  # noqa: E402
from qe.api.endpoints.memory_ops import router as memory_ops_router  # noqa: E402
from qe.api.endpoints.models_api import router as models_router  # noqa: E402
from qe.api.endpoints.profiles import router as profiles_router  # noqa: E402
from qe.api.endpoints.scout import router as scout_router  # noqa: E402
from qe.api.endpoints.setup import router as setup_router  # noqa: E402
from qe.api.endpoints.system import router as system_router  # noqa: E402
from qe.api.endpoints.telemetry import router as telemetry_router  # noqa: E402
from qe.api.endpoints.tools import router as tools_router  # noqa: E402
from qe.api.endpoints.webhooks import router as webhooks_router  # noqa: E402
from qe.api.endpoints.workflows import router as workflows_router  # noqa: E402

app.include_router(setup_router)
app.include_router(webhooks_router)
app.include_router(scout_router)
app.include_router(mass_intel_router)
app.include_router(harvest_router)
app.include_router(telemetry_router)
app.include_router(goals_v2_router)
app.include_router(projects_router)
app.include_router(knowledge_router)
app.include_router(system_router)
app.include_router(chat_router)
app.include_router(a2a_router)
app.include_router(playground_router)
app.include_router(guardrails_router)
app.include_router(memory_ops_router)
# Phase 5: New routers
app.include_router(profiles_router)
app.include_router(conv_router)
app.include_router(models_router)
app.include_router(workflows_router)
app.include_router(comms_router)
# Phase 1 enhancement routers
app.include_router(tools_router)
