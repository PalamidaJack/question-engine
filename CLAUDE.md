# Question Engine OS

## Project Overview

Multi-agent orchestration system being redesigned from a pipeline into a **cognitive system** with three nested loops (Inquiry, Knowledge, Strategy), four-tier memory, and a Cognitive Layer for genuine insight generation.

Full architecture plan: `.claude/plans/tranquil-hopping-harbor.md`

## Tech Stack

- Python 3.14, FastAPI, Pydantic v2, aiosqlite, litellm, instructor
- Virtual env: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
- LLM provider: Kilo Code (OpenRouter-compatible) at `https://kilo.ai/api/openrouter`
- Models: `openai/anthropic/claude-sonnet-4` (balanced), `openai/google/gemini-2.0-flash` (fast — note: Kilo Code model ID is now `google/gemini-2.0-flash-001`)
- Integration test model: `openai/anthropic/claude-3.5-haiku` (cheap, reliable structured output — Gemini models fail with nested Pydantic schemas via instructor tool calling)

## Key Directories

- `src/qe/api/app.py` — Main FastAPI app, lifespan, channel wiring, all endpoints
- `src/qe/channels/` — Channel adapters (telegram, slack, email, webhook) + notifications router
- `src/qe/services/` — Planner, Dispatcher, Executor, VerificationGate, Recovery, Checkpoint, Doctor, Chat
- `src/qe/services/inquiry/` — **NEW (v2)**: Dialectic Engine, Insight Crystallizer (Inquiry Loop components)
- `src/qe/bus/` — MemoryBus, event log, bus metrics
- `src/qe/substrate/` — Belief ledger (SQLite), cold storage, goal store, embeddings, BayesianBeliefStore
- `src/qe/models/` — Pydantic models (Envelope, Claim, GoalState, Genome Blueprint, Cognition)
- `src/qe/optimization/` — Prompt tuning (DSPy-based) + PromptRegistry (Thompson sampling over prompt variants)
- `src/qe/runtime/` — Service base, context curator, episodic memory, engram cache, metacognitor, epistemic reasoner, persistence engine
- `tests/unit/` — Unit tests (~50+ files)
- `tests/integration/` — Integration + E2E tests
- `config.toml` — Runtime config (model tiers, budget, logging)
- `.env` — API keys (gitignored)

## Running Tests & Linting

```bash
.venv/bin/pytest tests/ -m "not slow" --timeout=60 -q    # ~1550 unit/integration tests, all passing
.venv/bin/pytest tests/ -m slow --timeout=120 -v          # 6 real LLM integration tests (requires KILOCODE_API_KEY)
.venv/bin/ruff check src/ tests/ benchmarks/  # all clean
```

## Current State (2026-03-01)

~1550 tests pass (1038 v1 + 82 Phase 1 + 108 Phase 2 + 14 P1+2 wiring + 94 Phase 3 + 88 Phase 4 + 24 lint fixes + 33 Phase 5 + 23 Phase 6 + 6 real LLM integration + 47 Prompt Evolution), ruff clean. The 6 slow tests require KILOCODE_API_KEY and are excluded from default runs.

### v2 Redesign — Architecture Plan

Full plan at `.claude/plans/tranquil-hopping-harbor.md`. Key concepts:
- **Three Loops**: Inquiry (seconds-minutes), Knowledge (minutes-hours), Strategy (hours-days)
- **Four-Tier Memory**: Tier 0 Working (ContextCurator), Tier 1 Episodic (EpisodicMemory), Tier 2 Semantic (BayesianBeliefStore), Tier 3 Procedural (ProceduralMemory)
- **Cognitive Layer**: Metacognitor, Epistemic Reasoner, Dialectic Engine, Persistence Engine, Insight Crystallizer
- **Engram Cache**: Three-band (exact/template/full) replacing LLMCache

### v2 Phase 1: Memory Architecture — COMPLETE
All built, tested (82 tests), lint clean:
- `src/qe/substrate/bayesian_belief.py` — BayesianBeliefStore: Bayesian updating, evidence accumulation, knowledge graph, hypotheses. Migration: `src/qe/substrate/migrations/0012_bayesian_evidence.sql`
- `src/qe/runtime/context_curator.py` — ContextCurator: relevance-scored, goal-anchored Tier 0 working memory with drift detection
- `src/qe/runtime/episodic_memory.py` — EpisodicMemory: LRU in-memory hot store + SQLite warm overflow, recency-weighted search
- `src/qe/runtime/engram_cache.py` — EngramCache: three-band cache (exact SHA-256 / template similarity / full reasoning)
- Tests: `tests/unit/test_bayesian_belief.py`, `test_context_curator.py`, `test_episodic_memory.py`, `test_engram_cache.py`

### v2 Phase 2: Cognitive Layer — COMPLETE
All built, tested (108 tests), lint clean. These components make the system think outside the box:
- `src/qe/models/cognition.py` — ~25 Pydantic models for all cognitive reasoning outputs (ReasoningTrace, ApproachNode, EpistemicState, DialecticReport, CrystallizedInsight, etc.)
- `src/qe/runtime/metacognitor.py` — Self-awareness: capability registry, approach tree (tree not list — backtracks to siblings on failure), LLM-powered creative approach suggestion and tool combination
- `src/qe/runtime/epistemic_reasoner.py` — What we know vs. don't: absence detection, uncertainty assessment, known unknowns registry, surprise detection (integrates BayesianBeliefStore), blind spot warnings
- `src/qe/services/inquiry/dialectic.py` — Adversarial self-critique: devil's advocate (MUST argue against), perspective rotation (domain-aware: financial/tech/scientific/general), assumption surfacing (hidden > explicit), red team, full dialectic pipeline with confidence revision
- `src/qe/runtime/persistence_engine.py` — Determination: Why-Why-Why root cause (min 3 levels), 7 reframing strategies (inversion/proxy/stakeholder_shift/decompose_differently/implication/change_domain/temporal_shift), reframe cascade, lesson accumulation
- `src/qe/services/inquiry/insight.py` — Insight crystallizer: strict novelty gate, specific mechanism extraction, actionability scoring, cross-domain connections, provenance chains. Only novel + dialectic-survived findings become insights
- Tests: `tests/unit/test_cognition_models.py`, `test_metacognitor.py`, `test_epistemic_reasoner.py`, `test_dialectic_engine.py`, `test_persistence_engine.py`, `test_insight_crystallizer.py`

### v2 Phase 1+2 Wiring — COMPLETE
All Phase 1 memory and Phase 2 cognitive components wired into the running system (14 tests):
- `src/qe/bus/protocol.py` — 12 cognitive bus topics added (89 total)
- `src/qe/substrate/__init__.py` — Lazy `bayesian_belief` property on Substrate
- `src/qe/runtime/service.py` — LLMCache → EngramCache swap in `_call_llm()`; 8 shared class vars + classmethods for memory/cognitive components
- `src/qe/api/app.py` — All components initialized in lifespan, set on BaseService, cleaned up on shutdown
- Tests: `tests/unit/test_wiring.py`

### v2 Phase 3: Inquiry Loop + Knowledge Loop — COMPLETE
All built, tested (94 tests across 8 test files), lint clean:
- `src/qe/services/inquiry/schemas.py` — Pure Pydantic models: Question (tree structure), InvestigationResult, Reflection, InquiryConfig, InquiryState, InquiryResult
- `src/qe/services/inquiry/question_generator.py` — LLM-powered question generation + algorithmic prioritization (info_gain*0.4 + relevance*0.35 + novelty*0.25)
- `src/qe/services/inquiry/hypothesis.py` — HypothesisManager: POPPER-inspired hypothesis generation, falsification questions, Bayesian updating (belief store + local fallback), Bayes factor computation
- `src/qe/services/inquiry/engine.py` — InquiryEngine: 7-phase loop (Observe→Orient→Question→Prioritize→Investigate→Synthesize→Reflect), budget checking, 5 termination conditions, full cognitive component integration
- `src/qe/runtime/cognitive_agent.py` — CognitiveAgent model: agent identity, epistemic state, core memory (for Phase 4 multi-agent pools)
- `src/qe/runtime/procedural_memory.py` — Tier 3 memory: QuestionTemplate + ToolSequence with running average success tracking, SQLite + in-memory backends
- `src/qe/substrate/question_store.py` — SQLite question tree persistence: save, get, BFS tree ordering, status updates
- `src/qe/bus/schemas.py` — 10 inquiry payload schemas (all topics covered)
- `src/qe/bus/protocol.py` — 10 inquiry bus topics added (99 total)
- `src/qe/runtime/self_correction.py` — `evaluate_with_bayes_factor()` method: Bayes factor with log10 thresholds
- `src/qe/substrate/inference.py` — HypothesisTemplate: confirmed hypotheses boost claim confidence (1.3x), falsified weaken (0.5x)
- `src/qe/api/app.py` — All Phase 3 components wired in lifespan; `inquiry_mode` feature flag routes POST /api/goals to InquiryEngine
- Tests: `tests/unit/test_inquiry_schemas.py`, `test_question_generator.py`, `test_hypothesis_manager.py`, `test_inquiry_engine.py`, `test_cognitive_agent.py`, `test_procedural_memory.py`, `test_question_store.py`, `test_phase3_wiring.py`

### v2 Phase 4: Elastic Scaling + Strategy Loop — COMPLETE
All built, tested (88 tests across 6 test files), lint clean:
- `src/qe/runtime/routing_optimizer.py` — Added `BetaArm` (Beta-Binomial conjugate prior) and `thompson_select_model()` for true Thompson sampling model selection. Existing `select_model()` preserved for backward compat
- `src/qe/runtime/strategy_models.py` — Pydantic models: StrategyConfig, ScaleProfile, StrategyOutcome, StrategySnapshot. Predefined dicts: DEFAULT_STRATEGIES (breadth_first, depth_first, hypothesis_driven, iterative_refinement) and DEFAULT_PROFILES (minimal, balanced, aggressive)
- `src/qe/runtime/cognitive_agent_pool.py` — CognitiveAgentPool: multi-agent parallel inquiry execution with AgentSlot lifecycle, asyncio.gather fan-out with Semaphore concurrency control, result merging (insight dedup, cost sum, best-of-N findings)
- `src/qe/runtime/strategy_evolver.py` — StrategyEvolver: Thompson sampling strategy selection, outcome recording, background evaluation loop with automatic strategy switching on low performance. ElasticScaler: deterministic profile recommendation (budget/success-rate rules), spawn/retire to match target
- `src/qe/bus/protocol.py` — 6 new strategy/pool bus topics added (105 total)
- `src/qe/bus/schemas.py` — 6 new payload models (StrategySelected, StrategySwitchRequested, StrategyEvaluated, PoolScaleRecommended, PoolScaleExecuted, PoolHealthCheck) registered in TOPIC_SCHEMAS (31 total)
- `src/qe/api/app.py` — CognitiveAgentPool + StrategyEvolver + ElasticScaler wired in lifespan with engine_factory closure; `multi_agent_mode` feature flag for parallel inquiry; cleanup on shutdown
- Tests: `tests/unit/test_thompson_router.py`, `test_strategy_models.py`, `test_cognitive_agent_pool.py`, `test_strategy_evolver.py`, `test_elastic_scaler.py`, `test_phase4_wiring.py`

### v2 Phase 5: Integration + Polish — COMPLETE
All built, tested (33 tests across 3 new files + 9 new in existing), lint clean:
- `src/qe/services/inquiry/engine.py` — ProceduralMemory wired into Orient (template hints in findings_summary); PersistenceEngine wired into Reflect (root cause + reframe + lessons on drift); phase timing instrumentation (7 phases timed with start/stop monotonic, stats in result)
- `src/qe/services/inquiry/schemas.py` — `phase_timings: dict[str, list[float]]` on InquiryState; `phase_timings: dict[str, dict[str, float]]` on InquiryResult (count/total_s/avg_s/max_s per phase)
- `src/qe/api/app.py` — Lifespan hardened with try/finally (all cleanup runs on crash); EngramCache lifecycle (init before yield, clear in finally); each shutdown step wrapped in individual try/except; `GET /api/profiling/inquiry` endpoint (phase_timings, RSS, Python version, cache stats, component counts); `_last_inquiry_profile` updated after each inquiry run
- Tests: `tests/unit/test_inquiry_engine.py` (+9 new: procedural in orient, persistence in reflect, phase timings), `tests/unit/test_phase5_wiring.py` (4 tests: lifespan + profiling), `tests/unit/test_feature_flag_routing.py` (8 tests: flag defaults, toggle, precedence, rollout), `tests/integration/test_investment_e2e.py` (12 tests: full $50M lithium-ion investment walkthrough with procedural memory, persistence on drift, question persistence, bus event ordering, phase timings, budget termination)
- `tests/integration/conftest.py` — cognitive_stack fixture extended with real ProceduralMemory (pre-seeded finance templates) and real QuestionStore (SQLite); PersistenceEngine mock returns proper RootCauseAnalysis + ReframingResult objects

### v2 Phase 6: M1 Benchmarking + Production Hardening — COMPLETE
All built, tested (23 tests across 6 test files), lint clean:
- `benchmarks/inquiry_benchmark.py` — Standalone CLI benchmark harness: runs InquiryEngine N times with mock LLMs, reports per-phase timing percentiles (p50/p95/p99), RSS memory usage, throughput, cache stats. JSON output mode supported
- `src/qe/api/profiling.py` — InquiryProfilingStore: ring buffer (max 50 entries) storing phase_timings + duration per inquiry, with percentile aggregation across runs. Enhanced `GET /api/profiling/inquiry` endpoint with history_count, percentiles, last_inquiry fields
- `src/qe/services/inquiry/engine.py` — `_InquiryRateLimiter` (asyncio.Semaphore + token bucket for concurrency + RPM control); `_run_phase_with_retry()` with structured error classification and configurable retry; `_run_inquiry_loop()` extracted for `asyncio.wait_for` timeout wrapping; QEError catch in main loop
- `src/qe/services/inquiry/schemas.py` — `InquiryConfig` extended: `max_concurrent_inquiries`, `inquiry_rate_limit_rpm`, `inquiry_timeout_seconds` (10-3600s). `TerminationReason` extended: `rate_limited`, `timeout`
- `src/qe/errors.py` — `INQUIRY` ErrorDomain; `InquiryPhaseError` (retryable, 1000ms), `InquiryConfigError` (non-retryable), `InquiryTimeoutError` (non-retryable)
- `src/qe/runtime/readiness.py` — `inquiry_engine_ready`, `last_inquiry_status`, `last_inquiry_at`, `last_inquiry_duration_s` fields; `inquiry_healthy` property (healthy if engine ready and last failure >300s ago); `to_dict()` includes `inquiry` section
- `src/qe/api/app.py` — Profiling store wired; readiness updated after each inquiry run with status/timing
- Tests: `tests/unit/test_benchmark_harness.py` (3), `test_inquiry_profiling.py` (4), `test_inquiry_rate_limiter.py` (4), `test_inquiry_error_recovery.py` (4), `test_inquiry_timeout.py` (3), `test_health_check_inquiry.py` (5)

### Real LLM Integration Tests — COMPLETE
6 tests calling real LLMs through Kilo Code, verifying instructor+litellm structured output end-to-end:
- `src/qe/services/inquiry/engine.py` — Fixed two bugs in `_phase_synthesize`: `full_dialectic()` and `crystallize()` called without `goal_id`, and `evidence` passed as `list` instead of `str` (masked by mocks)
- `tests/integration/test_real_llm_inquiry.py` — 5 Layer-1 component tests (QuestionGenerator, Metacognitor, DialecticEngine, HypothesisManager, InsightCrystallizer) + 1 Layer-2 full InquiryEngine test. Uses `openai/anthropic/claude-3.5-haiku`. Marked `@pytest.mark.slow`, skipped when `KILOCODE_API_KEY` absent

### Prompt Evolution (Phases A-C) — COMPLETE
All built, tested (47 tests across 2 test files), lint clean. Thompson sampling over prompt variants for A/B testing and automated prompt evolution:
- `src/qe/substrate/migrations/0013_prompt_variants.sql` — SQLite schema: `prompt_variants` (variant_id, slot_key, content, alpha/beta arms, rollout_pct) + `prompt_outcomes` (success, quality_score, latency_ms)
- `src/qe/optimization/prompt_registry.py` — `PromptRegistry`: Thompson-samples among variants via `BetaArm`, `get_prompt()` hot path (no I/O), `record_outcome()` updates posteriors, SQLite persistence, `register_all_baselines()` registers 32 slots from all 7 components
- `src/qe/bus/protocol.py` — 4 new prompt topics (139 total): `prompt.variant_selected`, `prompt.outcome_recorded`, `prompt.variant_created`, `prompt.variant_deactivated`
- `src/qe/bus/schemas.py` — 4 new payload schemas (35 total) registered in `TOPIC_SCHEMAS`
- All 7 cognitive components updated with `prompt_registry` param, `_fallbacks` dict, `_get_prompt()` helper, and `record_outcome()` calls on LLM success/failure:
  - `dialectic.py` (8 slots: challenge, perspectives, assumptions, red_team × system/user)
  - `insight.py` (8 slots: novelty, mechanism, actionability, cross_domain × system/user)
  - `metacognitor.py` (4 slots: approach, tool_combo × system/user)
  - `epistemic_reasoner.py` (6 slots: absence, uncertainty, surprise × system/user)
  - `persistence_engine.py` (4 slots: root_cause, reframe × system/user)
  - `question_generator.py` (1 slot: generate.system)
  - `hypothesis.py` (1 slot: generate.system)
- `src/qe/api/app.py` — Registry initialized in lifespan, passed to all 7 components, `prompt_evolution` feature flag (disabled by default), `GET /api/prompts/stats` endpoint, shutdown persistence
- Phase D (PromptMutator: LLM-powered auto-generation of variants via rephrase/elaborate/simplify/restructure + auto-rollback) deferred
- Tests: `tests/unit/test_prompt_registry.py` (28 tests: models, disabled/enabled modes, Thompson sampling, rollout, deactivation, stats, SQLite round-trip, format fallback, bus events, baseline registration), `tests/unit/test_prompt_evolution_wiring.py` (19 tests: bus topics/schemas, per-component registry integration, feature flag)

### Next Steps
- Phase D: PromptMutator — LLM-powered auto-generation of prompt variants with auto-rollback on low performance

### v1 Recently Completed (pre-redesign)
- Phase 4: VerificationGate, RecoveryOrchestrator, CheckpointManager
- Multi-agent orchestration (planner, dispatcher, executor)
- Channel adapters (Telegram, Slack, Email, Webhook) with command routing
- Kilo Code LLM provider integration

## Important Patterns

- Kilo Code requires `litellm.register_model()` for its models — see `_configure_kilocode()` in app.py
- `OPENAI_API_BASE` env var must NOT have `/v1` suffix (Kilo Code path is `/api/openrouter/chat/completions`)
- Setting `OPENAI_API_BASE` at module import time in tests pollutes other tests — always use scoped fixtures with cleanup
- SQLite float values need `pytest.approx()` for comparison
- Bus topics are defined in `src/qe/bus/protocol.py`
- LLM structured output pattern: `instructor.from_litellm(litellm.acompletion)` + Pydantic response_model (see any service or cognitive component)
- Cognitive layer tests mock LLM via `patch("qe.runtime.metacognitor.instructor")` (or equivalent module path) + `AsyncMock` for `client.chat.completions.create`
- All cognitive components accept optional `episodic_memory`, `model`, and `prompt_registry` params for dependency injection and testability
- Prompt slot naming convention: `component.method.role` (e.g., `dialectic.challenge.user`, `insight.novelty.system`)
- Gemini models (via Kilo Code/OpenRouter) fail with instructor tool calling on nested `list[PydanticModel]` schemas — use Claude models for real LLM integration tests
- Real LLM integration tests use `@pytest.mark.slow` + `skipif(not KILOCODE_API_KEY)` — see `tests/integration/test_real_llm_inquiry.py`
