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
- `src/qe/services/` — Planner, Dispatcher, Executor, Synthesizer, VerificationGate, Recovery, Checkpoint, Doctor, Chat
- `src/qe/services/inquiry/` — **NEW (v2)**: Dialectic Engine, Insight Crystallizer (Inquiry Loop components)
- `src/qe/bus/` — MemoryBus, event log, bus metrics
- `src/qe/substrate/` — Belief ledger (SQLite), cold storage, goal store, embeddings, BayesianBeliefStore
- `src/qe/models/` — Pydantic models (Envelope, Claim, GoalState, Genome Blueprint, Cognition, Arena)
- `src/qe/optimization/` — Prompt tuning (DSPy-based) + PromptRegistry (Thompson sampling over prompt variants) + PromptMutator (LLM-powered variant generation)
- `src/qe/runtime/` — Service base, context curator, episodic memory, engram cache, metacognitor, epistemic reasoner, persistence engine
- `tests/unit/` — Unit tests (~50+ files)
- `tests/integration/` — Integration + E2E tests
- `config.toml` — Runtime config (model tiers, budget, logging)
- `.env` — API keys (gitignored)

## Running Tests & Linting

```bash
.venv/bin/pytest tests/ -m "not slow" --timeout=60 -q    # ~1950 unit/integration tests, all passing
.venv/bin/pytest tests/ -m slow --timeout=120 -v          # 6 real LLM integration tests (requires KILOCODE_API_KEY)
.venv/bin/ruff check src/ tests/ benchmarks/  # all clean
```

## Current State (2026-03-02)

~2165 tests pass (1038 v1 + 82 Phase 1 + 108 Phase 2 + 14 P1+2 wiring + 94 Phase 3 + 88 Phase 4 + 24 lint fixes + 33 Phase 5 + 23 Phase 6 + 6 real LLM integration + 47 Prompt Evolution A-C + 49 Prompt Evolution D + 48 Knowledge Loop + 38 Loop Integration + 39 Strategy Wiring + 69 Goal Orchestration Pipeline + 21 Enhanced Onboarding + 84 Competitive Arena + 158 Innovation Scout), ruff clean. The 6 slow tests require KILOCODE_API_KEY and are excluded from default runs.

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
- Tests: `tests/unit/test_prompt_registry.py` (28 tests: models, disabled/enabled modes, Thompson sampling, rollout, deactivation, stats, SQLite round-trip, format fallback, bus events, baseline registration), `tests/unit/test_prompt_evolution_wiring.py` (19 tests: bus topics/schemas, per-component registry integration, feature flag)

### Prompt Evolution Phase D: PromptMutator — COMPLETE
All built, tested (49 tests across 2 test files), lint clean. LLM-powered auto-generation of prompt variants with performance monitoring:
- `src/qe/optimization/prompt_mutator.py` — `PromptMutator`: background evaluation loop (follows StrategyEvolver pattern) with three-phase per-slot logic: (1) auto-rollback low performers (mean < 0.3 after 20+ samples), (2) promote high performers (mean > 0.6, rollout 10%→50%), (3) mutate via LLM (instructor+litellm). Four mutation strategies: rephrase, elaborate, simplify, restructure (rotated). `MutatedPrompt` response model, format key validation (`{placeholder}` preservation), configurable thresholds. Gated by `prompt_evolution` feature flag
- `src/qe/optimization/prompt_registry.py` — Extended: `add_variant()` now accepts `strategy` param (passes through to bus event), new `promote_variant()` method for rollout percentage increases
- `src/qe/bus/protocol.py` — 2 new topics (141 total): `prompt.mutation_cycle_completed`, `prompt.variant_promoted`
- `src/qe/bus/schemas.py` — 2 new payload schemas (37 total): `PromptMutationCyclePayload`, `PromptVariantPromotedPayload`
- `src/qe/api/app.py` — PromptMutator wired in lifespan (start/stop), 2 new endpoints: `GET /api/prompts/mutator/status`, `GET /api/prompts/slots/{slot_key}`
- Tests: `tests/unit/test_prompt_mutator.py` (37 tests: strategies, models, format key extraction/validation, init/lifecycle, rollback/promote/feature flag/max variants/max mutations, LLM mutation with mock instructor, bus events, full integration cycle), `tests/unit/test_prompt_mutator_wiring.py` (12 tests: bus topics, schemas, registry extensions, promote_variant)

### Knowledge Loop — COMPLETE
All built, tested (48 tests across 2 test files), lint clean. The missing middle loop that consolidates short-term episodic findings into long-term semantic beliefs:
- `src/qe/runtime/knowledge_loop.py` — `KnowledgeLoop`: background consolidation service (minutes-hours timescale) with four-phase cycle: (1) Episode Scan — recalls recent episodes, groups by type (synthesis/claim_committed), (2) Pattern Detection & Belief Promotion — LLM-extracts structured claims from episodic clusters via instructor, promotes high-confidence patterns to BayesianBeliefStore, (3) Hypothesis Review — checks active hypotheses for auto-confirm (≥0.95) or auto-falsify (≤0.05), (4) Procedural Retirement — flags templates/sequences with success_rate < threshold and ≥10 observations. Follows StrategyEvolver lifecycle pattern (start/stop/status). `ConsolidationResult` + `ExtractedClaim` Pydantic models. Gated by `knowledge_consolidation` feature flag (disabled by default)
- `src/qe/bus/protocol.py` — 3 new topics (144 total): `knowledge.consolidation_completed`, `knowledge.belief_promoted`, `knowledge.hypothesis_updated`
- `src/qe/bus/schemas.py` — 3 new payload schemas (40 total): `KnowledgeConsolidationCompletedPayload`, `KnowledgeBeliefPromotedPayload`, `KnowledgeHypothesisUpdatedPayload`
- `src/qe/api/app.py` — KnowledgeLoop wired in lifespan (after strategy evolver), `knowledge_consolidation` feature flag, `GET /api/knowledge/status` endpoint, shutdown cleanup
- Tests: `tests/unit/test_knowledge_loop.py` (36 tests: models, init, lifecycle, feature flag gating, episode scan/grouping, pattern detection with mock LLM, belief promotion, hypothesis review, procedural retirement, bus events, full integration cycle), `tests/unit/test_knowledge_loop_wiring.py` (12 tests: bus topics, schemas, TOPIC_SCHEMAS registration, payload validation, schema defaults)

### Loop Integration — COMPLETE
All built, tested (38 tests across 2 new + 2 modified test files), lint clean. Connects the three nested loops (Inquiry, Knowledge, Strategy) that were previously functionally disconnected:
- `src/qe/runtime/inquiry_bridge.py` — `InquiryBridge`: lightweight cross-loop glue (~200 lines) that subscribes to 4 inquiry bus events and orchestrates feedback. On `inquiry.started`: stores observation episode. On `inquiry.completed`: stores synthesis episode, records `StrategyOutcome` on evolver (Thompson sampling arms update), triggers immediate knowledge consolidation. On `inquiry.failed`: stores failure episode, records negative outcome. On `inquiry.insight_generated`: stores synthesis episode with headline. Publishes `bridge.strategy_outcome_recorded` events. Lifecycle: `start()`/`stop()`/`status()`
- `src/qe/runtime/knowledge_loop.py` — Added `trigger_consolidation()`: event-driven consolidation (coexists with timer-based loop), called by InquiryBridge on inquiry completion
- `src/qe/services/inquiry/engine.py` — Fixed `recall_for_goal` bug: was passing `(goal_id, goal_description, limit=10)` but method signature is `(goal_id, top_k=20)`. Silently failed in try/except
- `src/qe/runtime/readiness.py` — Added 3 informational loop readiness fields (`cognitive_layer_ready`, `strategy_loop_ready`, `knowledge_loop_ready`) + `"loops"` section in `to_dict()`. `is_ready` unchanged (only core 4 startup phases gate readiness)
- `src/qe/bus/protocol.py` — 1 new topic (145 total): `bridge.strategy_outcome_recorded`
- `src/qe/bus/schemas.py` — 1 new payload schema (41 total): `BridgeStrategyOutcomePayload`
- `src/qe/api/app.py` — InquiryBridge wired in lifespan (after knowledge loop, before chat service), `knowledge_loop_ready` readiness mark, v2 channel routing (`inquiry_mode` flag routes channel messages to InquiryEngine before v1 fallback), `GET /api/bridge/status` endpoint, shutdown cleanup
- Tests: `tests/unit/test_inquiry_bridge.py` (24 tests: init, lifecycle, all 4 event handlers, strategy outcome recording, knowledge consolidation trigger, bus events, status, full lifecycle integration), `tests/unit/test_inquiry_bridge_wiring.py` (12 tests: bus topic/schema registration, schema validation/defaults, readiness fields, trigger_consolidation), `tests/unit/test_knowledge_loop.py` (+2 tests: trigger_consolidation runs/noop), `tests/unit/test_p1_features.py` (+1 topic in expected set)

### Strategy → Inquiry Wiring — COMPLETE
All built, tested (39 tests in 1 new test file), lint clean. Closes the forward path: strategy selection → inquiry config → execution → outcome recording with duration/cost:
- `src/qe/runtime/strategy_models.py` — Added `strategy_to_inquiry_config()`: maps StrategyConfig → InquiryConfig (question_batch_size→questions_per_iteration, max_depth→max_iterations, exploration_rate→inverse confidence_threshold, preferred_model_tier→model lookup)
- `src/qe/api/app.py` — Wired `select_strategy()` at all 3 inquiry call sites (single-agent, multi-agent, channel routing) behind respective feature flags; auto-populate agent pool with diverse DEFAULT_STRATEGIES at startup (behind `multi_agent_mode` flag); pass `elastic_scaler` + `budget_tracker` to StrategyEvolver
- `src/qe/runtime/cognitive_agent_pool.py` — Per-agent strategy-derived configs in `run_parallel_inquiry()`: each agent uses its slot's StrategyConfig converted to InquiryConfig when no shared config is passed
- `src/qe/runtime/strategy_evolver.py` — Added `elastic_scaler`/`budget_tracker` params to `__init__`; elastic scaling in `_evaluate()` calls `recommend_profile()`, applies on change, publishes `pool.scale_executed` event
- `src/qe/bus/schemas.py` — Added `duration_s` and `cost_usd` fields to `InquiryCompletedPayload`
- `src/qe/services/inquiry/engine.py` — Added `duration_s`/`cost_usd` to `inquiry.completed` event payload in `_finalize()`
- `src/qe/runtime/inquiry_bridge.py` — Enriched `StrategyOutcome` with `duration_s`/`cost_usd` from bus event payload, completing the full feedback loop: InquiryEngine → bus → InquiryBridge → StrategyOutcome → StrategyEvolver
- Tests: `tests/unit/test_strategy_wiring.py` (39 tests: strategy_to_inquiry_config mapping/clamping/defaults, select_strategy at 3 call sites, per-agent configs, elastic scaler wiring/profile change/budget tracker, auto-populate pool, outcome enrichment with duration/cost, full cycle integration)

### Goal Orchestration Pipeline — COMPLETE
All built, tested (69 tests across 5 new test files + 1 modified), lint clean. Connects the v2 cognitive architecture to end-to-end goal execution with tool loops, result synthesis, and intelligent retry:
- **Phase A: Tool Infrastructure Wiring** — `src/qe/api/app.py` initializes `ToolRegistry` (6 built-in tools: web_search, web_fetch, file_read, file_write, code_execute, browser_navigate) and `ToolGate` (SecurityPolicy with rate limiting, domain blocking, capability checks) in lifespan. Set on `BaseService` shared refs. `goal_orchestration` feature flag added. `tool_registry` passed to `InquiryEngine`
- **Phase B: ExecutorService Upgrade** — `src/qe/services/executor/service.py` upgraded with: (1) agentic `_run_tool_loop()` (10-iteration cap, LLM↔tool cycle following InquiryEngine._phase_investigate pattern), (2) `_validate_preconditions()`/`_validate_postconditions()` for contract enforcement (non_empty, data_available, min_length:N), (3) structured retry with `classify_error()` and `recovery_history` tracking, (4) task_type→capabilities mapping (web_search→{web_search}, code_execution→{code_execute}, research→{web_search}). New errors: `ExecutorContractError` (non-retryable), `ExecutorToolError` (retryable, 2s). New bus topic: `tasks.contract_violated`
- **Phase C: GoalSynthesizer** — `src/qe/services/synthesizer/service.py` (new service): subscribes to `goals.completed`, aggregates subtask results via LLM synthesis (instructor+litellm with `SynthesisInput` response model), optional dialectic review (revises confidence), stores `GoalResult` in `state.metadata["goal_result"]`. Models: `GoalResult` (summary, findings, confidence, provenance, recommendations, cost/latency totals), `SynthesisInput` (LLM response). New bus topics: `goals.synthesized`, `goals.synthesis_failed`
- **Phase D: End-to-End Wiring** — Synthesizer wired in app.py lifespan (after executor, before agent registration). ExecutorService receives `tool_registry`, `tool_gate`, `workspace_manager`. Two new API endpoints: `GET /api/goals/{goal_id}/progress` (subtask DAG with statuses, pct_complete), `GET /api/goals/{goal_id}/result` (GoalResult or 202 if pending). Dispatcher upgraded with intelligent retry: per-subtask retry counts from `contract.max_retries`, reset-to-pending-and-redispatch on failure, fail goal only when >50% subtasks failed or no dispatchable subtasks remain. Enriched `goals.completed` payload with `subtask_results_summary`
- `src/qe/bus/protocol.py` — 3 new topics (148 total): `tasks.contract_violated`, `goals.synthesized`, `goals.synthesis_failed`
- `src/qe/bus/schemas.py` — 3 new payload schemas (44 total): `TaskContractViolatedPayload`, `GoalSynthesizedPayload`, `GoalSynthesisFailedPayload`. `GoalCompletedPayload` extended with `subtask_results_summary`
- `src/qe/errors.py` — 2 new error classes: `ExecutorContractError`, `ExecutorToolError`
- Tests: `tests/unit/test_tool_wiring.py` (11), `tests/unit/test_executor_upgrade.py` (24), `tests/unit/test_goal_synthesizer.py` (19), `tests/unit/test_goal_orchestration_wiring.py` (11), `tests/integration/test_goal_pipeline_e2e.py` (4: full 2-subtask pipeline, retry→success, dialectic synthesis, metadata retrieval)

### Enhanced Onboarding Flow — COMPLETE
All built, tested (21 tests across 2 test files), lint clean. Adds channel configuration, hatching UX, and post-setup reconfiguration:
- `src/qe/api/setup.py` — `CHANNELS` constant (web/telegram/slack/email with env_var metadata), `get_configured_channels()` (checks env vars, masks passwords, returns config status), `save_setup()` extended with `channels: dict[str, str] | None` param to merge channel env vars into `.env`
- `src/qe/api/app.py` — `GET /api/setup/status` extended with `channels` field, `GET /api/setup/channels` (static channel list with descriptions/env_var metadata), `POST /api/setup/save` extended to accept `channels` in body (403 message updated to point to reconfigure), `POST /api/setup/reconfigure` (same payload shape as `/save` but works after initial setup, returns restart note)
- `src/qe/api/static/index.html` — `HatchingScreen` component (SVG progress ring with pulsing QE logo, polls `/api/health/ready` every 1s, maps 4 readiness phases to 0-100% progress, phase checklist with green checkmarks, 1.5s delay before transition). `SetupWizard` replaces `SetupScreen` with 2-step flow: Step 1 (LLM providers + API keys + Ollama toggle) → Step 2 (channel picker with toggle switches and credential inputs, Web Dashboard always-on with DEFAULT badge). `Root` component uses `setSetupComplete(true)` instead of `window.location.reload()` for smooth React transition. Settings tab: collapsible "LLM Providers & Channels" card with provider status dots + masked keys, editable tier models, API key update inputs, channel status, save via `POST /api/setup/reconfigure`
- Tests: `tests/unit/test_setup.py` (+16 tests: TestSaveSetupWithChannels — writes channel env vars / multiple / empty / None; TestGetConfiguredChannels — no env web only / telegram / slack needs both / slack full / email full / masked passwords; endpoints — status includes channels / channels list / reconfigure works / reconfigure rejects empty / 403 points to reconfigure), `tests/unit/test_onboarding.py` (5 tests: health/ready returns phases dict + ready flag, CHANNELS has expected IDs, web always_on, env_var keys match adapters)

### Competitive Agent Arena + Missing Features — COMPLETE
All built, tested (84 tests across 4 new + 4 modified test files), lint clean. Tournament-style agent-vs-agent verification competition, context compression via LLM summarization, and HIL integration tests:
- `src/qe/models/arena.py` — 8 Pydantic models: `AgentEloRating` (persistent Elo 1200 default, wins/losses/draws), `CrossExamination` (structured challenge with `xex_` ID prefix), `DefenseResponse` (rebuttals + concessions), `MatchJudgment` (`jdg_` prefix, 3-axis scoring: factual accuracy, evidence quality, novelty), `DivergenceCheck` (`div_` prefix, anti-sycophancy similarity assessment), `ArenaMatch` (`mtch_` prefix, full head-to-head match), `ArenaConfig` (enabled=False, max_rounds=2, divergence_threshold=0.3, budget_limit_usd=0.50, round_robin/single_elimination), `ArenaResult` (`arena_` prefix, full tournament output with sycophancy_detected flag)
- `src/qe/runtime/competitive_arena.py` — `CompetitiveArena`: tournament orchestration with 4 phases: (1) Divergence check — LLM classifies similarity between agent outputs, flags sycophancy if above threshold, (2) Cross-examination — each agent challenges the other's findings (forced-adversarial prompt: "CANNOT simply agree"), (3) Judgment — independent LLM judge scores on 3 axes, picks winner or declares draw, (4) Elo update — standard Elo formula (K=32) + Thompson sampling BetaArm updates. Anti-sycophancy: if agents are too similar, skips expensive cross-examination and uses majority-vote fallback. Budget enforcement: aborts early if `_total_cost_usd >= budget_limit_usd`. Match pairing: round_robin (all pairs) or single_elimination (sequential). `select_agents_for_arena()` uses Thompson sampling for exploration/exploitation. 4 LLM prompt templates (cross-examine, defense, judge, divergence) all using `instructor.from_litellm(litellm.acompletion)` for structured output
- `src/qe/runtime/context_manager.py` — `compress()` upgraded from TODO/pass to `async def compress(keep_recent=3)`: LLM summarization via instructor with `ConversationSummary` Pydantic model (summary, key_facts, open_questions), keeps last N messages verbatim, summarizes older messages, fallback to truncation on LLM failure. Uses cheapest model (`gemini-2.0-flash`)
- `src/qe/runtime/cognitive_agent_pool.py` — Added `arena: CompetitiveArena | None = None` param to `__init__`; new `run_competitive_inquiry()` method: runs parallel inquiry, then tournament if arena is set and ≥2 results, otherwise falls back to `merge_results()`
- `src/qe/runtime/inquiry_bridge.py` — Subscribes to `arena.tournament_completed` (5 topics total); new `_on_arena_tournament_completed()` handler stores synthesis episode with winner info and sycophancy flag
- `src/qe/runtime/strategy_models.py` — Added `arena_enabled: bool = False` to `StrategyConfig` (lets strategies opt into competitive verification; StrategyEvolver learns via Thompson sampling whether arena-enabled strategies produce better outcomes)
- `src/qe/bus/protocol.py` — 6 new topics (158 total): `arena.tournament_started`, `arena.tournament_completed`, `arena.match_completed`, `arena.divergence_checked`, `arena.sycophancy_fallback`, `arena.elo_updated`
- `src/qe/api/app.py` — Arena wired into lifespan: `CompetitiveArena(bus=bus, config=ArenaConfig())` created after engine_factory and passed as `arena=` param to `CognitiveAgentPool`. `competitive_arena` feature flag (disabled by default). Multi-agent call site in `POST /api/goals` routes through `run_competitive_inquiry()` when both `multi_agent_mode` and `competitive_arena` flags are enabled — returns `ArenaResult` fields (arena_id, winner_id, sycophancy_detected, match_count, total_cost_usd) or falls back to standard multi-agent merge. `GET /api/arena/status` endpoint returns Elo rankings and arena status. Arena status included in `GET /api/monitoring` response. Shutdown cleanup sets `_competitive_arena = None`
- Tests: `tests/unit/test_arena_models.py` (13 tests: model construction, ID prefixes, field validation, serialization), `tests/unit/test_competitive_arena.py` (30 tests: Elo math — win/loss/draw/upset/expected/sum preservation, Thompson sampling — BetaArm/agent selection/strong prior, divergence — high/low similarity/fallback, cross-examination — challenges/fallback, defense, judgment — winner/draw/fallback, tournament — 2-agent/3-agent round robin/single elimination/budget exhaustion, sycophancy — skip cross-exam/majority vote/bus events, bus events — correct topics published/no events without bus, status, match pairing), `tests/unit/test_context_compression.py` (9 tests: ConversationSummary model, no-op when short/at limit, LLM compression with key facts, fallback to truncation, custom keep_recent), `tests/integration/test_hil_e2e.py` (8 tests: proposal creates pending file/contains expiry, approved/rejected publishes correct topic, timeout auto-rejects, pending file cleanup on approval/timeout, 3 concurrent proposals with mixed decisions), + extended `test_cognitive_agent_pool.py` (+4 tests: competitive inquiry with/without arena, single result, empty pool), `test_inquiry_bridge.py` (+3 tests: arena completed stores episode/sycophancy in summary/no crash on store failure; lifecycle updated to 5 topics), `test_strategy_models.py` (+1 test: arena_enabled flag)

### Innovation Scout — COMPLETE
All built, tested (158 tests across 9 test files), lint clean. Self-improving meta-agent that scouts the internet for improvements, generates code, tests in sandboxed git worktrees, and submits proposals for human approval:
- `src/qe/models/scout.py` — 6 Pydantic models: ScoutFinding, ImprovementIdea, CodeChange, TestResult, ImprovementProposal, ScoutFeedbackRecord (ID prefixes: fnd_, idea_, prop_, sfb_)
- `src/qe/substrate/scout_store.py` — Async SQLite CRUD for proposals, findings, feedback. Migration: `src/qe/substrate/migrations/0014_scout_proposals.sql` (3 tables)
- `src/qe/services/scout/sources.py` — SourceManager: LLM-generated search queries via instructor, web_search() execution, web_fetch() content extraction, rejected pattern avoidance
- `src/qe/services/scout/analyzer.py` — ScoutAnalyzer: LLM relevance/feasibility/impact scoring with composite score (0.4/0.3/0.3 weights), dynamic threshold, approved pattern biasing
- `src/qe/services/scout/codegen.py` — ScoutCodeGenerator: LLM code change generation with impact/risk/rollback assessments
- `src/qe/services/scout/sandbox.py` — ScoutSandbox: git worktree creation, file application, pytest execution with timeout, unified diff capture, branch merge/cleanup
- `src/qe/services/scout/pipeline.py` — ScoutPipeline: 6-phase orchestration (Source Discovery → Content Extraction → Analysis → Code Gen → Sandbox Testing → Submit)
- `src/qe/services/scout/service.py` — InnovationScoutService: poll loop, HIL approve/reject handlers (git merge on approve, branch delete on reject), learning loop (feedback-driven query refinement, dynamic threshold adjustment)
- `src/qe/bus/protocol.py` — 8 new topics (134 total): scout.cycle_started/completed, scout.finding_discovered, scout.idea_analyzed, scout.proposal_created/tested/applied, scout.learning_recorded
- `src/qe/bus/schemas.py` — 8 new payload schemas (52 total)
- `src/qe/config.py` — ScoutConfig (poll_interval, max_findings/proposals, min_composite_score, budget_limit, hil_timeout, search_topics). Feature flag: `innovation_scout` (disabled by default)
- `src/qe/api/app.py` — 6 API endpoints: GET /api/scout/status, GET/POST /api/scout/proposals (list, detail, approve, reject), GET /api/scout/learning
- `src/qe/api/static/index.html` — Scout dashboard tab with proposal cards, syntax-highlighted diffs, test results, approve/reject UI, filter tabs, real-time WebSocket updates
- Tests: `tests/unit/test_scout_models.py` (29), `test_scout_analyzer.py` (27), `test_scout_sources.py` (31), `test_scout_codegen.py` (7), `test_scout_sandbox.py` (19), `test_scout_pipeline.py` (9), `test_scout_service.py` (12), `test_scout_wiring.py` (16), `tests/integration/test_scout_e2e.py` (8)

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
- PromptMutator tests mock LLM via `patch("qe.optimization.prompt_mutator.instructor")` + mock client with `AsyncMock` for `chat.completions.create` returning `MutatedPrompt`; feature flag gating via `patch("qe.optimization.prompt_mutator.get_flag_store")`
- KnowledgeLoop tests mock LLM via `patch("qe.runtime.knowledge_loop.instructor")` + mock client; feature flag gating via `patch("qe.runtime.knowledge_loop.get_flag_store")`; `Claim` model requires `source_envelope_ids=[]` when constructing programmatically
- InquiryBridge tests use `MagicMock` for bus (subscribe/unsubscribe/publish), `AsyncMock(spec=EpisodicMemory)` for episodic memory, `MagicMock` with `_current_strategy` attribute for strategy evolver, `AsyncMock` with `trigger_consolidation` for knowledge loop
- ExecutorService tests mock LLM via `patch("qe.services.executor.service.litellm")` + `AsyncMock` for `acompletion`; tool registry/gate via `MagicMock` with `get_tool_schemas`/`execute`/`validate` methods; rate limiter via `patch("qe.services.executor.service.get_rate_limiter")`
- GoalSynthesizer tests mock LLM via `patch("qe.services.synthesizer.service.instructor")` + `AsyncMock` client returning `SynthesisInput`; goal store via `AsyncMock` with `load_goal`/`save_goal`; dialectic engine via `AsyncMock` with `full_dialectic` returning mock report with `revised_confidence`
- Goal pipeline E2E tests use `MemoryBus` (real) + `FakeGoalStore` (in-memory dict) + real `Dispatcher` + `ExecutorService`/`GoalSynthesizer` with mocked LLMs
- Setup/onboarding tests use `tmp_path` for `.env` files and `monkeypatch` for `CONFIG_PATH`; endpoint tests use `patch("qe.api.app.is_setup_complete")` and `patch("qe.api.app.save_setup")` to control setup state; `CHANNELS` constant validated in `test_onboarding.py`
- `POST /api/setup/save` is for initial setup only (403 after complete); `POST /api/setup/reconfigure` is for post-setup changes (no 403 guard)
- CompetitiveArena tests mock LLM via `patch("qe.runtime.competitive_arena.instructor")` + mock client with response dispatch based on `response_model` parameter (returns `_DivergenceResult`, `_CrossExamResult`, `_DefenseResult`, or `_JudgeResult`); bus events tested via `MagicMock` bus with `publish.call_args_list`
- Context compression tests mock LLM via `patch("qe.runtime.context_manager.instructor")` + `AsyncMock` returning `ConversationSummary`; verify fallback to truncation on LLM failure
- HIL integration tests use real `HILService` with `MagicMock` bus and `tmp_path` directories; pre-create decision files before `_handle_hil_request()` so poll picks them up immediately; use `asyncio.wait_for()` with 5s timeout on poll tasks
- InnovationScoutService is a standalone service (not BaseService subclass), like ModelDiscoveryService. Feature flag: `innovation_scout` (disabled by default). Tests mock pipeline via `AsyncMock()`, feature flags via `patch("qe.services.scout.service.get_flag_store")`. Sources tests patch `qe.tools.web_fetch.web_fetch` (lazy import) and `qe.services.scout.sources.web_search`. Sandbox tests use `patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec")` with `_make_proc()` helper

---

## v3 Architecture Vision: President + Ministers (Design Phase)

> Brainstormed 2026-03-04. Not yet implemented. Captures the full intended redesign.

### Core Concept

Restructure the system around a **President + Council of Ministers** hierarchy — same pattern as OpenClaw (one main agent) but extended into two tiers so the President stays strategic and ministers own domains.

- **President** — the only agent the user talks to. Sets goals, prioritizes, synthesizes, makes judgment calls. Never does detailed work directly.
- **Chief of Staff (CoS)** — sits between President and ministers. Manages President's context budget, routes tasks to ministers, compresses minister reports before they reach the President. Pass-through for simple tasks, LLM compression only when multiple ministers report back.
- **Ministers** (5 domains) — own specific capabilities, do the actual work, use the plugin infrastructure as tools, can spawn subagents.
- **Subagents** — disposable workers spawned by ministers for specific tasks, destroyed when done.

### The Four Layers

```
UI LAYER
  Chat · Mission Control (live map) · Config panels · Quick switch · Soul editor · Skill toggles
    ↕ WebSocket
GATEWAY LAYER
  Receives user messages → fans out to all active chains · Streams results · Manages chain lifecycle
    ↕
AGENT RUNTIME LAYER  (NEW)
  Chain A / Chain B / Chain C — each has: President → CoS → [Research, Memory, Analysis, Verification, Strategy Ministers] → Subagents
    ↕ tool calls
PLUGIN LAYER  (EXISTING SYSTEM, REWRAPPED)
  InquiryPlugin (InquiryEngine, CognitivePool, HypothesisMgr)
  MemoryPlugin (4-tier memory, BayesianBeliefs, EpisodicMemory)
  ArenaPlugin (CompetitiveArena, FactChecker)
  StrategyPlugin (StrategyEvolver, PromptRegistry, KnowledgeLoop)
  PlannerPlugin (Planner, Dispatcher, Executor)
  ToolsPlugin (web_search, code_exec, browser, files)
  Background: Digest, Monitor, Doctor, Scout
  Cross-cutting: Security, Recovery, Checkpoint, HIL
  [Event bus still coordinates these internally — untouched]
    ↕
SUBSTRATE (UNCHANGED)
  SQLite · BayesianBeliefStore · GoalStore · QuestionStore
```

### What Every Agent Is

Every node (President, CoS, Minister, Subagent) is the same `Agent` class. System prompt assembled as:
```
[CONSTITUTION — hardcoded, immutable]
[soul.md — who the agent is]
[role.md — what the agent owns]
[enabled SKILL.md files — injected when toggled on]
[memory context — filtered by CoS]
```
Ministers are lazy — instantiated per-task, destroyed when done. Only President and CoS are always-on.

### How Ministers Are Called

No event bus at the top level. President calls ministers as direct async tools:
```python
tools = [
    Tool("ask_research_minister",     research_minister.run),
    Tool("ask_analysis_minister",     analysis_minister.run),
    Tool("ask_memory_minister",       memory_minister.run),
    Tool("ask_verification_minister", verification_minister.run),
    Tool("ask_strategy_minister",     strategy_minister.run),
]
```

### Minister → Plugin Toolsets

| Minister     | Tools                                                                        |
|--------------|------------------------------------------------------------------------------|
| Research     | `deep_inquiry()` → InquiryPlugin, `spawn_agents()` → CognitivePool, `search_web()` |
| Memory       | `recall()` → Episodic+BeliefStore, `store_insight()`, `get_patterns()` → ProceduralMemory |
| Analysis     | `dialectic_critique()` → DialecticEngine, `epistemic_check()` → EpistemicReasoner |
| Verification | `run_tournament()` → CompetitiveArena, `fact_check()` → FactChecker          |
| Strategy     | `select_strategy()` → StrategyEvolver, `get_patterns()` → ProceduralMemory   |

### Skills vs Plugins (OpenClaw model)

- **Skills** = markdown protocols (SKILL.md) injected into agent system prompt when enabled. Toggle off → removed from prompt. Toggle on → injected. Teach the agent HOW to operate in a domain. Keep compact — description + trigger + protocol. Full runbook loaded on demand.
- **Plugins** = platform-level code exposing existing services as callable tools. Thin wrappers around existing services.

### The Soul System — Five Layers

| Layer | Who edits | Speed | Description |
|---|---|---|---|
| Constitution | Nobody (hardcoded) | Never | Ethical guardrails |
| Soul | User (deliberately) | Slow | Who the agent is — personality, values |
| Role | User (deliberately) | Moderate | What the agent owns and does |
| Configuration | User (casually) | Instant | Quick switch loadouts — skills, behaviors |
| Adaptive Memory | System (automatically) | Continuous | Learned through use |

Soul files: `agents/president/soul.md`, `agents/ministers/research/soul.md`, etc. Per-chain President souls at `chains/chain_a/president_soul.md`. Ministers share soul/role files across chains.

### Multi-Chain Execution

Multiple fully independent chains (President + full stack each) receive the same user prompt simultaneously. User designates one as active (conversation thread lives there). Others are observational. UI shows all chains side-by-side with live progress. Divergence alerts when chains reach different conclusions. Each chain has its own LLM config (`chains/chain_a/llm_config.json`) and independent memory.

### Quick Switch Configurations

JSON loadout files (`configs/deep_research.json`, etc.) define which ministers are active and which skills each agent has enabled. Switching config reinstantiates agents with new skill sets — no restart.

### Tiered Execution Model

| Tier | Latency | When | Who |
|---|---|---|---|
| Direct | <2s | Simple factual | President only |
| Focused | <15s | Single-domain | President + one minister |
| Deep | 60s+ | Complex multi-domain | Full council + subagents |

President decides tier. User can override. Default is Direct.

### Where the Three Loops Land

- **Inquiry Loop** (7 phases) → inside `InquiryPlugin`. Research Minister calls `deep_inquiry()`, plugin runs the full loop.
- **Knowledge Loop** → background service in Plugin layer. Runs independently. Memory Minister reads from BayesianBeliefStore on demand.
- **Strategy Loop** → background service in Plugin layer. StrategyEvolver runs independently. Chains record outcomes to their own strategy memory.

### Existing Service Mapping

**Becomes minister tools:** inquiry, researcher, harvest, query → Research; fact_checker, validator, verification → Verification; planner, dispatcher, executor → Planning (new minister); analyst, synthesizer → Analysis; memory → Memory; writer, coder → skills on any minister.

**Stays as background infrastructure:** checkpoint, digest, monitor, doctor → Plugin layer background services; security, recovery → cross-cutting on every agent call; ingestor → Gateway plugin.

**Special cases:**
- `chat` service → dissolves into President's conversation interface
- `mass_intelligence` → alternative InquiryPlugin coordination mode (optional)
- `scout` → stays as background Plugin but gets connected to skill registry: Scout finds improvements → generates SKILL.md → submits via HIL → user approves → skill added to registry

### HIL Integration

Any agent at any layer can pause and surface a decision to the user. That node turns yellow in the visual map. User can respond directly to the paused agent. Triggers when: ambiguous decision above confidence threshold, minister needs clarification before spawning expensive subagents, irreversible action, cost would exceed budget.

### What to Build vs Keep

**Keep (wrap as plugins):** All memory tiers, InquiryEngine, CognitiveAgentPool, CompetitiveArena, StrategyEvolver, KnowledgeLoop, BaseService LLM calling (instructor+litellm), budget tracking, rate limiting, EngramCache, all tools, event bus (internal plugin coordination).

**Build new:** `Agent` class (soul+role+skills+tools+ReAct loop), `Chain` class (President→CoS→Ministers as tools), `Gateway` (multi-chain fan-out, WebSocket), plugin wrapper interfaces (thin wrappers around existing services), soul/role markdown files, skill markdown files, config system (JSON loadouts), UI (React Flow map, config panels, chat, comparison view).

### Build Sequence

```
Phase 1: MVP President
  Agent class + single chain (President, direct tool access, no ministers yet)
  Establish evaluation baseline: 20 real queries, score outputs

Phase 2: Infrastructure Bridge
  Plugin wrappers + Research Minister with real InquiryEngine
  Validate: ministers produce measurably better results

Phase 3: Full Council
  All five ministers + CoS compression + soul/role files + skill files

Phase 4: Multi-Chain
  Multiple chains, independent memory, LLM config per agent

Phase 5: Configuration System
  JSON loadouts + quick switch + tiered execution + HIL throughout hierarchy

Phase 6: UI
  WebSocket streaming + React Flow map + config panels + comparison view + soul editor
```

### Known Gaps (Must Resolve Before Building)

1. **CoS compression** — lightweight LLM call for multi-minister compression, pass-through for single minister. Needs concrete design.
2. **Shared user profile above chains** — user preferences/identity should live above all chains, not in chain memory.
3. **Conversation thread ownership** — who holds multi-turn history? Must survive context compression and sessions.
4. **Memory access between ministers** — shared or isolated? If shared, peer-to-peer bypasses hierarchy. Explicit decision needed.
5. **Cross-chain divergence arbitration** — CompetitiveArena works within a chain, not across chains.
6. **Cost controls** — 3 chains × 5 ministers × 3 subagents = potentially $5-20/query. Deep mode is explicit opt-in. Aggressive defaults required.

### Honest Value Assessment

The inquiry engine + memory system is world-class infrastructure currently inaccessible behind an event bus. The President + Ministers architecture gives it a face. The visual map + skill configuration makes the system inspectable and controllable — something no consumer AI product offers. The hierarchy adds explainability and debuggability, not necessarily raw capability. A single agent with direct tool access might produce equivalent results for most queries with less overhead. The hierarchy's value is that every decision, every agent, every result is visible and attributable.

**Core risk:** Building a beautiful architecture that takes months to implement and ends up slower and more expensive than the original. Phase 1 (MVP President) validates the concept before committing to the full council.

---

## v3 Evolution: Visual Graph Composer + Two-Layer Architecture

> Extended design 2026-03-04. Supersedes the fixed President+Ministers hierarchy — that becomes one default template, not the only topology.

### The Shift: Fixed Hierarchy → Flexible Visual Graph

The President+Ministers layout is one possible arrangement. The actual system is a **visual node canvas** where the topology the user draws IS the system configuration. Inspired by Reactable (tangible musical instrument where placing and connecting physical pucks builds a synthesizer — the graph IS the instrument).

Users drag components onto a canvas, connect them, and each connection defines the execution flow. A Master Agent at the top receives user input, delegates to connected children, results flow back up. Multiple independent trees can run simultaneously doing different things with different agents and different LLMs.

### Two Orthogonal Layers

```
GRAPH LAYER (user configures — flexible, any topology)
  What agents exist · How they connect · Execution flow
  Visual canvas · Component library · Drag-and-drop

        every node sits on top of:

INFRASTRUCTURE LAYER (always on, automatic, invisible)
  Memory · Verification · Drift detection · Learning
  Runs under every node regardless of graph topology
  User benefits without wiring it in
```

The graph defines structure. The infrastructure ensures quality.

### Infrastructure Layer — What Runs Under Every Node Automatically

1. **ContextCurator** — loads relevant working memory before each agent executes
2. **EpistemicReasoner** — checks what the agent knows vs. infers before it produces output; flags uncertain claims
3. **Belief grounding** — every claim checked against BayesianBeliefStore; contradictions and novel claims flagged with confidence scores
4. **Episodic write** — every agent's output automatically written to Tier 1 after execution
5. **Goal anchor injection** — every delegation carries the root goal so agents deep in a chain never lose context of the ultimate purpose
6. **KnowledgeLoop** (background) — promotes episodic patterns to Tier 2 semantic beliefs; runs globally across all trees
7. **StrategyEvolver** (background) — learns which graph topologies and agent configurations produce best outcomes

### Memory Tier Scoping in the Graph

| Tier | Scope | Lifetime |
|---|---|---|
| Tier 0 Working | Per-node | Duration of that node's execution |
| Tier 1 Episodic | Per-tree | Duration of the session |
| Tier 2 Semantic (Bayesian) | Global — all trees, all sessions | Permanent |
| Tier 3 Procedural | Global — all trees, all sessions | Permanent |

Each graph template has a UUID. Its learnings are tagged. When the same graph runs again (days/weeks later), agents automatically load their own history from Tier 2.

### Confidence Propagation — The One Rule That Holds Everything Together

Every edge in the graph carries three things:
1. **Task/result** — the actual content
2. **Goal chain** — root goal preserved all the way down every delegation
3. **Confidence score** — produced by the upstream node, propagated and updated at each hop

```
[Research Agent] ──(confidence: 0.81)──→ [Analysis Agent]
```

Analysis Agent knows its input is 81% confident. It weights its reasoning accordingly. If confidence drops below threshold anywhere in the chain → automatic flag to user, optional Verification Node trigger, or route to alternate branch.

### Reducing Hallucinations — Three Stacked Mechanisms

1. **EpistemicReasoner** (infrastructure, automatic on every node) — agent assesses what it actually knows vs. infers before outputting; uncertain claims are marked
2. **Belief grounding** (infrastructure, automatic) — claims checked against BayesianBeliefStore; contradictions flagged; novel claims marked unverified; downstream agents see confidence scores not just claims
3. **Verification Node** (graph layer, explicit component) — user places this anywhere; runs CompetitiveArena on upstream output before it proceeds downstream; catches hallucinations at the boundary

### Reducing Context Drift — Goal Anchor Injection

Every delegation carries:
```python
{
  "root_goal": "...",      # original user request, always preserved
  "goal_chain": [...],     # every intermediate delegation in the path
  "immediate_task": "...", # what this specific agent needs to do
  "context_budget": 2000,  # tokens available to this agent
}
```
ContextCurator detects drift from root goal on every node. If severe → escalates to parent. Agents five levels deep always know their ultimate purpose.

### Reducing Compounding Errors

Bad output at node 2 becomes node 3's ground truth. Confidence propagation prevents silent amplification: each node's output carries the confidence chain. Master Agent at the top sees the full chain and can trigger reverification if cumulative confidence falls below threshold.

### Component Types

```
Orchestrators: Master Agent, Gateway (fan-out/merge), Sequencer (serial), Router (conditional)
Specialists:   Research, Memory, Analysis, Verification, Scraping, Writing, Coding
Infrastructure: Memory Store node, Tool node, Trigger (schedule/event), Fork, Merge
Custom:        User-defined — any soul + role + skills + tools + LLM
```

### Graph Execution Engine

Graph is serialized as JSON (nodes + edges + configs). Execution engine:
1. Loads graph JSON
2. Instantiates each node as an `Agent` with its config
3. Routes user input to the Master Agent node
4. Master executes ReAct loop, calls children via tool calls (edges define available children)
5. Results flow back through the graph with confidence scores
6. WebSocket streams node status events to UI in real time

WebSocket events: `node_started`, `node_status_update`, `node_completed`, `node_error`, `node_hil_required`, `edge_data_flow`, `cost_update`, `graph_completed`

### Graph Validation Rules

System warns when:
- Chain has no verification node (hallucinations may propagate unchecked)
- Master Agent has no output connection
- Circular dependency detected
- Agent budget limit likely to be exceeded
- Confidence threshold incompatible with downstream node's minimum requirement

### Updated Build Sequence

```
Phase 1: Graph Execution Engine + MVP Master Agent
  Graph JSON schema · Agent class · single-node graph works end-to-end
  Infrastructure layer runs automatically under every node
  Establish evaluation baseline

Phase 2: Infrastructure Bridge
  Plugin wrappers (InquiryPlugin, MemoryPlugin, ArenaPlugin, etc.)
  Confidence propagation on edges · Belief grounding on every node

Phase 3: Component Library
  All preconfigured component types · Verification Node
  Goal anchor injection · Goal chain propagation

Phase 4: Multi-tree + Multi-LLM
  Multiple independent trees · Per-node LLM config
  Cross-tree divergence detection

Phase 5: Configuration + Soul System
  Soul/role/skill files · Quick switch loadouts
  Graph templates · Save/load/export

Phase 6: UI
  React Flow canvas · Component palette · Properties panel
  Live execution visualization · Confidence on edges
  Chat integration · HIL experience · All tabs
```

### UI Architecture (Full Plan — see UI section below)

Three-panel layout: Chat (left, persistent) | Canvas (center, main) | Properties (right, contextual). Bottom tab bar: Templates | Memory | Scout | Logs | Settings. Two modes: Edit (build graph) and Run (live execution overlay). Dark theme, Reactable-inspired node glow aesthetic.

---

## v3 UI Plan

### Application Layout

Three persistent panels. No tab-switching to get to chat. Everything visible simultaneously.

```
┌─────────────────────────────────────────────────────────────────────┐
│  [QE]  Graph: EV Research ▼  [+ New]    [▶ Run] [⬛ Stop]  $0.23 ⚠ │  ← Top bar
├──────────────┬──────────────────────────────────┬───────────────────┤
│              │                                  │                   │
│   CHAT       │         CANVAS                   │   PROPERTIES      │
│   (280px)    │         (flex, main workspace)   │   (320px)         │
│              │                                  │                   │
│  [history]   │   drag · connect · configure     │  [node config]    │
│              │   live execution overlay         │  or               │
│  [input]     │                                  │  [exec log]       │
│              │                                  │  or               │
│              │                                  │  [system status]  │
│              │                                  │                   │
├──────────────┴──────────────────────────────────┴───────────────────┤
│  [Templates]  [Memory]  [Scout]  [Logs]  [Settings]                 │  ← Bottom tabs
└─────────────────────────────────────────────────────────────────────┘
```

**Top bar:** Graph name (editable inline, dropdown to switch graphs), New Graph button, Run/Stop controls, live cost ticker, warning badge (graph validation issues count), settings icon. All panels resizable. Left and right panels collapsible to icon strips.

**Left panel (Chat):** Always visible. The primary user interaction surface. Shows conversation history with the Master Agent, intermediate status updates from running agents, HIL request cards with inline reply. User never needs to look at the canvas to use the system — chat is self-contained. In Edit mode: second tab switches to Component Library.

**Center (Canvas):** Main workspace. Infinite pan/zoom. Edit mode or Run mode (overlay). Mini-map bottom-right.

**Right panel (Properties):** Contextual. Node config when a node is selected. Execution log when running with nothing selected. System status by default.

**Bottom tabs:** Templates | Memory | Scout | Logs | Settings. Secondary surfaces. Don't interfere with main workspace.

### Canvas — Edit Mode

Controls: scroll to zoom, middle-click drag to pan, click node to select, drag output→input port to connect, right-click for context menu (Add Node, Paste, Select All), Cmd+Z/Y undo/redo, Cmd+D duplicate, drag-rectangle multi-select, auto-layout button, fit-to-screen button.

**Graph validation badges:** Orange/red warning icons on violating nodes/edges. Issues: no verification node in chain, circular dependency, Master Agent has no output connection, budget limit likely exceeded, confidence threshold mismatch between connected nodes.

### Node Visual States

**Idle:** thin border, dim, shows type icon + name + LLM provider + active skill badges.

**Active (running):** animated glowing border (color = LLM provider: Anthropic=amber, OpenAI=green, Google=blue, Ollama=grey). Shows live status text + progress bar + running cost.

**Completed:** green checkmark, confidence score, cost, duration, "View Output" button.

**HIL — Waiting:** amber pulsing border, "Needs your input", question preview, "Answer in chat →" link.

**Error:** red border, error message, Retry/Skip/Log buttons.

### Edge Visual Design

- **Idle:** thin grey directional arrow
- **Data flowing:** animated particles moving along line, color from source node's provider color, line thickness scales with token count
- **Confidence pill on edge:** `0.81 ✓` green → `0.54 ⚠` amber → `0.31 ✗` red. Below threshold → edge turns red, parent node warns.
- **Disabled:** dashed grey line

### Component Library (Chat panel → Library tab in Edit mode)

```
ORCHESTRATORS:  Master Agent · Gateway (fan-out/merge) · Sequencer (serial) · Router (conditional)
SPECIALISTS:    Research · Memory · Analysis · Verification Node · Scraping · Writer · Coding · Fact Checker
INFRASTRUCTURE: Memory Store · Tool Node · Trigger · Fork · Merge
CUSTOM:         My Components · + Create New
```

Each component: icon, name, one-line description. Drag to canvas to instantiate with default config.

### Properties Panel — Node Config Tabs (8 tabs)

1. **General** — name, description, type, tags
2. **Identity** — Soul editor (markdown, syntax-highlighted) + Role editor. Version history (last 5 saves, diff/revert). Reset to default button.
3. **Skills** — toggle list. Enabled skills green-dotted. Drag to reorder (order = injection order). Expand to preview content. "+ Add Skill" → skill picker or custom markdown editor.
4. **LLM** — provider dropdown, model dropdown, temperature slider, max tokens, thinking level (Default/High), "Test config" button.
5. **Tools** — checkboxes: web_search, web_fetch, code_execute, file_read, file_write, browser_navigate. Per-tool settings on expand. "+ Add Custom Tool".
6. **Memory** — read/write access per tier (0/1/2/3). Memory scope (tree-only vs global). "Clear this agent's memory" with confirmation.
7. **Behavior** — confidence threshold slider, HIL triggers (Never/On uncertainty/On irreversible/Always), max retries, timeout, budget limit per call, execution tier (Auto/Direct/Focused/Deep).
8. **Advanced** — disable epistemic reasoner, disable belief grounding, disable episodic write (all not recommended), goal chain depth limit, custom system prompt injection (appended after role.md, before skills).

### Chat Panel — Run Behavior

User sends message → Master Agent receives it → chat streams:
- Immediate acknowledgment from Master Agent
- Status updates as graph executes: `● Research Agent: Generating questions...`
- HIL requests as special cards: `⚠ Research Agent asks: "Which data source?"` with inline reply input
- Final result as normal message with agent tag + confidence score + cost

User never needs to look at canvas. Canvas is for understanding and configuring, not for operating.

### Canvas — Run Mode Overlay

On execution start, canvas overlays run-mode visualization on top of edit graph:
- Nodes animate by state (idle/active/complete/error/HIL)
- Edges animate with data flow particles + confidence pills in real time
- Floating execution summary (top-right): nodes done/running/waiting, total cost, elapsed time
- Pause button (writes checkpoints after current step), Stop button
- Click any completed node → full output + reasoning trace in right Properties panel
- After execution: "ghost" of last run persists faintly. Clear with button.

### Bottom Tabs — Full Spec

**Templates:** Grid of cards (thumbnail preview, name, description, node count, last used). Default templates ship with system (President+Ministers, Research Pipeline, Deep Analysis, Simple Q&A). User templates alongside. Click → preview full graph → "Load as New Graph". Export/Import JSON.

**Memory — 4 sub-tabs:**
- *Episodic:* Timeline of recent executions. Entry: graph name, query summary, key findings, timestamp, cost. Expandable.
- *Beliefs:* Searchable BayesianBeliefStore. Filter by confidence / topic / source graph. Each belief: claim, confidence, evidence count, last updated, source episodes.
- *Patterns:* ProceduralMemory browser. Successful execution patterns, tool sequences. Active/retired toggle.
- *Graph History:* Per-graph performance — avg confidence, avg cost, common failure points, top learnings. "Reset this graph's memory" button.

**Scout:**
- Pending proposals: list with status badges (pending/approved/rejected/applied). Each card: description, composite score, test result, diff preview, approve/reject.
- Applied: history of applied improvements.
- Settings: enabled toggle, poll interval, budget limit, search topics list, min composite score threshold.

**Logs:**
- Real-time streaming log. Filter by graph / node / level.
- Cost breakdown accordion: session total → per graph → per node → per LLM call.
- Export as JSON or CSV.

**Settings — 5 sections:**
- *LLM Providers:* API keys (masked), default model per execution tier, model assignments by role type, test connection button.
- *Infrastructure:* Knowledge Loop (toggle + consolidation interval), Strategy Evolver (toggle + eval interval), Innovation Scout (toggle), Competitive Arena default, Prompt Evolution toggle.
- *Budget:* Global session limit, per-graph limit, per-node default, alert threshold %, hard stop vs warn-only toggle.
- *Security:* Allowed shell commands list, allowed external domains, rate limits per tool, confirmation requirements (file writes / external API / code execution).
- *Channels:* Telegram/Slack/Email/Webhook config. Per-channel: which graph it routes to, message format settings.

### Technical Connection: Graph → Backend

**Graph JSON** is the single source of truth. Stored at `graphs/{graph_id}.json`. Auto-saved on every config change. Locked at execution start (running executions use their start-time snapshot).

**Execution flow:**
1. `POST /api/graphs/{graph_id}/run` with user message
2. Graph Execution Engine loads JSON, instantiates Agent objects with their configs
3. Infrastructure layer hooks attach to every agent automatically (ContextCurator, EpistemicReasoner, episodic write, goal anchor injection, belief grounding)
4. Master Agent node starts ReAct loop; children called via tool calls defined by edges
5. Engine streams WebSocket events to frontend
6. Frontend updates node states + edge animations in real time

**WebSocket events:**
```
node_started        {node_id, timestamp}
node_status_update  {node_id, status_text, progress_pct}
edge_data_flow      {edge_id, tokens, confidence}
node_hil_required   {node_id, question, context}
node_completed      {node_id, confidence, cost_usd, duration_ms}
node_error          {node_id, error_message, retryable}
cost_update         {total_usd, delta_usd}
graph_completed     {total_cost, total_duration, root_confidence}
```

### Design Constraints — Solved

#### 1. Port Type System

Three port types only. Simplifying further is wrong — more types add friction without benefit.

**Types:**
- `delegation` — the main flow type. Task goes down from parent to child, result returns up. Used for all agent-to-agent connections.
- `memory` — Memory Store output → agent memory input. Agent automatically loads this context before executing.
- `signal` — Trigger output → agent activation input. No data, just an activation pulse.

**Edge directionality:** Single edge per connection (not two). The edge represents the delegation relationship. During execution, animation direction reverses when the result returns upward — visually shows the two-phase flow on one line. Execution engine tracks the call stack internally; the graph topology only defines who can delegate to whom.

**Port layout:** Every agent node has:
- One `delegation` input port (top center) — receives tasks from parent
- One `delegation` output port (bottom center) — delegates to children
- Optional `memory` input port (left side) — if connected to a Memory Store
- Optional `signal` input port (top-left corner) — if connected to a Trigger

Gateway, Fork: one `delegation` input, multiple `delegation` outputs.
Merge: multiple `delegation` inputs, one `delegation` output.
Router: one `delegation` input, multiple labeled `delegation` outputs (each edge has a condition label — required, set by clicking the edge).

**Connection UX:** When user starts dragging from a port, only compatible port types highlight as valid targets. Incompatible ports dim. Attempting to connect incompatible types shows tooltip with explanation. Router outgoing edges require a condition label before the connection is accepted — inline label field appears immediately on drop.

**Backend:** Graph execution engine validates port type compatibility before running. Type mismatches are a hard error, not a warning.

---

#### 2. Circular Dependency Prevention

**Algorithm:** When user releases a drag to connect node A → node B, run DFS from B. If A is reachable from B, the edge would create a cycle — reject it. Self-loops always rejected. Check runs on mouseup (not mousemove — no performance issue).

Backend execution engine independently validates acyclicity via Kahn's topological sort before any run starts. Defense in depth.

**No Feedback node type.** This concept was removed. Iteration in this system happens *inside* agents via their internal ReAct loop (already built). You do not need graph-level cycles to express iteration. The 7-phase inquiry loop, hypothesis refinement, multi-step reasoning — all of these are agent-internal behaviors, not graph topology.

**Error message:** "This connection would create a loop. Graph cycles aren't supported — each agent handles its own iteration internally via its reasoning loop. For multi-step workflows where one agent feeds the next in sequence, use a Sequencer node."

---

#### 3. Graph Versioning

**Decision: No automatic versioning. Graph UUID is stable and never changes regardless of edits.**

Rationale: Graph-scoped episodic memory is about *topic*, not topology. The EV Research graph's memories (CATL, BYD, LG are key players) remain valid whether the graph has 3 or 7 nodes. Automatically invalidating memory on structural changes would destroy useful context for no reason.

**What happens instead:**
- Every execution accumulates episodic memory tagged with `graph_id`
- User explicitly clears memory when they want a fresh start ("Clear graph memory" button in Memory tab)
- When user makes structural changes (add/remove nodes), a non-blocking toast appears: "Graph structure changed. Old memories may not apply to the new layout. [Clear graph memory] [Keep]" — user decides, no forced reset.

**Two separate concepts explicitly decoupled in the UI:**
- **Graph memory** — episodic + semantic learnings loaded at execution time. Clearable without affecting run history.
- **Run history** — record of past executions (inputs, outputs, node states, cost, HIL interactions). Kept for debugging regardless of memory state. Clearable separately.

"Clear graph memory" only clears the episodic context. It does not delete run history. This distinction is labeled clearly in the Memory → Graph History sub-tab with separate buttons.

**During execution:** Each node shows a small "Using N past episodes" indicator so users understand memory is being loaded.

---

#### 4. Execution History

**Storage:** Last 50 runs per graph (configurable in Settings). Node outputs compressed with gzip. Estimate: 50 runs × 10 nodes × avg 5KB = ~2.5MB per graph. Acceptable. Heavy subagent use (3 subagents × 10KB each = 30KB/node) → ~15MB for 50 runs. Still acceptable; enforce a per-run size cap (warn if exceeded, truncate oldest subagent outputs first).

**What is stored per run:**
- Run ID, timestamp, graph JSON snapshot (the exact topology at run start — not current topology)
- User input message
- Per-node: output text (compressed), confidence score, cost, duration, final status
- Per-edge: confidence scores that flowed through
- HIL interactions: {node_id, question_asked, user_answer, timestamp}
- Total: cost, duration, root confidence

**Inspect Run mode (renamed from "Replay" — it is read-only, not re-executable):**
- User clicks a past run in Memory → Graph History
- Canvas switches to read-only Inspect mode: nodes rendered from the *stored graph snapshot* not current graph (so old runs show the topology that actually ran, even if graph has changed)
- Nodes colored by their final status from that run
- Click any node → right Properties panel shows its stored output, confidence, cost, any HIL Q&A
- Top banner: `Inspecting run from [date] · $0.24 · 47s · Confidence 0.81   [Exit]  [Re-run with this input]`
- "Re-run" takes the stored input message and submits it to the *current* graph config (new run, not replay)

---

#### 5. Keyboard Shortcuts

All conflicts resolved. Focus-aware: shortcuts only fire when the relevant panel is active.

**Canvas shortcuts (canvas has focus):**
```
Scroll              Zoom in/out
Space + drag        Pan canvas
Cmd+Shift+H         Fit graph to screen
Cmd+Shift+L         Auto-layout nodes
Cmd+A               Select all
Escape              Deselect all / cancel current draw operation
Delete / Backspace  Delete selected node or edge
Cmd+C / X / V       Copy / cut / paste
Cmd+D               Duplicate selected node(s)
Cmd+G               Group selected nodes into a Composite node
Tab                 Cycle focus to next node
Shift+Tab           Cycle focus to previous node
Enter               Open Properties panel for selected node
F2                  Inline rename selected node
```

**Execution (fires from any focused panel):**
```
Cmd+Enter           Run graph  (NOT from chat input — chat Enter sends message)
Cmd+Shift+.         Stop execution
Cmd+Shift+P         Pause execution (writes checkpoints)
```

**Global:**
```
Cmd+S               Explicit save (auto-saves anyway)
Cmd+Z               Undo
Cmd+Shift+Z         Redo
Cmd+N               New graph
Cmd+O               Open graph picker
Cmd+K               Command palette (search all actions by text)
?                   Show/hide shortcut overlay
```

**Focus rules:**
- `Tab` cycles canvas nodes only when the canvas itself has focus (not when a text input inside Properties panel is focused — Tab should behave normally in text fields)
- `Cmd+Enter` runs the graph only when canvas or Properties panel has focus. When chat input is focused, `Enter` and `Cmd+Enter` both send the chat message (consistent with chat app conventions)
- `Escape` closes modal/dialog if one is open; otherwise deselects canvas selection; never stops a run (stop requires explicit Cmd+Shift+. or the Stop button — too destructive for an accidental keypress)
- Clear focus indicator on all three panels so users always know which is active

**Cmd+K command palette:** Text search over all available actions ("add research agent", "clear graph memory", "run graph", "open settings", "new graph"). Power user feature that reduces reliance on memorizing shortcuts.

---

#### 6. Onboarding

**Core insight:** Two fundamentally different user types require branched paths.

- **Simple user** — wants to talk to an AI. Graphs are intimidating and irrelevant to them.
- **Advanced user** — wants to build custom agent pipelines. Wants full canvas immediately.

The template picker is the branch point. This decision determines the entire UI they see.

**Flow:**

**Step 1 — Welcome (full-screen overlay):**
"Question Engine" title, one-line tagline, two buttons: "Get Started" and "I know what I'm doing →" (skips to Step 3 immediately for returning/technical users).

**Step 2 — API Key (required for real use):**
One field: API key for one provider (Anthropic recommended, dropdown to switch provider). Below it: "Try demo mode instead →" — 10 free queries, no key required. Demo mode shows a persistent subtle banner "Demo: 7 queries remaining · [Add API key]". Onboarding state saved to localStorage — if user closes and returns, resumes at this step.

**Step 3 — Choose your experience:**
Two cards side by side:
- **"Just Chat"** — "Talk to an AI assistant. No setup required." Preview: simple chat interface thumbnail.
- **"Build Agent Graphs"** — "Design custom AI pipelines with full control." Preview: canvas with nodes thumbnail.

**If "Just Chat" selected:**
- Canvas is hidden entirely. User sees Chat panel only (full width) + a minimal top bar.
- A small "⚙ Advanced Mode" toggle in the top bar allows switching to full three-panel view at any time.
- No tutorial needed — it's just a chat input. They start immediately.

**If "Build Agent Graphs" selected:**
- Full three-panel layout loads with a default template (President + Ministers recommended, shown pre-loaded)
- Non-blocking tutorial overlay: 4 small tooltip arrows pointing at each panel + top bar. First tooltip in top-right: "Skip tutorial ×" — obvious, large. Remaining tooltips: "Type here to chat ←", "Your agent graph →", "Configure selected node ←", "Run your graph ↑"
- Overlay auto-dismisses after user sends first message or explicitly skips
- After first successful run: nothing extra. The result appearing in chat is satisfying enough. No modal.

**Returning user behavior:** If `localStorage.hasCompletedOnboarding === true`, skip directly to the app in whichever mode they last used. No welcome screen.

---

## Audit Fix Plan (2026-03-04) — COMPLETED

All fixes applied in commit on `feature/error-recovery` branch. Net result: +30 passing tests, 0 regressions.

### Fixes Applied

1. **Add 3 missing `global` declarations in `lifespan()`** — `app.py` ✓
   - Added `global _memory_store, _prompt_registry, _inquiry_engine`

2. **Wire `memory_ops` router** — `app.py` + `memory_ops.py` ✓
   - Added prefix `/api/memory`, registered router, fixed empty `register_memory_ops_routes()`
   - Fixed 30 previously-failing tests in `test_memory_ops.py`

3. **Add SSRF protection to A2A peer registration** — `peer_registry.py` ✓
   - Validates URL scheme (http/https only), blocks private IPs, localhost, link-local

4. **Fix path traversal in harvest HIL endpoints** — `harvest.py` ✓
   - Sanitizes `hil_envelope_id` with `Path(id).name` before file path construction

5. **Require ADMIN scope for flag modification** — `middleware.py` ✓
   - Added `/api/flags` to `_ADMIN_PREFIXES`

6. **Fix WebSocket authentication bypass** — `middleware.py` + `chat.py` ✓
   - Removed `/ws` from `_PUBLIC_PREFIXES`
   - Added `_ws_authenticate()` helper that validates API key from query params

7. **Fix blocking I/O in async handlers** — `harvest.py`, `knowledge.py` ✓
   - Wrapped `Path.mkdir()` and `Path.write_text()` in `asyncio.to_thread()`
