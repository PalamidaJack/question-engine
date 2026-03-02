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

~1950 tests pass (1038 v1 + 82 Phase 1 + 108 Phase 2 + 14 P1+2 wiring + 94 Phase 3 + 88 Phase 4 + 24 lint fixes + 33 Phase 5 + 23 Phase 6 + 6 real LLM integration + 47 Prompt Evolution A-C + 49 Prompt Evolution D + 48 Knowledge Loop + 38 Loop Integration + 39 Strategy Wiring + 69 Goal Orchestration Pipeline + 21 Enhanced Onboarding + 84 Competitive Arena), ruff clean. The 6 slow tests require KILOCODE_API_KEY and are excluded from default runs.

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
