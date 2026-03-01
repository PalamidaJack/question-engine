# Question Engine OS

## Project Overview

Multi-agent orchestration system being redesigned from a pipeline into a **cognitive system** with three nested loops (Inquiry, Knowledge, Strategy), four-tier memory, and a Cognitive Layer for genuine insight generation.

Full architecture plan: `.claude/plans/tranquil-hopping-harbor.md`

## Tech Stack

- Python 3.14, FastAPI, Pydantic v2, aiosqlite, litellm, instructor
- Virtual env: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
- LLM provider: Kilo Code (OpenRouter-compatible) at `https://kilo.ai/api/openrouter`
- Models: `openai/anthropic/claude-sonnet-4` (balanced), `openai/google/gemini-2.0-flash` (fast)

## Key Directories

- `src/qe/api/app.py` — Main FastAPI app, lifespan, channel wiring, all endpoints
- `src/qe/channels/` — Channel adapters (telegram, slack, email, webhook) + notifications router
- `src/qe/services/` — Planner, Dispatcher, Executor, VerificationGate, Recovery, Checkpoint, Doctor, Chat
- `src/qe/services/inquiry/` — **NEW (v2)**: Dialectic Engine, Insight Crystallizer (Inquiry Loop components)
- `src/qe/bus/` — MemoryBus, event log, bus metrics
- `src/qe/substrate/` — Belief ledger (SQLite), cold storage, goal store, embeddings, BayesianBeliefStore
- `src/qe/models/` — Pydantic models (Envelope, Claim, GoalState, Genome Blueprint, Cognition)
- `src/qe/runtime/` — Service base, context curator, episodic memory, engram cache, metacognitor, epistemic reasoner, persistence engine
- `tests/unit/` — Unit tests (~50+ files)
- `tests/integration/` — Integration + E2E tests
- `config.toml` — Runtime config (model tiers, budget, logging)
- `.env` — API keys (gitignored)

## Running Tests & Linting

```bash
.venv/bin/pytest tests/ --timeout=60 -q    # ~1242 tests, all passing
.venv/bin/ruff check src/ tests/            # all clean
```

## Current State (2026-03-01)

~1242 tests pass (1038 v1 + 82 Phase 1 + 108 Phase 2 + 14 wiring), ruff clean.

### v2 Redesign — Architecture Plan

Full plan at `.claude/plans/tranquil-hopping-harbor.md`. Key concepts:
- **Three Loops**: Inquiry (seconds-minutes), Knowledge (minutes-hours), Strategy (hours-days)
- **Four-Tier Memory**: Tier 0 Working (ContextCurator), Tier 1 Episodic (EpisodicMemory), Tier 2 Semantic (BayesianBeliefStore), Tier 3 Procedural (pending)
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

### Next Steps (in order)
1. **Phase 3**: Inquiry Loop + Knowledge Loop — InquiryEngine (7-phase loop), question generator, HypothesisManager (POPPER), CognitiveAgent model, procedural memory
2. **Phase 4**: Elastic Scaling + Strategy Loop — Thompson router, CognitiveAgentPool, StrategyEvolver, scale profiles
3. **Phase 5**: Integration + Polish — E2E investment opportunity walkthrough, Mac M1 profiling

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
- All cognitive components accept optional `episodic_memory` and `model` params for dependency injection and testability
