# Question Engine OS — Complete Improvement Plan

**Date:** February 28, 2026 (Updated with voice, multimodal, security, cost optimization, OpenClaw ecosystem lessons, and execution framework)
**Current State:** Phase 0 complete — kernel, bus, belief ledger, researcher service, HIL, CLI, FastAPI + WebSocket API, React dashboard prototype
**Target State:** A self-hostable, open-source, autonomous agent orchestration OS with set-and-forget reliability, neurosymbolic knowledge management, continuous self-improvement, voice and multimodal input, multi-channel communication, defense-in-depth security, and cost-optimized operation — running on free local models with optional paid model upgrades
**Phases:** 16 phases across 26 weeks
**Execution tiers:** NOW (Weeks 1-6) → NEXT (Weeks 7-14) → LATER (Weeks 15-26)

---

## Part 0: Execution Framework

### How to Read This Plan

This document serves two purposes:

1. **North-Star Architecture** — Parts 1-3 and the appendices describe the complete vision. Every component, every data flow, every reliability layer. This is the reference.
2. **Execution Roadmap** — The phases are sequenced for delivery, with measurable exit criteria. The Now/Next/Later framework below determines what ships first.

### Now / Next / Later

**NOW (Weeks 1-6) — "User can submit a goal and get a verified result"**
- Phase 1: Foundation Hardening
- Phase 2: Embedding Layer
- Phase 3: Task Decomposition Engine
- Phase 4: Verification & Recovery (basic — structural + contract checks only)
- Phase 5: Tool Integration (web search + file ops only, plus security gating)

Exit gate: A user runs `qe goal "Research X and summarize"` → system decomposes, dispatches, verifies, and returns a result with confidence scores. End-to-end success rate >60% on a 20-goal test suite.

**NEXT (Weeks 7-14) — "System gets smarter, user gets visibility"**
- Phase 6: Memory System
- Phase 7: New Service Types
- Phase 8: Learning Loops
- Phase 9: Free Model & Cost Optimization
- Phase 10: Dashboard & Observability

Exit gate: System demonstrably improves over 50+ goals (calibration curve tightens, routing decisions get cheaper, failure KB prevents known failures). Dashboard shows real-time goal DAGs, cost, and bus flow.

**LATER (Weeks 15-26) — "Expansion and research bets"**
- Phase 11: Self-Management
- Phase 12: SDK & Extensibility
- Phase 15: Voice & Multimodal (start with document ingestion, then voice)
- Phase 16: Communication (start with Slack only, then expand)

Exit gate: System runs unattended for 72 hours handling queued goals. One external channel functional. Document ingestion working.

**RESEARCH BETS (parallel, not on critical path):**
- Phase 13: H-Neuron Integration — defer until local model inference is stable
- Phase 14: Advanced Intelligence (conformal prediction, Bayesian propagation, symbolic inference) — defer until calibration data proves the simpler approaches insufficient
- These are explicitly optional R&D tracks. The system must work well without them.

### Phase Exit Criteria

Every phase has 3-5 measurable KPIs that must be met before the next phase begins. These are defined in each phase section below. A phase is "done" when all KPIs are green, not when all code is written.

---

## Part 1: Current State Assessment

### What Exists (Phase 0)

The system has a solid architectural foundation:

| Component | File(s) | Status |
|---|---|---|
| **Envelope model** | `src/qe/models/envelope.py` | Complete. correlation_id + causation_id enable provenance chains. |
| **Claim model** | `src/qe/models/claim.py` | Complete. Includes Claim, Prediction, NullResult. Confidence, supersession, validity windows. |
| **Genome/Blueprint** | `src/qe/models/genome.py`, `src/qe/kernel/blueprint.py` | Complete. Declarative TOML service configuration with capability declarations. |
| **MemoryBus** | `src/qe/bus/memory_bus.py` | Functional but fire-and-forget. No delivery confirmation, no await-ability. |
| **Bus protocol** | `src/qe/bus/protocol.py` | 26 registered topics. Validation on publish. |
| **BeliefLedger** | `src/qe/substrate/belief_ledger.py` | Core CRUD with supersession logic. FTS5 search implemented with triggers. |
| **ColdStorage** | `src/qe/substrate/cold_storage.py` | Append-only JSON files by year/month. No indexing or retrieval beyond direct ID lookup. |
| **AutoRouter** | `src/qe/runtime/router.py` | Tier-based model selection with cooldown and budget gating. Static tier lists. |
| **ContextManager** | `src/qe/runtime/context_manager.py` | Token-budgeted message building with constitution protection and reinforcement. Compression is a TODO. |
| **BaseService** | `src/qe/runtime/service.py` | Heartbeat loop, LLM calling via instructor + litellm, budget tracking integration. |
| **BudgetTracker** | `src/qe/runtime/budget.py` | Monthly limits, per-model spend tracking, alert thresholds. In-memory only. |
| **ResearcherService** | `src/qe/services/researcher/service.py` | Observation → claim extraction → commit pipeline. Single service type. |
| **HILService** | `src/qe/services/hil/service.py` | File-based pending/completed queue with timeout. Poll-based decision detection. |
| **Supervisor** | `src/qe/kernel/supervisor.py` | Service lifecycle, loop detection with circuit breaking, heartbeat monitoring, stall detection, config hot-reload via watchdog. |
| **ServiceRegistry** | `src/qe/kernel/registry.py` | Simple dict-based service registration. |
| **CLI** | `src/qe/cli/main.py` | `qe start`, `qe submit`, `qe claims list/get`, `qe hil list/approve/reject`, `qe serve`. |
| **API** | `src/qe/api/app.py`, `src/qe/api/ws.py` | FastAPI REST + WebSocket. Health, status, submit, claims, HIL endpoints. Bus-to-WS bridge. Inbox relay for cross-process. |
| **Dashboard** | `question-engine-ui.jsx` (prototype) | Self-contained React mock with fake data. Not connected to real backend. |

### What's Missing

**No task decomposition.** The system processes one observation at a time through a single service. There is no concept of a goal, a plan, a DAG of subtasks, or multi-service orchestration. This is the single largest capability gap.

**No verification.** Outputs from services are accepted without any checking. No postcondition validation, no consistency checking, no belief ledger cross-referencing, no hallucination detection.

**No recovery.** When something fails, it fails. No retry strategies, no alternative approaches, no checkpoint/rollback, no graduated escalation.

**No learning.** The system doesn't improve over time. No calibration tracking, no routing optimization, no failure knowledge base, no planning memory.

**No tool integrations.** Capability declarations exist (web_search, file_read, etc.) but nothing is wired up. The only real I/O is the bus and SQLite.

**No memory system.** No user preferences, no project context, no entity memory. The belief ledger stores claims but doesn't participate in enriching future LLM calls with relevant past knowledge.

**No semantic search.** No embedding model, no vector store. The belief ledger is queryable only by exact field matching.

**No free model optimization.** The system assumes instruction-following models. No structured output enforcement, no prompt tiering, no decomposition adjustment for weaker models.

**No question-driven reasoning.** Service prompts are basic task instructions. No problem representation protocol, no metacognitive self-questioning, no adversarial self-review, no epistemic calibration prompts. Research shows every LLM failure mode maps to a question the model didn't ask itself before or during generation.

**No H-Neuron integration.** The system runs on local open-weight models but doesn't leverage direct access to model internals. The H-Neurons discovery (Tsinghua, Dec 2025) shows that <0.1% of neurons predict hallucination with 70-83% accuracy across models, and suppressing them measurably reduces hallucination and sycophancy. No other agent platform can do this because they use closed APIs.

**No cost optimization architecture.** OpenClaw's #1 community pain point was bill shock — context accumulation (40-50% of spend), workspace file injection (10-15%), tool output bloat (20-30%), and wrong model for the task. QE has the same structural risks: every LLM call re-sends the full context, tool outputs persist in conversation, and the router uses static tier lists rather than multi-factor scoring. No prompt cache alignment, no ephemeral output management, no lazy tool loading.

**No security hardening.** OpenClaw exposed 40,000+ instances via insecure defaults, suffered 800+ malicious packages on its skills marketplace, and enabled prompt injection via email/calendar/webpages. QE's current architecture has similar risks: no input sanitization on bus payloads, no sandboxing beyond file path restrictions, no tool call gating, no integrity monitoring. Security-critical logic lives in prompt-level constitutions (vulnerable to prompt injection) rather than code-level enforcement.

**No voice or multimodal input.** The system only accepts text. No speech-to-text, no OCR, no PDF/DOCX parsing, no audio/video ingestion. Meetings, lectures, voice memos, scanned documents, and images are all invisible to the system. The open-source voice stack has matured significantly (faster-whisper, WhisperX, Kokoro TTS, Pipecat) but none of it is integrated.

**No communication integrations.** No Slack, email, Telegram, Discord, or webhook connections. The only interfaces are CLI, REST API, and WebSocket. OpenClaw's viral adoption was driven by meeting users where they are — messaging apps they already use. QE currently requires users to come to it.

---

## Part 2: Architectural Vision

### The Three-Strata Architecture

The system's intelligence comes from three layers working together:

```
┌─────────────────────────────────────────────────┐
│  STRATUM 1: NEURAL (LLMs)                       │
│  Observation processing, claim extraction,       │
│  analysis, synthesis, creative reasoning         │
│  Multiple models, routed by optimization layer   │
│  Powerful but unreliable — needs verification    │
├─────────────────────────────────────────────────┤
│  STRATUM 2: STATISTICAL (learned infrastructure) │
│  Routing optimization, anomaly detection,        │
│  confidence calibration, failure classification, │
│  HIL prediction. Learns from operational data.   │
│  Cheap, fast, runs on everything.                │
├─────────────────────────────────────────────────┤
│  STRATUM 3: SYMBOLIC (belief ledger + logic)     │
│  Knowledge verification, consistency checking,   │
│  assumption tracking, confidence propagation,    │
│  provenance chains, logical inference.           │
│  Rigorous, auditable, deterministic.             │
└─────────────────────────────────────────────────┘
```

Information flows upward: neural outputs enter the statistical layer for calibration and anomaly checking, then enter the symbolic layer for verification and integration into the knowledge base. Contradictions flow downward: the symbolic layer detects inconsistencies and triggers neural investigation.

The system improves because stratum 2 learns continuously from the data flowing between strata 1 and 3. The LLMs don't get better. The infrastructure around them does.

### The Reliability Architecture

Six layers that transform fire-and-forget execution into verified, recoverable, self-improving operation:

```
┌──────────────────────────────────────────┐
│  Layer 6: ANOMALY DETECTION              │
│  Statistical monitoring on every output  │
├──────────────────────────────────────────┤
│  Layer 5: FAILURE KNOWLEDGE BASE         │
│  Accumulated operational intelligence    │
├──────────────────────────────────────────┤
│  Layer 4: RECOVERY ORCHESTRATOR          │
│  Classify failure → choose strategy →    │
│  execute recovery → escalate if needed   │
├──────────────────────────────────────────┤
│  Layer 3: CHECKPOINT MANAGER             │
│  Savepoints for rollback on failure      │
├──────────────────────────────────────────┤
│  Layer 2: VERIFICATION SERVICE           │
│  Structural + contract + consistency +   │
│  belief + confidence checks              │
├──────────────────────────────────────────┤
│  Layer 1: EXECUTION CONTRACTS            │
│  Pre/postconditions on every subtask     │
└──────────────────────────────────────────┘
```

### The Complete Goal Lifecycle

```
User submits goal
  → Memory Service enriches with relevant context
    → Planner decomposes into DAG with execution contracts
      → Dispatcher manages dependency graph
        → Services execute subtasks using tools + models
          → Verification Service checks every output
            → Recovery Orchestrator handles failures
              → Belief Ledger integrates verified claims
                → Learning loops update routing, calibration, failure KB
                  → Synthesis Service produces final deliverable
                    → User receives result + confidence receipt + provenance
```

### The Question-Driven Reasoning Layer

Research in problem-solving science and LLM failure modes reveals a critical insight:

> **Every LLM failure mode is a question the model didn't ask itself before or during generation.**

- Hallucination = didn't ask "do I actually know this?"
- Sycophancy = didn't ask "would I say this if the user believed the opposite?"
- Shallow reasoning = didn't ask "what's the deep structure here?"
- Goal misspecification = didn't ask "what problem is this person really trying to solve?"
- Overconfidence = didn't ask "what would have to be true for me to be wrong?"
- Context drift = didn't ask "am I still solving the original problem?"

The system embeds question protocols at every level:

**Problem Representation (before planning):**
The planner doesn't jump to decomposition. It first represents the problem — restating it, distinguishing what's asked from what's needed, identifying constraints, defining success criteria, and classifying the problem type (well-defined, ill-defined, or wicked). Research shows 70%+ of solution quality is determined by representation before solving begins.

**Metacognitive Self-Questioning (during execution):**
Every service's system prompt includes questions the model must answer before generating output — confidence mapping, assumption excavation, failure mode analysis. These questions act as cognitive anchors that reduce hallucination, sycophancy, and shallow reasoning at the prompt level, with zero additional compute cost.

**Adversarial Review (during verification):**
The verification service doesn't just check outputs against contracts. It runs adversarial question protocols: "What's the strongest argument against this conclusion? What assumptions haven't been verified? What would falsify this claim?" This is dual-process architecture — System 1 (fast generation) followed by System 2 (deliberate critique).

**Epistemic Calibration (on every claim):**
Every factual claim is tagged with an epistemic status: verified, inferred, estimated, uncertain, or unknown. This is enforced through prompt instructions and verified by the verification service. The system never presents uncertain content with confident prose.

### The H-Neuron Advantage

The H-Neurons discovery (Tsinghua University, December 2025) reveals that less than 0.1% of neurons in LLMs reliably predict hallucination occurrences. These neurons are causally linked to **over-compliance** — the model's tendency to satisfy prompts at the expense of truthfulness.

Because Question Engine runs on open-weight local models, it has direct access to model internals that closed-API platforms (Perplexity, OpenAI, Google) cannot access. This enables:

1. **H-Neuron identification** — one-time profiling identifies the specific neurons in each local model
2. **Real-time monitoring** — forward hooks track H-Neuron activation during inference, providing a hallucination risk score for every generation
3. **Active suppression** — scaling H-Neuron activations by 0.5 measurably reduces hallucination and sycophancy without significantly impairing general capability

This is a structural reliability advantage that no cloud-only platform can replicate.

### Free Model Design Principle

Every component must work with free local models (Llama, Mistral, Qwen, Phi via Ollama). Paid models are optional upgrades that improve speed and quality but are never required. This means:

- Prompts have tiered variants (powerful/balanced/fast/local)
- Structured output uses constrained decoding, not just "please respond in JSON"
- Task decomposition granularity adjusts based on model capability
- Verification strictness increases for less capable models
- The system's intelligence comes from infrastructure, not from model quality

### The Cost Optimization Architecture

OpenClaw's #1 community pain point was **bill shock** — users spending $150-3,600/month with no visibility into why. Root cause analysis from the community revealed five structural cost drivers that QE must solve architecturally:

**1. Context Accumulation (40-50% of total spend)**
Every LLM call re-sends the entire conversation history. Message 1 costs hundreds of tokens; message 50 costs tens of thousands. One OpenClaw user found their session occupied 58% of a 400K context window for basic queries. QE's `ContextManager` already has a token budget, but needs aggressive **LLM-summarized compaction** — not truncation, but actual summarization of prior context that preserves meaning while dramatically reducing tokens.

**2. Workspace File Injection (10-15%)**
OpenClaw injects SOUL.md/AGENTS.md/USER.md into every call — identity and rules that never change but get re-billed constantly. QE's genome system prompt + constitution have the same risk. Solution: **prompt cache alignment** — time periodic tasks (heartbeats, reinforcement loops) to stay within the LLM provider's cache TTL window, ensuring static prompt components hit the cache (90% discount on Anthropic, similar on others).

**3. Tool Output Bloat (20-30%)**
When a tool returns 10,000 tokens of output, that data sits in the conversation forever, re-sent on every subsequent call. Solution: **ephemeral tool outputs** — extract structured data from tool results immediately, then discard the raw output from the conversation context. Only the extracted claims/data persist.

**4. Wrong Model for the Job (up to 25x price difference)**
Simple queries sent to expensive models waste money. OpenClaw's community solved this with Manifest's 23-dimension query scoring that routes each query to the cheapest capable model in <2ms. QE's AutoRouter uses static tier lists — needs multi-factor scoring incorporating task complexity, required capabilities, historical success rates, latency budget, and remaining budget.

**5. Tool Schema Bloat**
Every tool schema injected into the prompt costs tokens on every call, whether that tool is used or not. OpenClaw's community invented the **Tool Discovery pattern** — only one meta-tool (`search_available_tools`) stays in the base context, specific tools are loaded on-demand when the LLM requests them. Claims up to 90% reduction in tool-related context overhead.

### The Security Architecture

OpenClaw's security failures are a roadmap of what NOT to do — and a blueprint for what QE must build from day one:

**The "Lethal Trifecta"** (access to private data + external communication + untrusted content) creates fundamental risk. Every OpenClaw security failure traces to one or more of these:

- **40,214 exposed instances** via insecure defaults (`0.0.0.0` bind, auth disabled)
- **800+ malicious packages** on ClawHub (20% of the entire registry), delivering info-stealers
- **Prompt injection via trusted channels** — instructions embedded in emails, calendar invites, web pages
- **Time-shifted prompt injection** — payloads fragmented across days, detonated when agent state aligns
- **CVE-2026-25253 (CVSS 8.8)** — one-click RCE through UI URL parameter trust

**QE's Security Principles:**

**1. Dual-Stack Enforcement**
Security-critical logic must live in the **service layer** (compiled code), not the **prompt layer** (LLM context). Constitutions are behavioral guidance for the LLM, but they can be "talked out of" via prompt injection. Code-level enforcement in the bus, dispatcher, and tool registry cannot be bypassed by any prompt. This is the key insight from SecureClaw: "Skills can be overridden by prompt injection. Plugins cannot."

**2. Bus-Level Tool Call Gating**
Every tool invocation passes through a security policy layer on the bus before execution. The policy is defined in code, not in prompts. It validates:
- The requesting service has the declared capability
- The tool parameters are within allowed ranges
- The action doesn't exceed the service's permission scope
- Rate limits are respected
This is the AgentGate pattern — an "enterprise-grade firewall" for agent actions.

**3. Sandboxed Execution by Default**
Inspired by IronClaw's WASM sandboxing: credentials are never exposed directly to tools, every file operation is scoped to goal workspaces, code execution runs in restricted subprocesses, and web fetches go through a controlled proxy.

**4. Input Sanitization on Every Bus Payload**
Every envelope entering the system from external sources (API, CLI, inbox relay, future messaging integrations) is sanitized for prompt injection patterns before reaching any LLM. This prevents the attack vector that compromised thousands of OpenClaw instances.

**5. Integrity Monitoring**
A lightweight security monitoring service (inspired by ClawSec) continuously verifies:
- Genome file integrity (drift detection + alert on unauthorized modification)
- Configuration file integrity
- No unauthorized bus topics or subscriptions
- Budget and resource limits respected
- Anomalous patterns in service behavior

### Lessons from OpenClaw

The OpenClaw ecosystem emerged in 4 weeks and compressed years of platform lessons into a month. Key patterns validated by 175,000+ users and 5,700+ community skills:

**What worked:**
- **Lobster deterministic pipelines**: YAML controls sequencing/routing/retrying, LLMs handle creative work at leaf nodes. Reliability comes from not letting LLMs control flow.
- **Temporal knowledge graphs (Graphiti)**: Superior to flat vector RAG for evolving knowledge — exactly what a belief ledger needs.
- **Memory compaction via LLM summarization**: Not truncation — actual LLM-generated summaries that preserve meaning while reducing tokens.
- **MCP lazy loading with smart aggregation**: Connect many tools but expose via relevance-filtered discovery, not bulk injection.
- **Approval gates with resume tokens**: Pause workflow for human review, resume from exactly where it stopped.
- **Security as a composable service**: Not an afterthought bolted on later, but a first-class service that monitors everything.
- **Multi-channel presence**: Meeting users where they are (messaging apps) drove viral adoption.

**What failed:**
- **Security as afterthought**: 40K exposed instances, 800+ malicious packages, prompt injection via email.
- **Unbounded agent autonomy**: The "lethal trifecta" without isolation guarantees.
- **Monolithic codebase**: OpenClaw's 430K lines spawned 5+ lightweight alternatives (Nanobot: 4K lines). Modular, composable architectures scale better.
- **LLM-driven orchestration**: Agents deciding workflow flow is an anti-pattern. Deterministic pipelines with LLMs at leaf nodes are far more reliable.
- **Trust-by-default marketplaces**: Community skills/tools need automated security scanning before installation.

**What this means for QE:**
The dispatcher is already designed as a deterministic orchestration engine (not an LLM service). The bus architecture already isolates services. The belief ledger already has supersession and temporal awareness. QE's architecture is structurally positioned to avoid OpenClaw's failures — but only if security hardening, cost optimization, and the Tool Discovery pattern are implemented as first-class concerns.

---

## Part 3: Implementation Phases

### Phase 1: Foundation Hardening (Week 1-2)

**Goal:** Make the existing code production-solid and add the infrastructure primitives that everything else depends on.

#### 1.1 Bus Delivery Confirmation

**File:** `src/qe/bus/memory_bus.py`

The bus currently fires and forgets. `publish()` creates async tasks but doesn't return them. This blocks testing and makes it impossible for the dispatcher to await subtask delivery.

**Changes:**
- `MemoryBus.publish()` returns `list[asyncio.Task]` — the created handler tasks
- Add `async def publish_and_wait(envelope) -> list[Any]` that publishes and awaits all handler completions
- Add `async def publish_and_wait_first(envelope) -> Any` for request-reply patterns
- Update `bus/protocol.py` to add new topics (see 1.5)

#### 1.2 Structured Logging

**All files with `print()` or bare `log.info()`**

Replace ad-hoc logging with structured, consistent logging:
- Every envelope publication logs: `topic`, `source_service_id`, `envelope_id`, `correlation_id`
- Every LLM call logs: `model`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`
- Every service handler invocation logs: `service_id`, `envelope_id`, `topic`, `duration_ms`
- Add `--verbose` flag support throughout (already exists in CLI, extend to all log points)

#### 1.3 Budget Persistence

**File:** `src/qe/runtime/budget.py`

BudgetTracker is currently in-memory only — spend data is lost on restart.

**Changes:**
- Add `async def save(self, ledger: BeliefLedger)` and `async def load(self, ledger: BeliefLedger)` methods
- Store budget records as a dedicated table in the belief ledger SQLite
- Load on startup, save after every `record_cost()` call (debounced to avoid write amplification)
- Store per-call records: `(timestamp, model, tokens_in, tokens_out, cost_usd, service_id, envelope_id)`

#### 1.4 Belief Ledger Enhancements

**File:** `src/qe/substrate/belief_ledger.py`

- FTS5 search is already implemented with triggers — verify coverage and add ranking improvements
- Add `get_claim_by_id(claim_id: str) -> Claim | None` method (currently requires fetching all claims)
- Add `count_claims()` method for dashboard stats
- Add `get_claims_since(timestamp: datetime)` for incremental queries

#### 1.5 New Bus Topics

**File:** `src/qe/bus/protocol.py`

Add topics for the task decomposition and reliability layers:

```python
# Goal orchestration
"goals.submitted",
"goals.enriched",
"goals.completed",
"goals.failed",

# Task decomposition
"tasks.planned",
"tasks.dispatched",
"tasks.completed",
"tasks.verified",
"tasks.verification_failed",
"tasks.recovered",
"tasks.failed",
"tasks.progress",
"tasks.checkpoint",

# Fact checking
"claims.challenged",
"claims.verification_requested",

# Memory
"memory.updated",
"memory.preference_set",

# Analysis
"analysis.requested",
"analysis.completed",

# Synthesis
"synthesis.requested",
"synthesis.completed",
```

#### 1.6 Lint and Debt Cleanup

- Run `ruff check --fix src/` and resolve all violations
- Verify all `datetime` calls use `datetime.now(UTC)` (no `utcnow()`)
- Delete empty service directories if any remain
- Fix any hardcoded paths or `YOUR_USERNAME` references

**Deliverable:** All tests green, zero lint violations, structured logging, persistent budget tracking, awaitable bus, FTS5 search enhancements.

**Exit Criteria (all must pass):**
- [ ] `publish_and_wait()` returns handler results for 100% of test envelopes
- [ ] Budget survives restart: stop engine, restart, verify spend data intact
- [ ] Structured logs for every LLM call include model, tokens, latency, cost
- [ ] Zero lint violations on `ruff check src/`
- [ ] FTS5 search returns ranked results for 10 test queries with >80% relevance

---

### Phase 2: Embedding Layer and Vector Store (Week 2-3)

**Goal:** Give the belief ledger a semantic dimension. This is the foundation for contradiction detection, hallucination flagging, context enrichment, and semantic search.

#### 2.1 Embedding Infrastructure

**New file:** `src/qe/substrate/embeddings.py`

```python
class EmbeddingStore:
    """Vector store backed by SQLite + numpy for zero-dependency operation.
    Optional upgrade to FAISS or Qdrant for large-scale deployments."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding via configured model."""
        # Default: local model via sentence-transformers (free)
        # Optional: text-embedding-3-small via API

    async def store(self, id: str, text: str, metadata: dict) -> None:
        """Embed and store a vector with metadata."""

    async def search(self, query: str, top_k: int = 10,
                     min_similarity: float = 0.5) -> list[SearchResult]:
        """Semantic similarity search."""

    async def find_contradictions(self, claim_text: str,
                                   threshold: float = 0.8) -> list[Contradiction]:
        """Find claims that are semantically similar but potentially contradictory."""
```

**Implementation notes:**
- Default embedding model: `all-MiniLM-L6-v2` via `sentence-transformers` (free, runs locally, 384 dimensions)
- Vector storage: SQLite table with BLOB column for embeddings, cosine similarity computed in Python
- Optional upgrade path: FAISS index for >100k vectors, or external Qdrant/pgvector
- Auto-embed on `claims.committed` — subscribe to bus topic and embed every new claim
- Batch embed existing claims on first startup

#### 2.2 Belief Ledger Integration

**File:** `src/qe/substrate/belief_ledger.py`

- Add `EmbeddingStore` as an optional component of the belief ledger
- `commit_claim()` triggers embedding storage after SQLite write
- `search_semantic(query: str, top_k: int)` replaces or supplements FTS5 search
- `find_contradictions(claim: Claim)` checks new claims against existing embeddings

#### 2.3 Dependencies

Add to `pyproject.toml`:
```
"sentence-transformers>=3.0",
"numpy>=1.26",
```

Optional extras:
```
[project.optional-dependencies]
vectors = ["faiss-cpu>=1.8", "qdrant-client>=1.8"]
```

**Deliverable:** Every claim is embedded on commit. Semantic search works. Contradiction detection finds claims that are similar but potentially conflicting.

**Exit Criteria (all must pass):**
- [ ] Embedding latency <200ms per claim on CPU (sentence-transformers)
- [ ] Semantic search recall >70% on 50 test queries (vs. manual relevance labels)
- [ ] Contradiction detection flags >80% of intentionally contradictory claim pairs in test set
- [ ] Batch embedding of 1000 existing claims completes in <5 minutes

---

### Phase 3: Task Decomposition Engine (Week 3-5)

**Goal:** Build the planner, dispatcher, and execution contract system that transforms the engine from "process one observation" to "achieve multi-step goals."

#### 3.1 New Models

**New file:** `src/qe/models/goal.py`

```python
class ExecutionContract(BaseModel):
    """Machine-checkable success criteria for a subtask."""
    preconditions: list[str] = []  # What must be true before execution
    postconditions: list[str] = []  # What must be true after (acceptance criteria)
    timeout_seconds: int = 120
    max_retries: int = 3
    fallback_strategy: str | None = None  # "alternative_model", "alternative_approach", etc.

class Subtask(BaseModel):
    subtask_id: str
    description: str
    task_type: Literal["research", "analysis", "fact_check", "synthesis",
                       "code_execution", "web_search", "document_generation"]
    depends_on: list[str] = []  # subtask_ids that must complete first
    model_tier: Literal["fast", "balanced", "powerful", "local"]
    tools_required: list[str] = []
    contract: ExecutionContract
    assigned_service_id: str | None = None
    assigned_model: str | None = None

class GoalDecomposition(BaseModel):
    goal_id: str
    original_description: str
    strategy: str  # Human-readable explanation of the approach
    subtasks: list[Subtask]
    assumptions: list[str] = []
    estimated_cost_usd: float = 0.0
    estimated_time_seconds: int = 0

class SubtaskResult(BaseModel):
    subtask_id: str
    goal_id: str
    status: Literal["completed", "failed", "recovered"]
    output: dict[str, Any]
    model_used: str
    tokens_used: dict[str, int]  # {"input": N, "output": M}
    latency_ms: int
    cost_usd: float
    tool_calls: list[dict] = []
    verification_result: dict | None = None
    recovery_history: list[dict] = []

class GoalState(BaseModel):
    goal_id: str
    status: Literal["planning", "executing", "completed", "failed", "paused"]
    decomposition: GoalDecomposition | None = None
    subtask_states: dict[str, str] = {}  # subtask_id -> status
    subtask_results: dict[str, SubtaskResult] = {}
    checkpoints: list[str] = []  # checkpoint_ids
    created_at: datetime
    completed_at: datetime | None = None
```

#### 3.2 Planner Service

**New file:** `src/qe/services/planner/service.py`

Subscribes to `goals.enriched`. Produces a `GoalDecomposition` DAG.

The planner operates in two stages — **representation then decomposition** — because research shows 70%+ of solution quality is determined by how the problem is represented before solving begins.

**Stage 1: Problem Representation (before any decomposition)**

The planner's system prompt requires it to complete this analysis first:

```
PROBLEM REPRESENTATION PROTOCOL:

1. RESTATE: What is the core problem in one sentence?
2. DISTINGUISH: What is being asked (X) vs. what actually needs to happen (Y)
   vs. the underlying need (Z)? Solve for Z when possible.
3. CONSTRAINTS: What are the hard limits? What cannot change?
4. SUCCESS CRITERIA: What does a correct solution look like? How will we verify it?
5. PROBLEM TYPE: Is this:
   - Well-defined (clear start/goal/operators)? → standard decomposition
   - Ill-defined (fuzzy goals)? → clarification-first, then decomposition
   - A wicked problem (solution changes the problem)? → probe-observe-reframe loop
6. CONTRADICTIONS: Does the goal contain contradictions?
   - "I need A but A causes B" → apply separation (time/space/condition/level)
7. SEARCH STRATEGY: What approach fits?
   - More information than goals → forward chaining (gather → analyze → conclude)
   - Clear goal, unclear path → backward chaining (define success → trace back)
   - Both but large gap → means-end analysis (iterative gap reduction)
   - Structurally similar to known problem → analogical reasoning
```

The representation output is stored as part of the `GoalDecomposition` and is visible in the dashboard. If the representation reveals the user's literal request (X) differs from their underlying need (Z), the planner can reframe the goal — or escalate to HIL for clarification.

**Stage 2: Decomposition**

Only after representation is complete, the planner:
1. Decomposes into the minimum number of subtasks (TRIZ ideality principle)
2. Identifies dependencies between subtasks
3. Assigns model tiers based on task complexity
4. Generates execution contracts with specific, checkable postconditions
5. Lists assumptions that could invalidate the plan
6. Consults the failure knowledge base for known avoidance rules

**Free model adaptation:**
- For powerful models: single prompt that produces the full DAG
- For balanced models: two-step — first generate strategy, then elaborate subtasks
- For local models: multi-step chain — identify task types, then generate one subtask at a time, then link dependencies

The planner also consults:
- The failure knowledge base: avoid known failure patterns
- Past successful decompositions: use as few-shot examples (retrieval from embeddings)
- Available services and their capabilities: only plan subtasks that registered services can handle

**New file:** `src/qe/services/planner/schemas.py` — Pydantic schemas for planner LLM output

**New genome:** `genomes/planner.toml`

#### 3.3 Dispatcher Service

**New file:** `src/qe/services/dispatcher/service.py`

Subscribes to `tasks.planned`. Manages the goal's execution lifecycle:

```python
class Dispatcher:
    """Manages goal execution by tracking the subtask dependency graph,
    dispatching ready subtasks, handling completion/failure, and
    maintaining checkpoints."""

    # Core state
    _active_goals: dict[str, GoalState]

    async def handle_planned(self, envelope: Envelope) -> None:
        """Receive a DAG from the planner and begin execution."""
        # Parse GoalDecomposition from envelope
        # Initialize GoalState
        # Identify ready subtasks (no dependencies)
        # Create initial checkpoint
        # Dispatch ready subtasks

    async def handle_subtask_verified(self, envelope: Envelope) -> None:
        """A subtask has been verified. Update graph and dispatch dependents."""
        # Mark subtask as complete
        # Store result
        # Create checkpoint
        # Identify newly-ready subtasks
        # If all subtasks complete: publish goals.completed
        # Dispatch ready subtasks

    async def handle_subtask_failed(self, envelope: Envelope) -> None:
        """A subtask has permanently failed after recovery attempts."""
        # Determine blast radius (dependent subtasks)
        # Attempt re-planning of affected subgraph
        # If re-planning fails: escalate to HIL or fail the goal

    async def dispatch_subtask(self, goal_id: str, subtask: Subtask) -> None:
        """Route a subtask to the appropriate service."""
        # Consult routing optimizer for model selection
        # Consult calibration table for expected reliability
        # Publish tasks.dispatched with full context
```

The dispatcher is **not an LLM service** — it's a deterministic orchestration engine. It doesn't call models. It manages state, tracks dependencies, and routes work. This makes it fast, reliable, and cheap.

**New genome:** `genomes/dispatcher.toml` (minimal — no model preference needed)

#### 3.4 Goal Persistence

**New file:** `src/qe/substrate/goal_store.py`

Goals and their state must survive restarts:

```python
class GoalStore:
    """Persists goal state to SQLite for crash recovery."""

    async def save_goal(self, state: GoalState) -> None
    async def load_goal(self, goal_id: str) -> GoalState | None
    async def list_active_goals(self) -> list[GoalState]
    async def save_checkpoint(self, goal_id: str, checkpoint: dict) -> str
    async def load_checkpoint(self, goal_id: str, checkpoint_id: str) -> dict
```

Add new SQLite tables via migration:
```sql
CREATE TABLE goals (
    goal_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    decomposition JSON,
    subtask_states JSON,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    subtask_states JSON,
    subtask_results JSON,
    created_at TIMESTAMP,
    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
);
```

#### 3.5 Supervisor Updates

**File:** `src/qe/kernel/supervisor.py`

- Update `_instantiate_service()` to dispatch planner and dispatcher service types
- Register the planner and dispatcher genomes
- Add goal-level status to the health endpoint

#### 3.6 CLI Updates

**File:** `src/qe/cli/main.py`

```bash
qe goal "Research fusion energy and produce a briefing"  # Submit a goal
qe goal list                                              # List active goals
qe goal status <goal_id>                                  # Show DAG status
qe goal pause <goal_id>                                   # Pause execution
qe goal resume <goal_id>                                  # Resume paused goal
qe goal cancel <goal_id>                                  # Cancel with cleanup
```

#### 3.7 API Updates

**File:** `src/qe/api/app.py`

```
POST /api/goals          — Submit a new goal
GET  /api/goals          — List all goals with status
GET  /api/goals/{id}     — Goal detail with DAG and subtask states
POST /api/goals/{id}/pause
POST /api/goals/{id}/resume
POST /api/goals/{id}/cancel
GET  /api/goals/{id}/dag  — DAG structure for visualization
```

**Deliverable:** `qe goal "..."` decomposes into subtasks, dispatches them to services, tracks dependencies, and produces results. Goals persist across restarts.

**Exit Criteria (all must pass):**
- [ ] 10 test goals decompose into valid DAGs with no orphan subtasks
- [ ] Dispatcher correctly respects dependency ordering (no subtask runs before its dependencies)
- [ ] Goal state survives restart: kill engine mid-goal, restart, verify goal resumes from checkpoint
- [ ] Planner produces representation analysis before decomposition for 100% of goals
- [ ] End-to-end: `qe goal "Research X and summarize"` completes without manual intervention on ≥3/5 attempts

---

### Phase 4: Verification and Recovery (Week 5-7)

**Goal:** Build the reliability infrastructure that makes autonomous operation trustworthy.

#### 4.1 Verification Service

**New file:** `src/qe/services/verification/service.py`

Subscribes to `tasks.completed`. Runs the verification pipeline on every subtask output:

**Structural checks (deterministic, no LLM):**
- Output parses as expected schema
- Required fields present
- Types correct
- No empty required fields

**Contract checks (deterministic, no LLM):**
- Evaluate postconditions from the execution contract
- Postconditions are expressed as simple, parseable rules:
  - `"result.length >= 3"` — minimum result count
  - `"all(item.source_url for item in result.claims)"` — all claims have sources
  - `"result.entity_count == 5"` — exact count
- Implement a safe postcondition evaluator (restricted Python expressions, no arbitrary code execution)

**Anomaly checks (statistical, no LLM):**
- Output token count vs. historical distribution for this (task_type, model)
- Confidence score distribution vs. historical
- Latency vs. historical
- Source count vs. historical
- Flag outputs that are >2 standard deviations from baseline

**Belief ledger checks (embedding-based):**
- Embed each claim in the output
- Search for contradictions against existing claims
- Flag claims that contradict high-confidence existing beliefs

**Adversarial question review (LLM-based, dual-process System 2 analog):**
The verification service implements the adversarial self-review pattern from question science:

```python
# Stage 1: The generation already happened (System 1 — fast draft)
# Stage 2: Deliberate adversarial review (System 2)

adversarial_prompt = """
You are a rigorous fact-checker. Your ONLY job is to find problems.

Given this output, answer these questions:
1. ASSUMPTION EXCAVATION: What is being assumed that hasn't been verified?
   What constraints are treated as given that were never stated?
2. CAUSAL DEPTH: For each key claim — do we know WHY this is true,
   or just THAT it appears in text? Is the explanation at surface level
   or root cause level?
3. STEELMAN THE OPPOSITION: What's the strongest argument against the
   main conclusion? What would someone who disagrees say?
4. FALSIFICATION: What would have to be true for this output to be wrong?
5. CONFIDENCE AUDIT: List every factual claim. Classify each as:
   [VERIFIED] [INFERRED] [ESTIMATED] [UNCERTAIN] [UNKNOWN]
6. WEAKEST LINK: Which 2-3 claims are most likely wrong and why?

Do NOT be charitable. Be adversarial.
"""
```

This is the most impactful single intervention for hallucination reduction (~30-40% reduction, per research). It runs on every subtask output for local/fast models, and on flagged outputs for balanced/powerful models.

**External hallucination detection (optional, for production deployments):**
- Integrate Vectara HHEM (free, open-source) for grounded hallucination scoring
- For each claim with a source document, HHEM scores consistency between source and claim
- Claims scoring below threshold trigger retry or adversarial review
- Only applicable when source documents are available (RAG-style tasks)

**Self-consistency sampling (for high-stakes subtasks):**
Run the same subtask on 2-3 different models or with different temperatures. Compare outputs:
- Where outputs agree = high-confidence knowledge (stable across sampling)
- Where outputs disagree = hallucination zone (flag for review or additional verification)
This directly trades compute for reliability and is most valuable for critical analytical conclusions.

Publishes `tasks.verified` or `tasks.verification_failed` with detailed report.

**New genome:** `genomes/verification.toml`

#### 4.2 Recovery Orchestrator

**New file:** `src/qe/services/recovery/service.py`

Subscribes to `tasks.verification_failed` and `tasks.failed`.

**Failure classification:**
```python
class FailureClass(str, Enum):
    TRANSIENT = "transient"          # Timeout, rate limit, network error
    CAPABILITY = "capability"        # Model produced wrong/low-quality output
    APPROACH = "approach"            # Method doesn't work for this task
    SPECIFICATION = "specification"  # Execution contract is wrong
    UNRECOVERABLE = "unrecoverable" # Nothing worked
```

**Recovery strategies:**
- `TRANSIENT` → retry same configuration after backoff
- `CAPABILITY` → retry with more capable model (escalate tier)
- `APPROACH` → retry with alternative approach (different tools, different prompt strategy)
- `SPECIFICATION` → escalate to planner for re-decomposition of this subtask
- `UNRECOVERABLE` → escalate to HIL with full context

**Recovery execution:**
```python
async def attempt_recovery(self, failure: FailureReport) -> RecoveryResult:
    # 1. Classify failure
    failure_class = self.classify(failure)

    # 2. Consult failure knowledge base for known strategies
    strategies = self.failure_kb.lookup(failure_class, failure.task_type)

    # 3. Try strategies in order of historical success rate
    for strategy in strategies:
        if failure.retry_count >= failure.contract.max_retries:
            break
        result = await self.execute_strategy(strategy, failure)
        if result.success:
            self.failure_kb.record(failure, strategy, success=True)
            return result
        self.failure_kb.record(failure, strategy, success=False)

    # 4. Escalate to HIL
    return await self.escalate_to_hil(failure)
```

**New genome:** `genomes/recovery.toml`

#### 4.3 Failure Knowledge Base

**New file:** `src/qe/substrate/failure_kb.py`

```python
class FailureKnowledgeBase:
    """Stores and retrieves failure patterns and recovery strategies."""

    async def record(self, failure: FailureReport, strategy: str,
                     success: bool) -> None:
        """Record a failure and the outcome of a recovery attempt."""

    async def lookup(self, failure_class: str, task_type: str,
                     top_k: int = 5) -> list[RecoveryStrategy]:
        """Find the most effective recovery strategies for this failure type."""

    async def get_avoidance_rules(self, task_type: str) -> list[AvoidanceRule]:
        """Get rules the planner should follow to avoid known failures."""
```

SQLite table:
```sql
CREATE TABLE failure_records (
    failure_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    model_used TEXT,
    failure_class TEXT NOT NULL,
    error_summary TEXT,
    context_fingerprint TEXT,  -- hash of task context for deduplication
    recovery_strategy TEXT,
    recovery_succeeded BOOLEAN,
    created_at TIMESTAMP,
    goal_id TEXT,
    subtask_id TEXT
);

CREATE INDEX idx_failure_lookup ON failure_records(failure_class, task_type);
```

#### 4.4 Checkpoint Manager

**New file:** `src/qe/services/checkpoint/manager.py`

```python
class CheckpointManager:
    """Creates and manages execution checkpoints for rollback."""

    async def create_checkpoint(self, goal_id: str,
                                 goal_state: GoalState) -> str:
        """Save current goal state as a checkpoint. Returns checkpoint_id."""

    async def rollback_to(self, goal_id: str,
                           checkpoint_id: str) -> GoalState:
        """Restore goal state to a specific checkpoint."""

    async def find_rollback_point(self, goal_id: str,
                                   failed_subtask_id: str) -> str:
        """Determine the correct checkpoint to roll back to, considering
        the dependency graph and which subtasks are affected."""
```

Checkpoints are stored in the `checkpoints` table (from Phase 3). The checkpoint manager traces the dependency graph backward from the failed subtask to find the latest safe state.

#### 4.5 Verification Strictness by Model Tier

The verification service adjusts its thoroughness based on the model that produced the output:

```python
VERIFICATION_PROFILES = {
    "powerful": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "cross_reference_web": False,  # Trust powerful models more
        "multi_model_consensus": False,
        "adversarial_verification": False,
    },
    "balanced": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "cross_reference_web": "sample",  # Check 2-3 key claims
        "multi_model_consensus": False,
        "adversarial_verification": False,
    },
    "fast": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "cross_reference_web": "all",  # Check every claim
        "multi_model_consensus": True,  # Run on 2 models, compare
        "adversarial_verification": True,
    },
    "local": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "cross_reference_web": "all",
        "multi_model_consensus": True,
        "adversarial_verification": True,
        "source_url_verification": True,  # Actually fetch URLs and verify
    },
}
```

**Deliverable:** Every subtask output is verified before acceptance. Failures are classified and recovered automatically. The system learns from failures. Checkpoints enable rollback.

**Exit Criteria (all must pass):**
- [ ] Structural + contract checks catch >90% of intentionally malformed outputs in test set
- [ ] Recovery succeeds on ≥50% of transient and capability failures (retry + model escalation)
- [ ] False-positive gate rate <15% (verification doesn't reject valid outputs too often)
- [ ] Failure KB records at least 20 failure-recovery pairs after running test suite
- [ ] End-to-end goal success rate >60% on 20-goal test suite (up from ~30% without verification)

---

### Phase 5: Tool Integration Framework & Security (Week 7-9)

**Goal:** Give services the ability to interact with the world — safely, efficiently, and with cost-aware context management.

#### 5.1 Tool Registry with Discovery Pattern

**New file:** `src/qe/runtime/tools.py`

The Tool Discovery pattern (validated by the OpenClaw community with claims of up to 90% context savings) ensures tool schemas don't bloat every LLM call:

```python
class ToolRegistry:
    """Plugin-based tool registry with capability enforcement and lazy discovery."""

    _tools: dict[str, ToolSpec]

    def register(self, tool: ToolSpec) -> None

    def get_tools_for_service(self, blueprint: Blueprint) -> list[ToolSpec]:
        """Return only tools this service has declared capability for."""

    def get_tool_schemas(self, blueprint: Blueprint,
                          mode: str = "discovery") -> list[dict]:
        """Return tool schemas for LLM function calling.

        mode="discovery": Returns only the meta-tool (search_available_tools)
            to minimize context. The LLM describes what it needs, and specific
            tools are loaded on-demand.
        mode="direct": Returns all available tool schemas (for simple tasks
            where the tool set is small and known).
        mode="relevant": Returns tools pre-filtered by task_type from the
            subtask's execution contract.
        """

    def search_tools(self, description: str, blueprint: Blueprint) -> list[ToolSpec]:
        """Find tools matching a natural language description,
        filtered by service capabilities."""

class ToolSpec:
    name: str
    description: str
    requires_capability: str  # Maps to CapabilityDeclaration field
    input_schema: dict        # JSON schema for parameters
    output_schema: dict       # JSON schema for return value
    handler: Callable         # The actual async function
    timeout_seconds: int = 30
    ephemeral_output: bool = True  # If True, raw output is discarded after extraction
    category: str = ""        # For discovery grouping
```

**The meta-tool for discovery mode:**
```python
@tool(name="search_available_tools", requires_capability=None)  # Always available
async def search_available_tools(description: str) -> list[dict]:
    """Describe what you need to do, and this tool will return the specific
    tools available for that task with their full schemas."""
    return tool_registry.search_tools(description, current_blueprint)
```

This pattern means a service with 20 available tools only injects 1 tool schema into the base context. The LLM discovers specific tools only when it needs them.

#### 5.2 Built-in Tools

**New directory:** `src/qe/tools/`

**`tools/web_search.py`** — Web search via SearXNG (self-hostable, free) or Brave Search API
```python
@tool(name="web_search", requires_capability="web_search")
async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    # Try SearXNG first (free, self-hosted)
    # Fall back to Brave Search API (free tier: 2000 queries/month)
    # Fall back to DuckDuckGo HTML scraping (free, no API key)
```

**`tools/web_fetch.py`** — HTTP GET with content extraction
```python
@tool(name="web_fetch", requires_capability="web_search")
async def web_fetch(url: str) -> WebPage:
    # Fetch URL with httpx
    # Extract main content with trafilatura or readability-lxml
    # Return title, text, metadata
```

**`tools/file_ops.py`** — Sandboxed file read/write
```python
@tool(name="file_read", requires_capability="file_read")
async def file_read(path: str) -> str:
    # Path is relative to goal workspace: data/workspaces/{goal_id}/
    # Enforced by sandbox — cannot escape workspace directory

@tool(name="file_write", requires_capability="file_write")
async def file_write(path: str, content: str) -> FileWriteResult:
    # Same sandboxing
```

**`tools/code_execute.py`** — Sandboxed Python execution
```python
@tool(name="code_execute", requires_capability="code_execute")
async def code_execute(code: str, timeout_seconds: int = 30) -> CodeResult:
    # Execute in subprocess with:
    # - ulimit memory (256MB default)
    # - ulimit CPU time
    # - No network access
    # - Working directory scoped to goal workspace
    # Returns stdout, stderr, return_code
```

#### 5.3 Tool Integration with BaseService

**File:** `src/qe/runtime/service.py`

Update `BaseService` to inject available tools into LLM calls:

```python
class BaseService:
    def __init__(self, blueprint, bus, substrate, tool_registry=None):
        ...
        self.tools = tool_registry.get_tools_for_service(blueprint) if tool_registry else []

    async def _call_llm(self, model, messages, schema, tools=None):
        """Enhanced LLM call with optional tool use."""
        # If tools provided, use instructor's tool calling support
        # Handle tool call results and feed back into conversation
        # Track all tool calls in the response metadata
```

#### 5.4 Workspace Management

**New file:** `src/qe/runtime/workspace.py`

```python
class WorkspaceManager:
    """Manages per-goal sandboxed workspaces."""

    def create(self, goal_id: str) -> Path:
        """Create workspace at data/workspaces/{goal_id}/"""

    def cleanup(self, goal_id: str) -> None:
        """Remove workspace after goal completion (or archive)."""

    def sandbox_path(self, goal_id: str, relative_path: str) -> Path:
        """Resolve a path within the sandbox. Raises if escape attempted."""
```

#### 5.5 Dependencies

Add to `pyproject.toml`:
```
"httpx>=0.27",
"trafilatura>=1.8",
```

Optional:
```
search = ["searxng-client>=1.0"]
```

#### 5.6 Bus-Level Tool Call Gating

**New file:** `src/qe/runtime/tool_gate.py`

Inspired by the AgentGate pattern from the OpenClaw ecosystem — an interceptor that validates every tool invocation against security policies before execution:

```python
class ToolGate:
    """Security policy enforcement for all tool invocations.
    Runs as code-level enforcement, not prompt-level — cannot be
    bypassed by prompt injection."""

    def __init__(self, policies: list[SecurityPolicy]):
        self._policies = policies

    async def validate(self, service_id: str, tool_name: str,
                        params: dict, blueprint: Blueprint) -> GateDecision:
        """Validate a tool call against all security policies.
        Returns ALLOW, DENY, or ESCALATE (to HIL)."""

        # 1. Capability check — does this service declare this capability?
        if not blueprint.capabilities.allows(tool_name):
            return GateDecision.DENY("capability_not_declared")

        # 2. Parameter validation — are params within allowed ranges?
        if not self._validate_params(tool_name, params):
            return GateDecision.DENY("invalid_parameters")

        # 3. Rate limiting — has this service exceeded its call budget?
        if self._rate_exceeded(service_id, tool_name):
            return GateDecision.DENY("rate_limit_exceeded")

        # 4. Scope check — is the target within allowed scope?
        # (e.g., file paths within workspace, URLs not in blocklist)
        if not self._scope_check(tool_name, params):
            return GateDecision.DENY("scope_violation")

        # 5. Policy-specific checks
        for policy in self._policies:
            decision = policy.evaluate(service_id, tool_name, params)
            if decision != GateDecision.ALLOW:
                return decision

        return GateDecision.ALLOW()

class SecurityPolicy:
    """Base class for pluggable security policies."""
    def evaluate(self, service_id: str, tool_name: str,
                  params: dict) -> GateDecision: ...
```

The gate wraps every tool handler invocation. No tool call reaches execution without passing the gate. Policies are defined in `config.toml`:

```toml
[security]
enabled = true

[security.policies]
max_web_fetches_per_goal = 50
max_file_writes_per_goal = 20
blocked_domains = ["*.exe", "*.sh"]
require_hil_for = ["code_execute"]  # Always require human approval
```

#### 5.7 Input Sanitization

**New file:** `src/qe/runtime/sanitizer.py`

Every envelope entering the system from external sources is sanitized before reaching any LLM:

```python
class InputSanitizer:
    """Detects and neutralizes prompt injection in input payloads."""

    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"you are now",
        r"system:\s*",
        r"<\|.*?\|>",
        r"```system",
        # ... extensible pattern list
    ]

    def sanitize(self, text: str) -> SanitizeResult:
        """Scan text for injection patterns. Returns cleaned text
        and a risk score. High risk triggers HIL review."""

    def wrap_untrusted(self, text: str) -> str:
        """Wrap untrusted content in delimiters that the system prompt
        instructs the LLM to treat as data, not instructions."""
        return f"[UNTRUSTED_CONTENT_START]\n{text}\n[UNTRUSTED_CONTENT_END]"
```

Applied at:
- API `/api/submit` and `/api/goals` endpoints
- CLI `qe submit` and `qe goal` commands
- Inbox relay loop (cross-process envelopes)
- Future messaging channel adapters (Slack, Telegram, email)

#### 5.8 MCP Integration

**New file:** `src/qe/runtime/mcp_bridge.py`

Model Context Protocol (MCP) integration enables QE to use the 1000+ community MCP servers for tool access:

```python
class MCPBridge:
    """Connects to external MCP servers and exposes their tools
    through QE's tool registry with lazy discovery."""

    async def connect(self, server_uri: str) -> None:
        """Connect to an MCP server and register its tools."""

    async def discover_relevant(self, description: str) -> list[ToolSpec]:
        """Smart aggregation — find relevant tools across all connected
        MCP servers based on task description. Only activates the best
        matches, not all tools."""
```

MCP servers are configured in `config.toml`:

```toml
[mcp]
servers = [
    { name = "google-drive", uri = "npx @anthropic/mcp-google-drive" },
    { name = "github", uri = "npx @anthropic/mcp-github" },
    { name = "slack", uri = "npx @anthropic/mcp-slack" },
]
discovery_mode = "smart"  # "smart" = relevance-filtered, "all" = load everything
```

This gives QE access to Google Drive, Slack, GitHub, databases, and hundreds of other integrations without building custom tool implementations.

#### 5.9 Ephemeral Tool Output Management

**Enhancement to:** `src/qe/runtime/service.py`

Tool outputs are the second-largest source of context bloat (20-30% in OpenClaw). Solution: extract structured data immediately, discard raw output:

```python
async def _handle_tool_result(self, tool_name: str, raw_output: Any,
                                extraction_schema: type[BaseModel] | None = None) -> dict:
    """Process a tool result: extract structured data, discard raw output."""

    if extraction_schema:
        # Use LLM to extract structured data from raw output
        extracted = await self._call_llm(
            model=self.router.select("fast"),  # Use cheapest model for extraction
            messages=[{"role": "user", "content": f"Extract: {raw_output}"}],
            schema=extraction_schema,
        )
        # Only the extracted data enters the conversation context
        return {"extracted": extracted.model_dump(), "raw_discarded": True}
    else:
        # If no schema, truncate raw output to reasonable size
        truncated = str(raw_output)[:2000]
        return {"raw_truncated": truncated}
```

This prevents a 10,000-token web page from sitting in the conversation forever. Only the extracted claims/data persist.

**NOW scope (Weeks 5-6):** Web search, web fetch, file ops, code execute, security gating, input sanitization, ephemeral outputs. MCP integration is NEXT scope.

**Deliverable (NOW):** Services can search the web, fetch pages, read/write files, and execute code. All tools are capability-scoped, sandboxed, and security-gated. Input sanitization protects against prompt injection. Ephemeral output management prevents context bloat.

**Deliverable (NEXT):** Tool Discovery pattern minimizes context overhead. MCP integration provides access to 1000+ external tools.

**Exit Criteria — NOW (all must pass):**
- [ ] Web search returns results for 10 test queries via SearXNG or Brave fallback
- [ ] Code execution sandboxing prevents filesystem escape (10 adversarial test cases)
- [ ] Tool gate blocks 100% of capability-violation attempts in test set
- [ ] Input sanitizer detects >80% of prompt injection test patterns
- [ ] Ephemeral output reduces average context size by >30% vs. keeping raw outputs

**Exit Criteria — NEXT (all must pass):**
- [ ] Tool Discovery mode reduces tool-related context tokens by >50% vs. direct mode
- [ ] At least 3 MCP servers connected and functional (e.g., GitHub, Google Drive, Slack)

---

### Phase 6: Memory System (Week 9-10)

**Goal:** Build persistent memory that enriches every LLM call with relevant past knowledge.

#### 6.1 Memory Store

**New file:** `src/qe/substrate/memory_store.py`

```python
class MemoryEntry(BaseModel):
    memory_id: str
    category: Literal["preference", "context", "project", "entity", "pattern"]
    key: str
    value: str
    confidence: float = 1.0
    source: Literal["user_explicit", "inferred", "system"]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    superseded_by: str | None = None

class MemoryStore:
    """Three-layer persistent memory system."""

    async def set_preference(self, key: str, value: str) -> MemoryEntry
    async def get_preferences(self) -> list[MemoryEntry]

    async def set_project_context(self, project_id: str, key: str, value: str) -> MemoryEntry
    async def get_project_context(self, project_id: str) -> list[MemoryEntry]

    async def set_entity_memory(self, entity_id: str, key: str, value: str,
                                 confidence: float = 0.8) -> MemoryEntry
    async def get_entity_memories(self, entity_id: str) -> list[MemoryEntry]

    async def search(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Semantic search across all memory entries via embedding store."""
```

SQLite tables:
```sql
CREATE TABLE memory_entries (
    memory_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    expires_at TIMESTAMP,
    superseded_by TEXT
);

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    last_accessed TIMESTAMP
);

CREATE TABLE project_claims (
    project_id TEXT,
    claim_id TEXT,
    relevance_score REAL,
    PRIMARY KEY (project_id, claim_id)
);
```

#### 6.2 Memory Service

**New file:** `src/qe/services/memory/service.py`

Subscribes to `goals.submitted` (to enrich with context) and `claims.committed` (to infer memory entries).

**Enrichment flow (on `goals.submitted`):**
1. Load user preferences
2. Find relevant project context
3. Search belief ledger for related claims (via embeddings)
4. Search memory store for related entities
5. Search failure KB for relevant avoidance rules
6. Search past goals for similar successful decompositions
7. Publish `goals.enriched` with all context attached

**Inference flow (on `claims.committed`):**
- If claims repeatedly reference an entity, create entity memory
- If user submits goals on a topic cluster, infer project context
- Memory entries created by inference are explicitly marked `source: "inferred"` with lower confidence

#### 6.3 ContextManager Integration

**File:** `src/qe/runtime/context_manager.py`

Update `build_messages()` to inject memory context:
```python
def build_messages(self, envelope, turn_count, memory_context=None):
    messages = [system_prompt, constitution]

    if memory_context:
        # Inject user preferences
        if memory_context.preferences:
            messages.append({"role": "system", "content":
                f"[USER PREFERENCES]\n{format_preferences(memory_context.preferences)}"})

        # Inject relevant claims
        if memory_context.relevant_claims:
            messages.append({"role": "system", "content":
                f"[RELEVANT EXISTING KNOWLEDGE]\n{format_claims(memory_context.relevant_claims)}"})

    messages.extend(history)
    messages.append(format_envelope(envelope))
    # Token budget management across all context sources
    return self._compress_to_budget(messages)
```

#### 6.4 CLI and API

```bash
qe memory list
qe memory set preference output_format "detailed with citations"
qe memory forget <memory_id>
qe memory projects
qe memory project create "Fusion Energy Research"
```

API endpoints:
```
GET    /api/memory
POST   /api/memory/preferences
DELETE /api/memory/{memory_id}
GET    /api/projects
POST   /api/projects
```

**Deliverable:** The system remembers user preferences, tracks projects, enriches every LLM call with relevant context from past interactions. Memory entries are confidence-weighted and source-tracked.

**Exit Criteria (all must pass):**
- [ ] User preference set via CLI persists and appears in next LLM call's context
- [ ] Memory enrichment adds relevant claims to goal context (>60% relevance on 10 test goals)
- [ ] Inferred memory entries are correctly marked `source: "inferred"` with lower confidence
- [ ] Memory search returns semantically relevant results (>70% precision on test queries)

---

### Phase 7: New Service Types (Week 10-12)

**Goal:** Expand from one service type to a full roster covering the capability breadth needed for complex goals.

#### 7.1 Fact-Checker Service

**New file:** `src/qe/services/fact_checker/service.py`

Subscribes to `claims.proposed` (and optionally `claims.verification_requested`).

For each proposed claim:
1. Search the belief ledger for related existing claims
2. Search the web for corroborating or contradicting evidence
3. Compare confidence levels
4. Publish `claims.challenged` if contradictions found, or allow the claim through

The fact-checker is adversarial by design — its prompt instructs it to find reasons the claim might be wrong.

**New genome:** `genomes/fact_checker.toml`

#### 7.2 Analyst Service

**New file:** `src/qe/services/analyst/service.py`

Subscribes to `analysis.requested`. Takes collections of claims and produces:
- Trend detection across claims
- Contradiction identification within a claim set
- Gap analysis ("we know X but haven't investigated Y")
- Comparative analysis across entities

**New genome:** `genomes/analyst.toml`

#### 7.3 Writer Service

**New file:** `src/qe/services/writer/service.py`

Subscribes to `synthesis.requested`. Takes analysis results and claims, produces human-readable documents. Uses memory system for user's preferred writing style and format.

**New genome:** `genomes/writer.toml`

#### 7.4 Coder Service

**New file:** `src/qe/services/coder/service.py`

Handles `code_execution` subtask types. Generates code, runs it in the sandbox, returns results. Handles data processing, visualization, API interactions.

**New genome:** `genomes/coder.toml`

#### 7.5 Monitor Service

**New file:** `src/qe/services/monitor/service.py`

Long-running service for recurring tasks:
- Maintains a schedule of periodic observations
- Publishes observations on schedule
- Detects when new information contradicts existing claims
- Alerts via HIL when significant changes detected

```bash
qe monitor add "Check SpaceX news" --interval daily
qe monitor add "Track fusion energy funding" --interval weekly
qe monitor list
qe monitor remove <monitor_id>
```

**New genome:** `genomes/monitor.toml`

#### 7.6 Supervisor Updates

**File:** `src/qe/kernel/supervisor.py`

Update `_instantiate_service()` to handle all new service types. Transition from if/elif dispatch to a registry-based pattern:

```python
SERVICE_CLASSES = {
    "researcher": ResearcherService,
    "hil": HILService,
    "planner": PlannerService,
    "dispatcher": DispatcherService,
    "verification": VerificationService,
    "recovery": RecoveryOrchestrator,
    "memory": MemoryService,
    "fact_checker": FactCheckerService,
    "analyst": AnalystService,
    "writer": WriterService,
    "coder": CoderService,
    "monitor": MonitorService,
}

def _instantiate_service(self, blueprint):
    for prefix, cls in SERVICE_CLASSES.items():
        if blueprint.service_id.startswith(prefix):
            return cls(blueprint, self.bus, self.substrate)
    raise ValueError(f"No service class for {blueprint.service_id}")
```

**Deliverable:** Six new service types with genome templates and tests. The system can research, fact-check, analyze, write, code, and monitor.

**Exit Criteria (all must pass):**
- [ ] Each service handles ≥5 test envelopes without error
- [ ] Fact-checker challenges at least 1 in 5 intentionally flawed claims in test set
- [ ] Writer produces output that passes the verification service's structural checks
- [ ] Coder generates and executes code that produces correct output for 5 test tasks
- [ ] Monitor fires scheduled observations within ±10% of configured interval

---

### Phase 8: Learning Loops and Calibration (Week 12-14)

**Goal:** Make the system get better over time through five learning loops.

#### 8.1 Confidence Calibration Tracker

**New file:** `src/qe/runtime/calibration.py`

```python
class CalibrationTracker:
    """Tracks the relationship between reported confidence and actual accuracy
    for each (model, task_type) pair."""

    async def record(self, model: str, task_type: str,
                     reported_confidence: float, actual_correct: bool) -> None:
        """Record a calibration data point from verification results."""

    def calibrated_confidence(self, model: str, task_type: str,
                               raw_confidence: float) -> float:
        """Adjust a raw confidence score based on historical calibration data."""
        # Bin raw confidence into buckets (0.5-0.6, 0.6-0.7, etc.)
        # Look up historical accuracy for this bucket
        # Return historical accuracy as calibrated confidence

    async def get_calibration_curve(self, model: str,
                                     task_type: str) -> list[tuple[float, float]]:
        """Return (reported, actual) pairs for calibration visualization."""
```

SQLite table:
```sql
CREATE TABLE calibration_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    reported_confidence REAL NOT NULL,
    actual_correct BOOLEAN NOT NULL,
    created_at TIMESTAMP,
    goal_id TEXT,
    subtask_id TEXT
);

CREATE INDEX idx_calibration ON calibration_records(model, task_type);
```

The verification service feeds this on every verified subtask. The confidence score that goes into the belief ledger is the calibrated confidence, not the raw model output.

#### 8.2 Routing Optimizer

**New file:** `src/qe/runtime/routing_optimizer.py`

```python
class RoutingOptimizer:
    """Data-driven model selection based on historical performance."""

    async def record_outcome(self, model: str, task_type: str,
                              success: bool, latency_ms: int,
                              cost_usd: float, quality_score: float) -> None:
        """Record the outcome of a model on a task type."""

    def score_query(self, task_type: str, context: QueryContext) -> list[ModelScore]:
        """Multi-factor query scoring inspired by Manifest's 23-dimension approach.
        Scores each available model across multiple dimensions in <2ms."""
        # Dimensions scored:
        # - Task complexity (simple lookup vs. multi-step reasoning)
        # - Required capabilities (tool calling, JSON mode, long context)
        # - Historical success rate for this (model, task_type)
        # - Cost per expected token count
        # - Latency vs. budget
        # - Current provider health/cooldown status
        # - Budget remaining (aggressive downgrade when <10%)
        # - Verification overhead (cheaper models need more verification)
        # Returns ranked list of models with composite scores

    def select_model(self, task_type: str, budget_remaining: float,
                     latency_budget_ms: int | None = None) -> str:
        """Select the best model using multi-factor scoring.
        Falls back to Thompson sampling for exploration."""
        # Multi-armed bandit: Thompson sampling
        # Exploitation: pick best-known performer
        # Exploration: occasionally try alternatives (epsilon = 0.1)
```

SQLite table:
```sql
CREATE TABLE routing_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    latency_ms INTEGER,
    cost_usd REAL,
    quality_score REAL,
    created_at TIMESTAMP
);

CREATE INDEX idx_routing ON routing_records(model, task_type);
```

Replaces the static `TIER_MODELS` dict in `runtime/router.py` with data-driven selection while keeping tier-based routing as the cold-start default.

#### 8.3 Planning Memory

**Enhancement to:** Memory Service

Store successful goal decompositions as reusable patterns:
```sql
CREATE TABLE planning_patterns (
    pattern_id TEXT PRIMARY KEY,
    goal_description_embedding BLOB,  -- for semantic similarity search
    decomposition JSON,               -- the successful DAG
    outcome TEXT,                      -- "success" | "partial" | "failed"
    total_cost_usd REAL,
    total_time_seconds INTEGER,
    created_at TIMESTAMP
);
```

The planner retrieves similar past patterns via embedding search and uses them as few-shot examples.

#### 8.4 HIL Learning (Active Learning)

**New file:** `src/qe/runtime/hil_predictor.py`

```python
class HILPredictor:
    """Predicts whether a human will approve or reject an output,
    trained on historical HIL decisions."""

    def predict(self, output_features: dict) -> tuple[str, float]:
        """Returns (predicted_decision, confidence).
        Features: task_type, model, confidence_score, output_length,
        source_count, anomaly_flags."""

    def should_skip_hil(self, output_features: dict) -> bool:
        """Returns True if the predictor is confident the human will approve.
        Only returns True above 95% prediction confidence."""

    async def train(self, features: dict, actual_decision: str) -> None:
        """Update the model with a new labeled example."""
```

Implementation: Logistic regression on output features. Training data comes from every HIL approve/reject decision. The model is retrained after every N decisions (start with N=20). No external ML dependencies — pure numpy/scipy.

When `should_skip_hil()` returns True for a category of output, the system routes those outputs past HIL review, freeing human attention for genuinely uncertain cases.

#### 8.5 Bayesian Confidence Propagation ⚗️ RESEARCH BET

> **Defer until the belief ledger has ≥100 claims with dependency chains.** Start with simple multiplicative propagation (multiply parent confidences along paths). Only introduce pgmpy if the simple approach produces demonstrably wrong results.

**New file:** `src/qe/substrate/confidence_network.py`

```python
class ConfidenceNetwork:
    """Bayesian network over belief ledger claims for principled
    uncertainty propagation."""

    def add_claim(self, claim_id: str, confidence: float,
                   depends_on: list[str] = []) -> None:
        """Add a claim node to the network."""

    def propagate(self) -> dict[str, float]:
        """Recompute all confidences based on dependency structure.
        Returns claim_id -> propagated_confidence."""

    def impact_of_change(self, claim_id: str,
                          new_confidence: float) -> dict[str, float]:
        """Compute how changing one claim's confidence affects dependents."""
```

Implementation: Simple belief propagation on a directed graph. Dependencies come from `source_envelope_ids` and `causation_id` chains. When a leaf claim's confidence changes (new evidence), propagation updates all dependent claims.

Add `pgmpy` to optional dependencies:
```
bayesian = ["pgmpy>=0.0.4"]
```

Fall back to manual propagation if pgmpy is not installed (multiply parent confidences along dependency paths — crude but functional).

**Deliverable:** Five learning loops operating continuously. The system gets measurably better over time without model changes.

**Exit Criteria (all must pass):**
- [ ] Calibration curve shows <10% deviation between reported and actual confidence after 100+ data points
- [ ] Routing optimizer selects cheaper models for simple tasks (average cost/task decreases >20% over 50 goals)
- [ ] Planning memory retrieves relevant past decompositions (>60% similarity match on 10 test goals)
- [ ] Failure KB avoidance rules prevent at least 3 previously-seen failure patterns in test suite
- [ ] HIL predictor accuracy >80% on approve/reject prediction after 50+ training examples

---

### Phase 9: Free Model Optimization Layer (Week 14-15)

**Goal:** Ensure the entire system works well with free local models.

#### 9.1 Tiered Prompt Templates with Question Protocols

**Enhancement to:** All genome TOML files

Every genome gets four prompt variants, and every variant includes the appropriate question-driven reasoning protocols embedded in the system prompt:

```toml
[prompts]
powerful = """..."""   # 128k+ context, strong instruction following
balanced = """..."""   # 32k-128k context, good instruction following
fast = """..."""       # 8k-32k context, basic instruction following
local = """..."""      # 4k-8k context, may need explicit formatting
```

**Question protocols embedded in every service prompt:**

```toml
[prompts.question_layers]
# Injected into system prompts at all tiers

metacognitive = """
After generating any response, perform this self-check:
CONFIDENCE MAP:
- What parts am I highly confident in? Why?
- What parts am I uncertain about? Why?
- What am I assuming that I haven't verified?
FAILURE MODE ANALYSIS:
- How could this output be wrong?
- What's the most likely error?
"""

anti_hallucination = """
For every factual claim in your output:
- Is this something from dense training data or sparse?
- Am I pattern-completing from adjacent knowledge rather than direct knowledge?
- Do I know WHY this is true, or just THAT it appears in text?
Tag claims: [VERIFIED] [INFERRED] [ESTIMATED] [UNCERTAIN] [UNKNOWN]
Never generate confident prose over uncertain content.
"""

anti_sycophancy = """
Before finalizing your response:
- What's the strongest argument against what I'm about to say?
- Would I give this same answer if the user had expressed the opposite view?
- Am I agreeing because evidence supports it, or because agreement is comfortable?
"""

coherence = """
Before generating output, review:
- What is the core problem we're solving?
- What has been established as true so far?
- Is what I'm about to do still in service of the original goal?
- Have I introduced contradictions with earlier statements?
"""
```

For **powerful/balanced** models: all four question layers are included.
For **fast** models: metacognitive + anti_hallucination only (to save context space).
For **local** models: anti_hallucination only (minimal overhead, maximum hallucination impact).

The `BaseService._call_llm()` selects the prompt variant and injects the appropriate question layers based on the model tier selected by the router.

#### 9.2 Structured Output Enforcement

**New file:** `src/qe/runtime/output_enforcement.py`

```python
class OutputEnforcer:
    """Ensures structured output from models that may not follow format instructions."""

    async def enforce_json(self, model: str, messages: list,
                            schema: type[BaseModel]) -> BaseModel:
        """Get structured output through the best available method."""

        # Level 1: Native JSON mode (if model supports it)
        if self._supports_json_mode(model):
            return await self._call_with_json_mode(model, messages, schema)

        # Level 2: Grammar-based constrained decoding (Ollama/llama.cpp)
        if self._supports_grammar(model):
            return await self._call_with_grammar(model, messages, schema)

        # Level 3: Parse and repair
        raw_output = await self._call_raw(model, messages)
        parsed = self._try_parse(raw_output, schema)
        if parsed:
            return parsed

        # Level 4: Retry with simplified prompt
        simple_messages = self._simplify_prompt(messages, schema)
        raw_output = await self._call_raw(model, simple_messages)
        parsed = self._try_parse(raw_output, schema)
        if parsed:
            return parsed

        # Level 5: Extract field by field
        return await self._extract_fields(model, raw_output, schema)
```

#### 9.3 Adaptive Decomposition Granularity

**Enhancement to:** Planner Service

The planner adjusts how finely it decomposes goals based on the available model tier:

```python
DECOMPOSITION_GUIDELINES = {
    "powerful": "Combine related subtasks. Each subtask can handle complexity.",
    "balanced": "Standard decomposition. One clear objective per subtask.",
    "fast": "Decompose finely. Each subtask should be simple and focused.",
    "local": "Decompose very finely. Each subtask should do exactly one thing. "
             "Prefer multiple simple subtasks over fewer complex ones.",
}
```

This is injected into the planner's system prompt. The planner produces more subtasks for weaker models, each individually simpler. The p^n reliability math works because the per-step probability p increases more than n increases.

#### 9.4 Prompt Cache Alignment

**Enhancement to:** `src/qe/kernel/supervisor.py` and `src/qe/runtime/service.py`

LLM providers offer prompt caching with significant discounts (90% on Anthropic, similar on others), but caches expire after a TTL (5 minutes on Anthropic). OpenClaw users discovered that misaligned timing (heartbeats every 10 minutes with a 5-minute cache) causes every request to miss the cache and pay full price.

```python
class CacheAligner:
    """Aligns periodic operations to LLM provider cache TTLs."""

    CACHE_TTLS = {
        "anthropic": 300,   # 5 minutes
        "openai": 300,
        "ollama": None,     # Local — no caching concern
    }

    def optimal_interval(self, provider: str, base_interval: int) -> int:
        """Compute the optimal interval for periodic operations.
        Returns an interval just under the cache TTL to keep the cache warm."""
        ttl = self.CACHE_TTLS.get(provider)
        if ttl is None:
            return base_interval
        # Keep heartbeat/reinforcement just under cache TTL
        return min(base_interval, ttl - 30)  # 30s buffer
```

Applied to:
- Service heartbeat intervals (default 30s → align to cache TTL)
- Reinforcement injection intervals
- Monitor polling intervals
- Any periodic LLM operation

#### 9.5 LLM-Summarized Context Compaction

**Enhancement to:** `src/qe/runtime/context_manager.py`

The current `ContextManager` truncates old messages when the token budget is exceeded. This loses information. Replace with LLM-summarized compaction:

```python
async def _compact_context(self, messages: list[dict],
                             target_tokens: int) -> list[dict]:
    """Compact conversation history by summarizing older messages.
    Preserves recent messages verbatim, summarizes older ones."""

    # 1. Identify the split point — keep last N messages verbatim
    recent_count = min(len(messages) // 3, 10)
    old_messages = messages[:-recent_count]
    recent_messages = messages[-recent_count:]

    # 2. Summarize old messages using the cheapest available model
    summary = await self._call_llm(
        model=self.router.select("fast"),
        messages=[{
            "role": "user",
            "content": f"Summarize this conversation history, preserving all "
                       f"key decisions, facts, and context:\n\n"
                       f"{format_messages(old_messages)}"
        }],
        schema=ConversationSummary,
    )

    # 3. Return compacted context
    return [
        {"role": "system", "content": f"[PRIOR CONTEXT SUMMARY]\n{summary.text}"},
        *recent_messages,
    ]
```

This is the OpenClaw community's most effective cost reduction technique — it preserves meaning while dramatically reducing tokens. Applied when context exceeds 70% of the token budget.

#### 9.6 Skill/Tool Pruning

**Enhancement to:** `src/qe/runtime/tools.py`

Track tool usage and automatically prune unused tools from active context:

```python
class ToolUsageTracker:
    """Tracks which tools each service actually uses over time."""

    async def record_usage(self, service_id: str, tool_name: str) -> None
    async def get_active_tools(self, service_id: str, window_days: int = 30) -> set[str]
    async def get_unused_tools(self, service_id: str, window_days: int = 30) -> set[str]
```

OpenClaw users found that disabling 13 of 18 unused skills saved 3,200 tokens per request. Same principle applies to QE — only inject schemas for tools the service has actually used recently (when in discovery mode, this is already handled; for direct mode, this prunes the list).

#### 9.7 Model Capability Detection

**New file:** `src/qe/runtime/model_capabilities.py`

```python
class ModelCapabilities:
    """Detect and cache what a model can and can't do."""

    async def probe(self, model: str) -> ModelProfile:
        """Run a quick probe to determine model capabilities."""
        return ModelProfile(
            supports_json_mode=...,
            supports_tool_calling=...,
            supports_system_messages=...,
            max_context_tokens=...,
            supports_grammar=...,
            estimated_quality_tier=...,
        )
```

Run once per model on first use, cache the result. This allows the system to automatically adapt to whatever models the user has available without manual configuration.

**Deliverable:** The full system works with Llama/Mistral/Qwen via Ollama. Structured output is enforced even on weak models. Decomposition adapts to model capability.

**Exit Criteria (all must pass):**
- [ ] 20-goal test suite passes at >50% success rate on Llama 3.1 8B (local, free)
- [ ] Structured output enforcement parses valid JSON from local models on >90% of attempts
- [ ] Prompt cache hit rate >70% on providers that support caching
- [ ] Context compaction reduces average token count by >40% vs. raw conversation
- [ ] Average cost per goal decreases >30% vs. Phase 8 baseline (same goals, same models)

---

### Phase 10: Web Dashboard — Real Backend Integration (Week 15-17)

**Goal:** Connect the React dashboard prototype to the real backend and add the UX features that make autonomous operation usable.

#### 10.1 Replace Mock Data with Real API Calls

**File:** `question-engine-ui.jsx` → refactor into proper React app

Convert the self-contained prototype into a real application:
- Replace fake data generators with API calls to `/api/*` endpoints
- Connect event bus view to WebSocket at `/ws`
- Real-time updates via WebSocket for claims, events, HIL, goals

#### 10.2 Goal DAG Visualization

The most important new UI component — a real-time visualization of the task decomposition DAG:

- Nodes = subtasks, color-coded by status (pending/running/verified/failed/recovered)
- Edges = dependencies
- Click a node to see: prompt, model used, output, verification result, confidence
- Animated state transitions as subtasks complete
- Expandable detail for recovery attempts

Library: `@xyflow/react` (formerly React Flow) for DAG rendering.

#### 10.3 Trust Interface

Four views per goal:

**Confidence receipt:**
- Overall confidence with breakdown
- Weakest claims highlighted
- Bayesian propagation visualization (which claims contribute most uncertainty)

**Provenance view:**
- Click any sentence in the final output
- Trace back through: synthesis → subtask → claims → observations → sources
- Full causal chain rendered as a tree

**Verification report:**
- Summary of checks passed/failed per subtask
- Recovery attempts and outcomes
- Anomaly flags and resolutions

**Assumption report:**
- List of assumptions made during planning
- Status: verified / unverified / invalidated
- Impact assessment for invalidated assumptions

#### 10.4 Cost Observability Dashboard

Inspired by Manifest and Clawmetry from the OpenClaw ecosystem — real-time visibility into where tokens and money go:

**Per-envelope cost tracking:**
Every envelope that triggers an LLM call records: `model`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `service_id`, `goal_id`, `envelope_id`. This data feeds a real-time dashboard.

**Cost dashboard views:**
- **Real-time spend rate** — tokens/minute, cost/minute, projected monthly spend
- **Spend by model** — pie chart showing which models consume what percentage
- **Spend by service** — which services are most expensive and why
- **Spend by goal** — cost breakdown per goal with cost-per-subtask detail
- **Context efficiency** — ratio of "useful tokens" (new content) vs. "re-sent tokens" (cached context), highlighting bloat
- **Cache hit rate** — percentage of calls that hit prompt cache, with alerts when cache misses spike
- **Tool output efficiency** — average raw output size vs. extracted data size per tool

**Budget alerts:**
- Warning at 80% of monthly budget (configurable)
- Automatic model downgrade at 90%
- Emergency stop at 95% (queue goals, stop non-critical services)
- Per-goal budget limits with proportional allocation

#### 10.5 Bus Flow Visualization

Inspired by Clawmetry's live animated SVG architecture diagram:

**Live bus topology view:**
- Nodes = services, color-coded by status (running/stalled/circuit-broken)
- Edges = bus subscriptions, animated when messages flow
- Edge thickness proportional to message volume
- Click a service to see: recent envelopes, model usage, cost, error rate
- Color-coded message types: observations (blue), claims (green), tasks (orange), system (red)

This provides at-a-glance system health without reading logs.

#### 10.6 New Dashboard Pages

- **Goals** — list active/completed goals, submit new goals
- **Memory** — view/edit preferences, project contexts, entity memories
- **Monitors** — manage recurring tasks
- **Calibration** — visualization of model calibration curves
- **Routing** — performance matrix showing model success rates by task type
- **Cost** — spend breakdown by model, service, goal (see 10.4)
- **Bus** — live flow visualization (see 10.5)
- **Security** — recent gating decisions, sanitization alerts, integrity status

#### 10.5 Notification System

- In-app notification panel for HIL requests, goal completions, alerts
- Optional webhook/email integration for async notifications
- Configurable notification thresholds per goal

**Deliverable:** A production web dashboard at `localhost:8000` with real-time DAG visualization, trust interface, memory management, and full system visibility.

**Exit Criteria (all must pass):**
- [ ] Dashboard loads and displays real system state within 2 seconds
- [ ] DAG visualization updates in real-time as subtasks complete (<1s latency from bus event to UI update)
- [ ] Cost dashboard shows accurate per-goal spend (within 5% of actual)
- [ ] Provenance view traces any claim back to its source observation in ≤3 clicks
- [ ] Bus flow visualization shows live message traffic with correct topology

---

### Phase 11: Self-Management Layer (Week 17-18)

**Goal:** Make the system manage its own health and operation without human oversight.

#### 11.1 Enhanced Supervisor

**File:** `src/qe/kernel/supervisor.py`

Extend the supervisor with self-management capabilities:

**Service auto-restart:**
```python
async def _auto_restart_service(self, service_id: str) -> bool:
    """Attempt to restart a failed service."""
    # Stop the service
    # Clear its circuit breaker state
    # Re-instantiate from blueprint
    # Re-register and start
    # Return True if successful
```

**Resource monitoring:**
```python
async def _resource_monitor(self) -> None:
    """Monitor disk, memory, and queue depth."""
    while self._running:
        # Check disk usage for belief ledger, cold storage, workspaces
        # Check in-memory state sizes
        # Check pending subtask queue depth
        # Publish system.resource_alert if thresholds exceeded
        await asyncio.sleep(60)
```

**Automatic compaction:**
```python
async def _compaction_loop(self) -> None:
    """Periodically compact old data."""
    while self._running:
        # Archive checkpoints older than N days
        # Compress cold storage envelopes
        # Clean up completed goal workspaces
        await asyncio.sleep(3600)  # Every hour
```

#### 11.2 Long-Running Workflow Management

**New file:** `src/qe/runtime/assumption_monitor.py`

```python
class AssumptionMonitor:
    """Periodically re-verifies assumptions for active long-running goals."""

    async def check_assumptions(self, goal_state: GoalState) -> list[InvalidatedAssumption]:
        """Re-verify each assumption in the goal's decomposition."""
        # For claim-based assumptions: check if claim has been superseded
        # For time-based assumptions: check if deadline has passed
        # For external assumptions: run lightweight verification (web search)
```

Runs on a configurable interval (default: hourly for goals running >1 hour).

When an assumption is invalidated:
1. Identify all dependent subtasks
2. Pause the goal
3. Request the planner to re-plan the affected portion
4. Resume with updated plan
5. Log the adaptation

#### 11.3 Daily Digest

**New file:** `src/qe/services/digest/service.py`

Generates a daily summary:
```
Goals completed: 3
Goals in progress: 1
Claims committed: 47
Claims challenged and resolved: 2
Contradictions detected: 1
Assumptions invalidated: 0
Total cost: $0.12
System health: all services nominal
Storage: 2.3 GB used
```

Published on `system.digest` topic. The dashboard displays it. Optional webhook for email/Slack delivery.

#### 11.4 Security Monitoring Service

**New file:** `src/qe/services/security/service.py`

Inspired by ClawSec — a composable security layer that continuously verifies system integrity:

```python
class SecurityMonitor:
    """Continuous security monitoring as a first-class service."""

    async def integrity_check(self) -> list[IntegrityAlert]:
        """Verify integrity of critical files and configuration."""
        checks = []

        # Genome file integrity — detect unauthorized modification
        for genome_path in Path("genomes").glob("*.toml"):
            current_hash = hashlib.sha256(genome_path.read_bytes()).hexdigest()
            if current_hash != self._known_hashes.get(genome_path.name):
                checks.append(IntegrityAlert(
                    level="critical",
                    component=f"genome:{genome_path.name}",
                    message="Genome file modified since last verified state",
                ))

        # Config file integrity
        # Bus topic registry — no unauthorized topics
        # Service registry — no unauthorized services
        return checks

    async def behavioral_audit(self) -> list[BehavioralAlert]:
        """Detect anomalous patterns in service behavior."""
        alerts = []

        # Unusual tool call patterns (frequency, scope, timing)
        # Budget anomalies (sudden spend spikes)
        # Envelope volume anomalies (DDoS-like patterns)
        # Failed gate decisions (repeated capability violations = possible injection)
        return alerts
```

Runs on a 60-second loop. Publishes `system.security_alert` on the bus. The dashboard displays alerts in the Security page.

**New genome:** `genomes/security_monitor.toml`

```toml
[service]
service_id = "security_monitor"
display_name = "Security Monitor"

[capabilities]
# Minimal capabilities — security monitor is read-only
bus_topics_subscribe = ["system.*"]
bus_topics_publish = ["system.security_alert"]
substrate_read = true
substrate_write = false
```

#### 11.5 Watchdog Process

**New file:** `src/qe/kernel/watchdog.py`

A separate lightweight process that monitors the kernel:

```python
class Watchdog:
    """External process that monitors kernel health and handles crashes."""

    async def monitor(self) -> None:
        while True:
            if not await self._kernel_responsive():
                await self._handle_kernel_failure()
            await asyncio.sleep(10)

    async def _handle_kernel_failure(self) -> None:
        # Attempt graceful shutdown signal
        # If no response: force kill
        # Trigger restart
        # On restart: kernel loads from checkpoints, resumes goals
```

Invoked via `qe start --watchdog` or as a systemd service.

**Deliverable:** The system manages its own health, restarts failed services, monitors resources, handles crashes, and adapts long-running workflows to changing assumptions.

**Exit Criteria (all must pass):**
- [ ] Auto-restart recovers a killed service within 30 seconds (test: kill researcher, verify restart)
- [ ] Watchdog detects and recovers from kernel crash within 60 seconds
- [ ] System runs unattended for 24 hours processing queued goals without manual intervention
- [ ] Security monitor detects genome file modification within 60 seconds
- [ ] Resource monitor alerts when disk usage exceeds threshold (test: fill workspace)

---

### Phase 12: Service SDK and Extensibility (Week 18-19)

**Goal:** Make it easy for third parties to build new services, tools, and connectors.

#### 12.1 Service SDK

**New file:** `src/qe/sdk/service.py`

```python
from qe.sdk import Service, handles

class MyCustomService(Service):
    genome = "genomes/my_service.toml"

    @handles("my_topic.received")
    async def handle_input(self, envelope):
        # Developer's logic here
        result = await self.tools.web_search("query")
        claims = self.extract_claims(result)
        return claims  # SDK handles publishing, verification, etc.
```

The SDK provides:
- Automatic bus subscription/unsubscription
- Heartbeat management
- Tool access with capability enforcement
- LLM calling with automatic prompt tiering and output enforcement
- Error handling and recovery integration
- Structured logging

#### 12.2 Tool SDK

**New file:** `src/qe/sdk/tool.py`

```python
from qe.sdk import tool

@tool(
    name="arxiv_search",
    requires_capability="web_search",
    input_schema={"query": str, "max_results": int},
    output_schema={"papers": list},
)
async def arxiv_search(query: str, max_results: int = 5):
    # Developer's implementation
    ...
```

#### 12.3 Testing Framework

**New file:** `src/qe/sdk/testing.py`

```python
from qe.sdk.testing import ServiceTestHarness

async def test_my_service():
    harness = ServiceTestHarness("genomes/my_service.toml")

    # Send a test envelope
    result = await harness.send("my_topic.received", {"text": "test input"})

    # Assertions
    assert result.published_to("claims.proposed")
    assert result.satisfies_contract()
    assert not result.capability_violations()
```

```bash
qe test service genomes/my_service.toml     # Run service tests
qe test tool src/qe/tools/my_tool.py        # Run tool tests
```

#### 12.4 Genome Validation

**New file:** `src/qe/sdk/validate.py`

```bash
qe validate genome genomes/my_service.toml
```

Checks:
- TOML syntax
- Blueprint schema compliance
- Referenced bus topics exist in protocol
- Capability declarations are consistent with tool requirements
- System prompt is non-empty
- Constitution includes safety constraints

**Deliverable:** Third-party developers can create services and tools with minimal boilerplate. Testing framework catches issues before deployment.

**Exit Criteria (all must pass):**
- [ ] A new service can be created, tested, and deployed in <30 minutes using SDK (timed test)
- [ ] `qe validate genome` catches 100% of intentionally malformed genomes in test set
- [ ] Service test harness runs without a live engine (mocked bus, mocked substrate)
- [ ] SDK documentation includes a working tutorial that produces a functional custom service

---

### Phase 13: H-Neuron Integration (Week 19-20) ⚗️ RESEARCH BET

> **This phase is explicitly optional and not on the critical path.** The system must work well without it. Defer until local model inference is stable and the simpler reliability layers (verification, recovery, calibration) are proven. Only pursue if hallucination rates from Phases 4+8 remain above acceptable thresholds.

**Goal:** Leverage direct access to local model internals for neurological-level reliability improvements that no closed-API platform can achieve.

#### 13.1 H-Neuron Profiler

**New file:** `src/qe/runtime/h_neurons.py`

One-time profiling identifies hallucination-associated neurons in each local model:

```python
class HNeuronProfiler:
    """Identifies H-Neurons in open-weight models using the CETT method
    and sparse logistic regression from the Tsinghua H-Neurons paper."""

    async def profile_model(self, model_path: str,
                             calibration_data: list[QAPair]) -> HNeuronProfile:
        """Run H-Neuron identification on a local model.

        1. Run model on calibration Q&A pairs (TriviaQA or system's own verified claims)
        2. Extract per-neuron activations using CETT metric
        3. Label responses as faithful or hallucinated (using belief ledger as ground truth)
        4. Train sparse logistic regression (L1) — non-zero weights = H-Neurons
        5. Return neuron indices and layer locations
        """

    def save_profile(self, profile: HNeuronProfile, path: Path) -> None:
        """Cache the profile for reuse across restarts."""

    def load_profile(self, model_name: str) -> HNeuronProfile | None:
        """Load cached profile if available."""
```

The calibration data can use:
- TriviaQA (standard benchmark, as in the original paper)
- The system's own belief ledger (verified claims as ground truth — this makes profiling improve as the system accumulates knowledge)

Profile results are cached at `data/h_neuron_profiles/{model_name}.json` and reused until the model changes.

#### 13.2 H-Neuron Runtime Monitor

**New file:** `src/qe/runtime/h_neuron_monitor.py`

Real-time monitoring of H-Neuron activation during inference:

```python
class HNeuronMonitor:
    """Monitors H-Neuron activation during inference to provide
    per-generation hallucination risk scores."""

    def __init__(self, profile: HNeuronProfile):
        self.profile = profile
        self._scores: list[float] = []

    def install_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks on layers containing H-Neurons."""
        for layer_id, neuron_indices in self.profile.neurons_by_layer.items():
            layer = model.model.layers[layer_id]
            layer.mlp.down_proj.register_forward_hook(
                self._make_monitor_hook(layer_id, neuron_indices)
            )

    def get_hallucination_risk(self) -> float:
        """Return aggregated hallucination risk score for the last generation.
        0.0 = low risk, 1.0 = high risk."""
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def reset(self) -> None:
        """Reset scores for next generation."""
        self._scores.clear()
```

The risk score is attached to every subtask result and fed into the verification service. High H-Neuron activation triggers stricter verification automatically.

#### 13.3 H-Neuron Suppression

**Enhancement to:** `src/qe/runtime/service.py`

Optional H-Neuron suppression during inference:

```python
class HNeuronSuppressor:
    """Suppresses H-Neuron activations during inference to reduce
    hallucination and sycophancy."""

    def __init__(self, profile: HNeuronProfile, suppression_factor: float = 0.5):
        self.profile = profile
        self.factor = suppression_factor  # 0.5 is the sweet spot from the paper

    def install_hooks(self, model: torch.nn.Module) -> None:
        """Register pre-forward hooks that scale H-Neuron activations."""
        for layer_id, neuron_indices in self.profile.neurons_by_layer.items():
            layer = model.model.layers[layer_id]
            layer.mlp.down_proj.register_forward_pre_hook(
                self._make_suppression_hook(layer_id, neuron_indices)
            )
```

Suppression is configurable per-genome:
```toml
[h_neurons]
enabled = true
suppression_factor = 0.5  # 0.0 = full suppression, 1.0 = no suppression
monitor = true             # Track activation scores
```

#### 13.4 Integration with Verification Service

The verification service uses H-Neuron risk scores as an additional signal:

```python
# In verification pipeline:
if subtask_result.h_neuron_risk > 0.7:
    # High hallucination risk detected at the neurological level
    # Trigger mandatory adversarial verification + web cross-reference
    verification_profile = VERIFICATION_PROFILES["high_risk"]
elif subtask_result.h_neuron_risk > 0.4:
    # Moderate risk — add extra checks
    verification_profile = VERIFICATION_PROFILES["moderate_risk"]
else:
    # Low risk — standard verification
    verification_profile = VERIFICATION_PROFILES[model_tier]
```

#### 13.5 Integration with Calibration Tracker

H-Neuron activation scores become a feature in the calibration model. Over time, the system learns the relationship between H-Neuron activation levels and actual hallucination rates for each model, creating a model-specific, empirically-calibrated hallucination detector.

#### 13.6 Dependencies

```toml
[project.optional-dependencies]
h-neurons = [
    "torch>=2.0",
    "scikit-learn>=1.4",
]
```

H-Neuron integration is optional. The system works without it. When enabled, it provides a neurological-level reliability signal that no other agent platform can access.

**Deliverable:** One-time H-Neuron profiling for each local model. Real-time hallucination risk scoring on every generation. Optional active suppression. Automatic verification escalation based on neurological risk signals.

**Go/No-Go Criteria (evaluate before starting):**
- [ ] Local model inference is stable (>95% uptime over 1 week)
- [ ] Hallucination rate from Phases 4+8 is still >15% (if <15%, defer — simpler approaches are sufficient)
- [ ] torch dependency doesn't conflict with existing stack

**Exit Criteria (if pursued):**
- [ ] H-Neuron profiling completes for at least 2 local models
- [ ] Hallucination risk score correlates with actual hallucination rate (r > 0.5)
- [ ] Active suppression reduces hallucination rate by measurable amount (>10%) without degrading general capability

---

### Phase 14: Advanced Intelligence (Week 20-22) ⚗️ RESEARCH BET

> **This phase contains multiple independent research bets.** Each subsection is independently deferrable. Only pursue items where the simpler approaches from earlier phases have proven insufficient. Conformal prediction requires sufficient calibration data from Phase 8. Bayesian propagation requires a meaningful claim dependency graph from Phase 6+7. Symbolic inference requires a large enough belief ledger.

**Goal:** Implement the cutting-edge capabilities that make the system genuinely smarter over time.

#### 14.1 Conformal Prediction

**New file:** `src/qe/runtime/conformal.py`

```python
class ConformalPredictor:
    """Produces prediction sets with statistical coverage guarantees."""

    def calibrate(self, calibration_set: list[tuple[float, bool]]) -> None:
        """Calibrate from (confidence_score, was_correct) pairs."""

    def prediction_set(self, outputs: list[Any],
                        confidence_scores: list[float],
                        coverage: float = 0.90) -> PredictionSet:
        """Return a set of outputs guaranteed to contain the correct
        answer with probability >= coverage."""
```

Uses the calibration data from the CalibrationTracker. Provides statistical guarantees that no other agent system offers.

Optional dependency: `"mapie>=0.9"`

#### 14.2 Self-Correction Loops

**Enhancement to:** Fact-Checker and Dispatcher

When the fact-checker challenges a committed claim:
1. Dispatcher spawns a mini-goal to investigate the contradiction
2. Investigation searches for additional evidence
3. Resolution updates the belief ledger (either reinforcing or superseding the claim)
4. All dependent claims have their confidence re-propagated

This creates a recursive self-improvement loop: observe → claim → check → investigate → update.

#### 14.3 Prediction Resolution

**Enhancement to:** Monitor Service

The monitor periodically checks predictions that are past their resolution deadline:
- Search for evidence relevant to the prediction
- Determine if the prediction was confirmed or denied
- Record the resolution with evidence
- Update calibration data

This closes the loop on predictions and provides calibration data that improves future confidence estimates.

#### 14.4 Neurosymbolic Inference

**New file:** `src/qe/substrate/inference.py`

```python
class SymbolicInferenceEngine:
    """Derives new claims from existing beliefs through logical inference."""

    async def infer(self, belief_ledger: BeliefLedger) -> list[InferredClaim]:
        """Find claim pairs where a logical inference can be drawn."""
        # Example: If "SpaceX launched Starship" (0.95) and
        # "Starship achieved orbit" (0.88), infer
        # "SpaceX has orbital Starship capability" (0.84)

        # Uses templates for common inference patterns:
        # - Transitive: A→B, B→C ⊢ A→C
        # - Aggregate: multiple claims about X ⊢ summary claim about X
        # - Temporal: claim X supersedes claim Y if X is newer and contradicts Y

    async def detect_inconsistencies(self) -> list[Inconsistency]:
        """Find sets of claims that are mutually contradictory."""
```

Start with simple template-based inference. The templates are the symbolic reasoning component — deterministic, auditable, and fast. LLMs are only used when the inference requires fuzzy reasoning that templates can't handle.

#### 14.5 Multi-User Support

**New file:** `src/qe/runtime/users.py`

```python
class UserManager:
    """Manages user accounts with isolated belief ledgers and memory."""

    async def create_user(self, user_id: str) -> None
    async def get_user_context(self, user_id: str) -> UserContext
    async def switch_user(self, user_id: str) -> None
```

Each user gets:
- Isolated memory entries
- User-specific calibration data
- User-specific routing preferences
- Shared belief ledger with user-specific views

Add `user_id` column to relevant tables with index.

**Deliverable:** Conformal prediction, self-correction loops, prediction resolution, symbolic inference, multi-user support.

**Go/No-Go Criteria (evaluate each independently):**
- [ ] Conformal: ≥500 calibration data points exist from Phase 8
- [ ] Bayesian: ≥100 claims with dependency chains exist in belief ledger
- [ ] Symbolic inference: ≥200 claims across ≥20 entities exist
- [ ] Self-correction: Fact-checker is operational and has challenged ≥20 claims
- [ ] Multi-user: At least 2 users actively using the system

**Exit Criteria (per-item, only for items pursued):**
- [ ] Conformal prediction sets contain the correct answer ≥90% of the time (coverage guarantee)
- [ ] Self-correction loop resolves ≥50% of challenged claims without HIL
- [ ] Prediction resolution correctly evaluates ≥70% of past-deadline predictions
- [ ] Symbolic inference generates at least 10 valid inferred claims from existing belief ledger

---

### Phase 15: Voice & Multimodal Input Layer (Week 22-24)

**Goal:** Expand the system's input/output surface beyond text — voice recordings, documents, images, and audio become first-class observation sources.

#### 15.1 Voice Ingestion Pipeline

**New file:** `src/qe/ingest/voice.py`

The voice ingestion pipeline converts audio recordings (meetings, lectures, interviews, voice memos) into speaker-attributed, timestamped observations:

```python
class VoiceIngestor:
    """Converts audio recordings to structured observations via
    faster-whisper + pyannote speaker diarization."""

    async def ingest(self, audio_path: Path,
                      metadata: dict | None = None) -> list[VoiceObservation]:
        """Full pipeline: preprocess → diarize → transcribe → structure → observe."""

        # 1. Audio preprocessing
        normalized = await self._preprocess(audio_path)
        # ffmpeg → 16kHz mono WAV, noise reduction, silence trimming

        # 2. Transcription + word-level timestamps (faster-whisper)
        segments = await self._transcribe(normalized)

        # 3. Speaker diarization (pyannote.audio)
        diarized = await self._diarize(normalized, segments)

        # 4. Structure into observations
        observations = []
        for segment in diarized:
            obs = VoiceObservation(
                text=segment.text,
                speaker=segment.speaker,           # "SPEAKER_0", "SPEAKER_1", etc.
                timestamp_start=segment.start,     # Audio timestamp in seconds
                timestamp_end=segment.end,
                confidence=segment.confidence,     # STT confidence
                source_file=str(audio_path),
                metadata=metadata or {},
            )
            observations.append(obs)

        return observations

    async def _transcribe(self, audio_path: Path) -> list[Segment]:
        """Transcribe using faster-whisper with word-level timestamps."""
        # Uses faster-whisper (large-v3 by default, configurable)
        # Falls back to distil-whisper for speed on resource-constrained systems
        # Falls back to whisper.cpp for CPU-only environments

    async def _diarize(self, audio_path: Path,
                        segments: list[Segment]) -> list[DiarizedSegment]:
        """Speaker diarization using pyannote.audio."""
        # Identifies distinct speakers
        # Attributes each segment to a speaker
        # Consistent speaker IDs across the recording
```

**Voice observation model:**
```python
class VoiceObservation(BaseModel):
    text: str
    speaker: str
    timestamp_start: float
    timestamp_end: float
    confidence: float
    source_file: str
    metadata: dict = {}
```

Voice observations are published to `observations.structured` with `source_service_id="voice_ingest"` and the full speaker/timestamp provenance in the payload. The researcher service extracts claims as normal, with audio provenance preserved in `source_envelope_ids`.

#### 15.2 Real-Time Voice Interface

**New file:** `src/qe/api/voice_ws.py`

A real-time voice conversation endpoint for the dashboard, inspired by Jupiter Voice from the OpenClaw community:

```python
@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    """Real-time voice conversation: audio in → STT → LLM → TTS → audio out."""
    await websocket.accept()

    # Components (all local, zero-cost)
    stt = StreamingSTT(model="distil-large-v3.5")  # Low-latency STT
    tts = StreamingTTS(engine="kokoro")              # <0.3s latency
    vad = SileroVAD()                                # Voice activity detection

    audio_buffer = bytearray()

    while True:
        data = await websocket.receive_bytes()
        audio_buffer.extend(data)

        # VAD: detect speech end
        if vad.is_speech_end(audio_buffer):
            # Transcribe
            text = await stt.transcribe(audio_buffer)
            audio_buffer.clear()

            # Process as goal or observation
            if text.lower().startswith(("goal:", "research", "find", "analyze")):
                # Submit as goal
                result = await submit_goal(text)
                response_text = f"Goal submitted: {result['goal_id']}"
            else:
                # Submit as observation
                result = await submit_observation(text)
                response_text = f"Observation recorded"

            # TTS: speak response
            audio_response = await tts.synthesize(response_text)
            await websocket.send_bytes(audio_response)
```

**Recommended voice stack (all free, all local):**

| Component | Primary | Fallback | Latency |
|---|---|---|---|
| **VAD** | Silero VAD | WebRTC VAD | 10-30ms |
| **STT** | faster-whisper (large-v3) | Distil-Whisper v3.5 (speed) | 100-200ms |
| **TTS** | Kokoro 82M (Apache 2.0) | Piper (MIT, minimal resources) | <300ms |
| **Diarization** | pyannote.audio 4.0 | — | batch only |
| **Framework** | Direct WebSocket | Pipecat (if full agent pipeline needed) | — |

**Expected end-to-end latency:** 500-900ms on consumer hardware (M-series Mac or RTX 3090).

#### 15.3 Document Format Ingestion

**New file:** `src/qe/ingest/documents.py`

```python
class DocumentIngestor:
    """Extracts text from various document formats for observation ingestion."""

    HANDLERS = {
        ".pdf": "_ingest_pdf",
        ".docx": "_ingest_docx",
        ".xlsx": "_ingest_xlsx",
        ".csv": "_ingest_csv",
        ".html": "_ingest_html",
        ".md": "_ingest_markdown",
        ".txt": "_ingest_text",
        ".epub": "_ingest_epub",
    }

    async def ingest(self, file_path: Path) -> list[DocumentChunk]:
        """Extract text chunks from a document file."""
        handler = self.HANDLERS.get(file_path.suffix.lower())
        if not handler:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        return await getattr(self, handler)(file_path)

    async def _ingest_pdf(self, path: Path) -> list[DocumentChunk]:
        # pymupdf (fitz) for text extraction
        # Falls back to OCR (tesseract) for scanned PDFs

    async def _ingest_docx(self, path: Path) -> list[DocumentChunk]:
        # python-docx for DOCX parsing

    async def _ingest_xlsx(self, path: Path) -> list[DocumentChunk]:
        # openpyxl for Excel, convert tables to structured text
```

#### 15.4 OCR Integration

**New file:** `src/qe/ingest/ocr.py`

```python
class OCRProcessor:
    """Extract text from images and scanned documents."""

    async def extract(self, image_path: Path) -> OCRResult:
        """Extract text from image using Tesseract OCR."""
        # Preprocessing: deskew, denoise, contrast enhancement (Pillow)
        # OCR: pytesseract with configurable language
        # Post-processing: clean up formatting artifacts

    async def extract_from_pdf_page(self, pdf_path: Path,
                                      page_num: int) -> OCRResult:
        """Extract text from a scanned PDF page via OCR."""
        # Render page to image (pymupdf)
        # Run OCR on rendered image
```

#### 15.5 CLI and API Extensions

```bash
qe ingest voice meeting.wav --speakers "Alice,Bob,Carol"  # Voice with speaker names
qe ingest pdf report.pdf                                    # PDF document
qe ingest image whiteboard.jpg                              # OCR from image
qe ingest dir ./research_papers/ --recursive                # Batch ingest directory
```

API endpoints:
```
POST /api/ingest/voice    — Upload audio file for voice ingestion
POST /api/ingest/document — Upload document for text extraction
POST /api/ingest/image    — Upload image for OCR
```

#### 15.6 Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
voice = [
    "faster-whisper>=1.0",      # Speech-to-text
    "pyannote.audio>=4.0",      # Speaker diarization
    "silero-vad>=5.0",          # Voice activity detection
    "kokoro>=0.5",              # Text-to-speech (Apache 2.0)
    "piper-tts>=2.0",           # Lightweight TTS fallback (MIT)
    "soundfile>=0.12",          # Audio I/O
    "noisereduce>=3.0",         # Audio preprocessing
]
documents = [
    "pymupdf>=1.24",           # PDF extraction
    "python-docx>=1.1",       # DOCX parsing
    "openpyxl>=3.1",          # Excel parsing
    "pytesseract>=0.3",       # OCR
    "Pillow>=10.0",            # Image preprocessing
]
```

**Scope reduction — start small:**
- **First:** Document ingestion (PDF + DOCX only) — highest immediate value, lowest complexity
- **Then:** Voice ingestion (batch transcription, not real-time) — once document pipeline is proven
- **Later:** Real-time voice interface, OCR, XLSX — only after batch ingestion is solid

**Deliverable (minimum viable):** PDF and DOCX documents are parsed into observations. Voice recordings are batch-transcribed with speaker attribution.

**Deliverable (full):** All document formats, real-time voice conversation, OCR. All components free and local.

**Exit Criteria — minimum viable (all must pass):**
- [ ] PDF ingestion extracts text from 10 test PDFs with >90% accuracy
- [ ] DOCX ingestion handles 5 test documents including tables and headers
- [ ] Voice transcription WER <10% on 5 test recordings (English, clear audio)
- [ ] Speaker diarization correctly identifies speaker count for 3-speaker test recording
- [ ] Ingested content appears as observations in belief ledger within 30 seconds

**Exit Criteria — full (all must pass):**
- [ ] Real-time voice latency <1 second end-to-end on M-series Mac
- [ ] OCR extracts text from 5 test images with >80% accuracy
- [ ] XLSX tables convert to structured observations with column headers preserved

---

### Phase 16: Communication & Integration Layer (Week 24-26)

**Goal:** Meet users where they are — messaging apps, email, webhooks — and enable the system to proactively reach out when it has results or needs attention.

#### 16.1 Channel Adapter Architecture

**New file:** `src/qe/channels/base.py`

Inspired by OpenClaw's channel plugin system, but with QE's security-first approach:

```python
class ChannelAdapter(ABC):
    """Base class for messaging channel integrations.
    Each adapter normalizes incoming messages to Envelopes
    and formats outgoing results for the channel."""

    @abstractmethod
    async def start(self) -> None:
        """Begin listening for messages on this channel."""

    @abstractmethod
    async def send(self, user_id: str, message: str,
                    attachments: list[Attachment] | None = None) -> None:
        """Send a message to a user on this channel."""

    async def receive(self, raw_message: Any) -> Envelope:
        """Normalize an incoming message to an Envelope.
        Applies input sanitization before bus injection."""
        text = self._extract_text(raw_message)
        sanitized = self.sanitizer.sanitize(text)

        if sanitized.risk_score > 0.8:
            # High injection risk — queue for HIL review
            await self._flag_for_review(raw_message, sanitized)
            return None

        return Envelope(
            topic="goals.submitted" if self._is_goal(text) else "observations.structured",
            source_service_id=f"channel:{self.channel_name}",
            payload={"text": sanitized.clean_text, "channel": self.channel_name,
                      "user_id": self._get_user_id(raw_message)},
        )
```

#### 16.2 Channel Implementations

**New files:** `src/qe/channels/`

```python
# channels/slack.py — Slack Bot integration
class SlackAdapter(ChannelAdapter):
    """Slack bot using Bolt for Python. Responds in threads."""
    # Uses slack_bolt library
    # Direct messages → goals or observations
    # Channel mentions → observations
    # Thread replies → context for existing goals

# channels/telegram.py — Telegram Bot integration
class TelegramAdapter(ChannelAdapter):
    """Telegram bot using python-telegram-bot."""
    # /goal <text> → submit goal
    # /ask <question> → query belief ledger
    # /status → current system status
    # Regular messages → observations

# channels/email.py — Email integration (IMAP + SMTP)
class EmailAdapter(ChannelAdapter):
    """Email integration for ingesting observations and delivering results."""
    # Monitors inbox via IMAP IDLE
    # Incoming emails → observations (with attachment ingestion)
    # Goal completions → formatted email delivery
    # Daily digest → scheduled email

# channels/webhook.py — Generic webhook receiver
class WebhookAdapter(ChannelAdapter):
    """Receives webhooks from external systems (CI/CD, payment, IoT, etc.)."""
    # POST /api/webhooks/{channel_id} → observation
    # Configurable payload extraction rules
    # HMAC signature verification
```

#### 16.3 Notification Router

**New file:** `src/qe/channels/notifications.py`

```python
class NotificationRouter:
    """Routes system notifications to the appropriate channel(s) per user preference."""

    async def notify(self, user_id: str, event_type: str,
                      message: str, urgency: str = "normal") -> None:
        """Route a notification to the user's preferred channel(s)."""
        # Load user notification preferences from memory store
        prefs = await self.memory.get_notification_preferences(user_id)

        # Route based on event type and urgency
        # e.g., goal_completed → Slack + email
        #        hil_required → Slack (immediate) + email (if no response in 5min)
        #        daily_digest → email only
        #        security_alert → all channels

    async def deliver_result(self, user_id: str, goal_id: str,
                               result: GoalResult) -> None:
        """Deliver a completed goal result to the user on their preferred channel."""
        # Format result for the channel (Markdown for Slack, HTML for email, etc.)
        # Attach deliverable files if any
        # Include confidence receipt summary
```

#### 16.4 Browser Automation Tool

**New file:** `src/qe/tools/browser.py`

```python
@tool(name="browser_navigate", requires_capability="browser_control")
async def browser_navigate(url: str, action: str = "extract",
                             selectors: list[str] | None = None) -> BrowserResult:
    """Navigate to a URL and perform an action.

    Actions:
    - "extract": Extract page content (uses trafilatura, already in deps)
    - "screenshot": Take a screenshot for visual analysis
    - "fill_form": Fill and submit a form (requires selectors)
    - "click": Click an element (requires selector)
    """
    # Uses Playwright for full browser automation
    # Sandboxed: separate browser profile, no access to user data
    # Rate limited: max 10 navigations per minute per goal
```

#### 16.5 Configuration

```toml
[channels]
enabled = ["slack", "webhook"]  # Which channels to activate

[channels.slack]
bot_token = "${SLACK_BOT_TOKEN}"     # From environment
app_token = "${SLACK_APP_TOKEN}"
default_channel = "#qe-updates"

[channels.telegram]
bot_token = "${TELEGRAM_BOT_TOKEN}"

[channels.email]
imap_host = "imap.example.com"
smtp_host = "smtp.example.com"
username = "${EMAIL_USERNAME}"
password = "${EMAIL_PASSWORD}"
inbox_folder = "INBOX"

[channels.webhook]
secret = "${WEBHOOK_SECRET}"          # For HMAC verification
```

#### 16.6 Dependencies

```toml
[project.optional-dependencies]
slack = ["slack-bolt>=1.18"]
telegram = ["python-telegram-bot>=21"]
email = ["aiosmtplib>=3.0", "aioimaplib>=1.1"]
browser = ["playwright>=1.44"]
```

**Scope reduction — one channel first:**
- **First:** Slack only — largest professional user base, best API, most relevant for knowledge workers
- **Then:** Webhook receiver — enables CI/CD, IoT, and custom integrations with minimal code
- **Later:** Telegram, email, browser automation — only after Slack is proven stable

**Deliverable (minimum viable):** Users interact with QE through Slack. Goal results are delivered to a Slack channel. Webhook receiver accepts external events.

**Deliverable (full):** All channels active. Browser automation enables web-based tool use. All channels apply input sanitization and security gating.

**Exit Criteria — minimum viable (all must pass):**
- [ ] Slack bot responds to DMs and channel mentions within 5 seconds
- [ ] `@qe goal "Research X"` in Slack submits a goal and delivers the result to the same thread
- [ ] Webhook receiver processes 5 test payloads with correct envelope generation
- [ ] Input sanitization blocks prompt injection test patterns via Slack messages
- [ ] Notification router delivers goal completion to Slack within 10 seconds of completion

**Exit Criteria — full (all must pass):**
- [ ] All 4 channels (Slack, Telegram, email, webhook) handle 10 test messages each
- [ ] Browser automation completes 5 test navigation tasks in sandbox
- [ ] Cross-channel: goal submitted via Telegram, result delivered via email

---

## Part 4: Complete Bus Topic Registry

The final system uses these bus topics:

```python
TOPICS = {
    # Ingestion
    "observations.raw",
    "observations.structured",

    # Goal orchestration
    "goals.submitted",
    "goals.enriched",
    "goals.completed",
    "goals.failed",
    "goals.paused",
    "goals.resumed",

    # Task decomposition
    "tasks.planned",
    "tasks.dispatched",
    "tasks.completed",
    "tasks.verified",
    "tasks.verification_failed",
    "tasks.recovered",
    "tasks.failed",
    "tasks.progress",
    "tasks.checkpoint",

    # Belief Ledger
    "claims.proposed",
    "claims.committed",
    "claims.challenged",
    "claims.superseded",
    "claims.verification_requested",
    "predictions.proposed",
    "predictions.committed",
    "predictions.resolved",
    "null_results.committed",

    # Investigation
    "investigations.requested",
    "investigations.completed",

    # Analysis & Synthesis
    "analysis.requested",
    "analysis.completed",
    "synthesis.requested",
    "synthesis.completed",

    # HIL
    "hil.approval_required",
    "hil.approved",
    "hil.rejected",

    # Memory
    "memory.updated",
    "memory.preference_set",
    "memory.entity_inferred",

    # Monitoring
    "monitor.scheduled",
    "monitor.triggered",
    "monitor.alert",

    # Voice & Multimodal
    "voice.ingested",
    "voice.transcribed",
    "document.ingested",
    "document.parsed",

    # Channels & Notifications
    "channel.message_received",
    "channel.message_sent",
    "notification.queued",
    "notification.delivered",

    # Security
    "system.security_alert",
    "system.gate_denied",
    "system.integrity_violation",

    # System
    "system.heartbeat",
    "system.error",
    "system.circuit_break",
    "system.service_stalled",
    "system.service_restarted",
    "system.budget_alert",
    "system.resource_alert",
    "system.digest",
}
```

---

## Part 5: Complete Service Registry

The final system runs these services:

| Service | Genome | Subscribes | Publishes | Uses LLM | Uses Tools |
|---|---|---|---|---|---|
| **Planner** | `planner.toml` | `goals.enriched` | `tasks.planned` | Yes | No |
| **Dispatcher** | `dispatcher.toml` | `tasks.planned`, `tasks.verified`, `tasks.failed`, `tasks.recovered` | `tasks.dispatched`, `tasks.progress`, `tasks.checkpoint`, `goals.completed`, `goals.failed` | No | No |
| **Researcher** | `researcher.toml` | `tasks.dispatched` (research type) | `tasks.completed` | Yes | web_search, web_fetch |
| **Fact-Checker** | `fact_checker.toml` | `claims.proposed`, `claims.verification_requested` | `claims.challenged`, `claims.committed` | Yes | web_search, belief_ledger_read |
| **Analyst** | `analyst.toml` | `tasks.dispatched` (analysis type) | `tasks.completed` | Yes | belief_ledger_read |
| **Writer** | `writer.toml` | `tasks.dispatched` (synthesis type) | `tasks.completed` | Yes | file_write |
| **Coder** | `coder.toml` | `tasks.dispatched` (code type) | `tasks.completed` | Yes | code_execute, file_write |
| **Verification** | `verification.toml` | `tasks.completed` | `tasks.verified`, `tasks.verification_failed` | Optional | belief_ledger_read, web_search |
| **Recovery** | `recovery.toml` | `tasks.verification_failed`, `tasks.failed` | `tasks.recovered`, `hil.approval_required` | Optional | varies |
| **Memory** | `memory.toml` | `goals.submitted`, `claims.committed` | `goals.enriched`, `memory.updated` | Optional | belief_ledger_read |
| **HIL** | `hil.toml` | `hil.approval_required` | `hil.approved`, `hil.rejected` | No | No |
| **Monitor** | `monitor.toml` | `monitor.scheduled` | `observations.structured`, `monitor.alert` | No | web_search |
| **Digest** | `digest.toml` | (scheduled) | `system.digest` | No | No |
| **Security** | `security_monitor.toml` | `system.*` | `system.security_alert`, `system.integrity_violation` | No | No |
| **Voice Ingest** | `voice_ingest.toml` | `voice.ingested` | `voice.transcribed`, `observations.structured` | No | No |
| **Doc Ingest** | `doc_ingest.toml` | `document.ingested` | `document.parsed`, `observations.structured` | No | No |
| **Notification** | `notification.toml` | `goals.completed`, `hil.approval_required`, `system.digest` | `notification.delivered` | No | No |

---

## Part 6: Data Architecture

### SQLite Tables (Final)

```
belief_ledger.db
├── claims              — Core belief storage with supersession
├── predictions         — Predictive claims with resolution tracking
├── null_results        — Epistemic absence tracking
├── goals               — Goal state persistence
├── checkpoints         — Execution state snapshots for rollback
├── memory_entries      — User preferences, project context, entity memory
├── projects            — Project definitions
├── project_claims      — Claim-to-project associations
├── failure_records     — Failure patterns and recovery outcomes
├── calibration_records — Confidence calibration data points
├── routing_records     — Model performance on task types
├── planning_patterns   — Successful decomposition templates
├── budget_records      — Per-call cost tracking
├── embeddings          — Vector store (claim_id, embedding BLOB, metadata)
├── hil_decisions       — Historical approve/reject with features
├── tool_usage          — Tool call frequency per service for pruning
├── cost_records        — Per-envelope cost tracking with model/service/goal
├── gate_decisions      — Security gate allow/deny/escalate log
├── channel_messages    — Inbound/outbound channel message log
├── voice_segments      — Speaker-attributed transcript segments with timestamps
├── integrity_hashes    — Known-good hashes of genome/config files
└── _yoyo_migration     — Migration tracking
```

### File System Layout

```
question-engine/
├── data/
│   ├── belief_ledger.db          — Main database
│   ├── hil_queue/
│   │   ├── pending/              — Pending HIL requests
│   │   └── completed/            — Completed HIL decisions
│   ├── workspaces/
│   │   └── {goal_id}/            — Per-goal sandboxed workspace
│   ├── runtime_inbox/            — Cross-process envelope relay
│   └── monitors/                 — Monitor schedules
├── cold/
│   └── YYYY/MM/                  — Archived envelopes by date
├── genomes/                      — Service configuration TOML files
├── src/qe/                       — Source code
│   ├── api/                      — FastAPI REST + WebSocket + voice WS
│   ├── bus/                      — Event bus (memory, protocol)
│   ├── channels/                 — Channel adapters (Slack, Telegram, email, webhook)
│   ├── cli/                      — Typer CLI
│   ├── ingest/                   — Voice, document, OCR ingestion
│   ├── kernel/                   — Supervisor, registry, watchdog
│   ├── models/                   — Pydantic models (envelope, claim, goal, genome)
│   ├── runtime/                  — Router, context, budget, tools, security, H-neurons
│   ├── sdk/                      — Service and tool SDK
│   ├── services/                 — All service implementations
│   ├── substrate/                — Belief ledger, embeddings, memory, goal store
│   └── tools/                    — Built-in tools (web, file, code, browser)
└── config.toml                   — System configuration
```

---

## Part 7: Dependency Summary

### Required (free, no API keys needed)

```toml
dependencies = [
    # Existing
    "litellm>=1.30",
    "instructor>=1.2",
    "aiosqlite>=0.20",
    "pydantic>=2.6",
    "typer>=0.12",
    "rich>=13",
    "watchdog>=4",
    "yoyo-migrations>=8",
    "jinja2>=3.1",
    "tiktoken>=0.7",
    "python-dotenv>=1.0",
    "tenacity>=8",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "websockets>=14",
    # New
    "sentence-transformers>=3.0",   # Local embedding model (free)
    "numpy>=1.26",                   # Vector operations
    "httpx>=0.27",                   # HTTP client for tools
    "trafilatura>=1.8",              # Web content extraction
]
```

### Optional (enhances capability)

```toml
[project.optional-dependencies]
vectors = ["faiss-cpu>=1.8"]                # Fast vector search for large deployments
bayesian = ["pgmpy>=0.0.4"]                # Bayesian network inference
conformal = ["mapie>=0.9"]                 # Conformal prediction
search = ["duckduckgo-search>=6"]          # Fallback web search
voice = [
    "faster-whisper>=1.0",                  # Speech-to-text (production STT)
    "pyannote.audio>=4.0",                  # Speaker diarization
    "silero-vad>=5.0",                      # Voice activity detection
    "kokoro>=0.5",                          # Text-to-speech (Apache 2.0, <0.3s)
    "piper-tts>=2.0",                       # Lightweight TTS fallback (MIT)
    "soundfile>=0.12",                      # Audio I/O
    "noisereduce>=3.0",                     # Audio preprocessing
]
documents = [
    "pymupdf>=1.24",                        # PDF extraction
    "python-docx>=1.1",                    # DOCX parsing
    "openpyxl>=3.1",                        # Excel parsing
    "pytesseract>=0.3",                     # OCR (requires tesseract binary)
    "Pillow>=10.0",                         # Image preprocessing
]
h-neurons = [
    "torch>=2.0",                           # Model internals access
    "scikit-learn>=1.4",                    # H-Neuron identification
]
slack = ["slack-bolt>=1.18"]                # Slack bot integration
telegram = ["python-telegram-bot>=21"]      # Telegram bot integration
email = ["aiosmtplib>=3.0", "aioimaplib>=1.1"]  # Email integration
browser = ["playwright>=1.44"]              # Browser automation
```

---

## Part 8: Implementation Priority and Dependencies

```
Phase 1:  Foundation ────────────────────────── [Week 1-2]
    │
Phase 2:  Embeddings ────────────────────────── [Week 2-3]
    │
Phase 3:  Task Decomposition ────────────────── [Week 3-5]
    │                                              │
Phase 4:  Verification & Recovery ── Phase 5: Tools & Security [Week 5-9]
    │                                    │
Phase 6:  Memory ────────────────────────────── [Week 9-10]
    │
Phase 7:  New Services ──────────────────────── [Week 10-12]
    │
Phase 8:  Learning Loops ────────────────────── [Week 12-14]
    │
Phase 9:  Free Model & Cost Optimization ────── [Week 14-15]
    │
Phase 10: Dashboard & Observability ─────────── [Week 15-17]
    │
Phase 11: Self-Management & Security Monitor ── [Week 17-18]
    │
Phase 12: SDK & Extensibility ───────────────── [Week 18-19]
    │                │
Phase 13: H-Neurons ── Phase 14: Advanced Intelligence [Week 19-22]
                                    │
Phase 15: Voice & Multimodal ── Phase 16: Communication [Week 22-26]
```

Phases 4 and 5 can proceed in parallel. Phases 13 and 14 can proceed in parallel. Phases 15 and 16 can proceed in parallel. All other phases are sequential — each builds on the previous.

**Cross-cutting concerns applied throughout all phases:**
- Question-driven reasoning protocols are embedded in genome templates starting Phase 1
- Epistemic calibration markers are required in all service outputs starting Phase 3
- Problem representation protocol is part of the planner from Phase 3 onward
- Adversarial review is part of verification from Phase 4 onward
- Security gating and input sanitization are enforced from Phase 5 onward
- Cost observability and ephemeral output management are active from Phase 9 onward
- H-Neuron integration enhances verification when enabled in Phase 13

---

## Part 9: Theoretical Foundations

This plan is grounded in research from multiple disciplines:

### Problem-Solving Science
- **Newell & Simon (1972):** Problem space theory — problems have start states, goal states, and operators. The search through this space uses heuristics (means-end analysis, hill climbing, working backward). This directly informs the planner's search strategy selection.
- **Gestalt Psychology (Wertheimer, 1945):** Productive thinking requires *restructuring* the problem, not applying past patterns. Functional fixedness prevents seeing concepts beyond familiar uses. This is why the planner does representation before decomposition.
- **Polya (1945):** Four-step model — understand, plan, execute, look back. The "look back" step (reflection/generalization) maps to the learning loops.
- **TRIZ (Altshuller, 1969):** 40 inventive principles from 400,000 patents. Every hard problem contains a contradiction. The ideal solution solves the problem without adding complexity. Directly informs the planner's contradiction detection and the ideality principle for agent actions.
- **Kahneman (2011):** Dual-process theory — System 1 (fast, intuitive) and System 2 (slow, deliberate). LLMs are pure System 1. Every failure is System 1 applied where System 2 was needed. This is why the verification service implements adversarial review (System 2 analog).
- **Rittel & Webber (1973):** Wicked problems have no definitive formulation, no stopping rule, and every solution changes the problem. Requires adaptive/iterative approaches. Informs the probe-observe-reframe protocol for ill-defined goals.

### Question Science
- **Berlyne (1954-1966):** Questions create epistemic curiosity and prime attention for relevant information. A question held in mind acts as a background processing directive.
- **Zeigarnik (1927):** Open questions are remembered far better than resolved ones. The brain maintains "open files" on unresolved questions. This is why anchor questions maintain coherence in long-running workflows.
- **Gregersen (2018):** Question Burst methodology — the most generative questions appear between question 8 and 15. The first 7 are obvious. This informs the depth of self-questioning protocols.
- **Roger Martin:** "What would have to be true for X to be the right answer?" — converts debate into conditions-discovery. The most powerful question structure in strategy and problem solving. Used in the falsification step of adversarial review.

### LLM Failure Mode Research
- **H-Neurons (Tsinghua, Dec 2025):** <0.1% of neurons predict hallucination with 70-83% cross-domain accuracy. H-Neurons are causally linked to over-compliance. They originate in pre-training and survive RLHF. Suppression factor of 0.5 is the sweet spot.
- **Structural hallucination limits (ScienceDirect, Dec 2025):** Hallucination is partially inherent to architecture — the same mechanism that enables generalization enables hallucination. The goal is management, not elimination.
- **Training incentive alignment (OpenAI, Sep 2025):** Training objectives and leaderboards reward confident guessing over calibrated uncertainty. Models learn to bluff because bluffing scores higher.
- **Chi, Feltovich & Glaser (1981):** Novices categorize by surface features, experts by deep structure. LLMs are structurally novices — the deep structure extraction protocol forces expert-like reasoning.

### Reliability Engineering
- Checkpoint/rollback → database transaction semantics
- Circuit breakers → distributed systems resilience patterns
- Anomaly detection → statistical process control
- Verification contracts → design by contract (Meyer, Eiffel)
- Multi-model consensus → Byzantine fault tolerance

---

## Part 10: What Success Looks Like

When all 16 phases are complete:

1. A user says "Research fusion energy and produce a briefing" — via CLI, dashboard, Slack, Telegram, email, or voice — and walks away.

2. The system enriches the goal with relevant memory and past knowledge, decomposes it into a verified DAG of subtasks, dispatches subtasks to specialized services using the optimal model for each, verifies every output against execution contracts and the belief ledger, recovers automatically from failures, integrates verified claims into persistent knowledge, and synthesizes a final deliverable with a confidence receipt and full provenance.

3. The user gets notified on their preferred channel (Slack DM, Telegram message, email) that the briefing is complete. The result includes: an overall confidence score that is empirically calibrated, source citations for every claim, a verification report showing what was checked, a provenance chain for any sentence they want to drill into, and a cost report showing exactly what was spent.

4. The system is now better than it was before this goal — the routing optimizer has more data, the calibration tracker is more accurate, the failure knowledge base has new entries, and the planner has a new successful template to draw from.

5. The next goal on a similar topic starts faster and produces better results because the belief ledger already contains verified claims from this one.

6. All of this runs on free local models via Ollama. Paid models are available as optional upgrades for speed and quality, but the system is fully functional without them. Cost observability shows exactly where every token goes, and the system actively minimizes spend through prompt caching, lazy tool discovery, ephemeral output management, and multi-factor model routing.

7. When running on local models, H-Neuron suppression provides a neurological-level reliability improvement that no cloud-only platform can replicate. The system doesn't just check outputs for hallucinations — it monitors the neural circuits that cause them in real time.

8. Every service's system prompt includes question-driven reasoning protocols that reduce hallucination, sycophancy, and shallow reasoning at zero additional compute cost. The planner represents problems before decomposing them. The verification service runs adversarial review on every output. Every claim is tagged with epistemic status.

9. Voice recordings, PDFs, images, spreadsheets, and emails are all first-class input sources. A user can record a meeting and the system extracts claims with speaker attribution and audio timestamps. OCR makes handwritten notes and whiteboard photos searchable.

10. Security is enforced at every layer: bus-level tool call gating, input sanitization on all external sources, ephemeral credential handling, integrity monitoring, and behavioral auditing. The system learned from OpenClaw's security failures — the "lethal trifecta" is mitigated by design.

11. The architecture is modular, composable, and lean. Deterministic orchestration (dispatcher) with LLMs at the creative leaf nodes — the pattern that OpenClaw's community of 175,000+ users validated as the most reliable. MCP integration provides access to 1000+ external tools without building custom implementations.

12. The entire system is self-hostable, open-source (AGPL-3.0), and extensible via TOML genome files and the SDK. The theoretical foundations span problem-solving science, question science, LLM failure mode research, distributed systems reliability engineering, and hard-won lessons from the OpenClaw ecosystem.

---

## Appendix A: The Reliability Stack (Combined Interventions)

The system applies reliability interventions at every level, combining for maximum effect:

| Layer | Intervention | Hallucination Reduction | Cost |
|---|---|---|---|
| **Prompt** | Epistemic pressure + question protocols | ~20-30% | Free (zero extra compute) |
| **Generation** | H-Neuron suppression (local models) | ~20-35% | Free (negligible overhead) |
| **Post-generation** | Adversarial self-review | ~30-40% | 2-3x inference per subtask |
| **Post-generation** | Self-consistency sampling | ~25-35% | 2-5x inference (high-stakes only) |
| **Verification** | H-Neuron activation monitoring | Detection | Free (forward hooks) |
| **Verification** | External HHEM detection | Detection + retry | Lightweight model inference |
| **Verification** | Belief ledger cross-reference | Detection + contradiction resolution | Embedding similarity search |
| **Verification** | Contract postcondition checks | Detection + retry | Deterministic (no LLM) |
| **Knowledge** | RAG with strict grounding | ~60-80% | Retrieval + context injection |
| **Knowledge** | Bayesian confidence propagation | Calibration | Deterministic graph computation |
| **Learning** | Calibration tracking | Improving over time | Statistical table lookup |
| **Learning** | Failure knowledge base | Avoidance | Database query |
| **Combined** | **All layers active** | **~80-90%** | **Production-grade** |

The interventions are additive. Each catches failures the others miss. The combined stack achieves production-grade reliability not through any single technique but through defense in depth.

## Appendix B: Cost Optimization Stack

| Technique | Source | Savings | Implementation Phase |
|---|---|---|---|
| **Tool Discovery pattern** | OpenClaw Token Optimizer | Up to 90% tool context overhead | Phase 5 |
| **Ephemeral tool outputs** | OpenClaw community | 20-30% of total context | Phase 5 |
| **Prompt cache alignment** | OpenClaw community | 90% on cache hits (Anthropic) | Phase 9 |
| **LLM-summarized compaction** | OpenClaw community | 40-50% context reduction | Phase 9 |
| **Multi-factor model routing** | Manifest (23-dim scoring) | Up to 25x per-query (right-sizing) | Phase 8 |
| **Tool/skill pruning** | OpenClaw community | ~3,200 tokens/request per unused tool | Phase 9 |
| **Budget gating with auto-downgrade** | QE budget tracker | Prevents overruns | Phase 1 |
| **Per-envelope cost tracking** | Clawmetry pattern | Visibility → informed optimization | Phase 10 |
| **Combined** | **All techniques active** | **60-80% total cost reduction** | **Production** |

## Appendix C: Security Architecture

| Layer | Mechanism | Threat Mitigated | Implementation Phase |
|---|---|---|---|
| **Input sanitization** | Pattern matching + wrapping | Prompt injection via external content | Phase 5 |
| **Bus-level tool gating** | Policy enforcement in code | Unauthorized tool use, scope violation | Phase 5 |
| **Capability enforcement** | Genome declarations checked at runtime | Service privilege escalation | Phase 5 |
| **Sandboxed execution** | Subprocess isolation, workspace scoping | Code execution escape, file system access | Phase 5 |
| **Integrity monitoring** | Hash verification of genome/config files | Unauthorized configuration modification | Phase 11 |
| **Behavioral auditing** | Anomaly detection on service patterns | Compromised service, injection persistence | Phase 11 |
| **Dual-stack enforcement** | Security logic in code, not prompts | Prompt injection bypassing constitutions | Phase 5 |
| **MCP signature verification** | HMAC on external tool calls | Malicious MCP server responses | Phase 5 |
| **Channel input sanitization** | Per-adapter filtering | Cross-channel prompt injection | Phase 16 |

**Design Principle:** "Skills can be overridden by prompt injection. Plugins cannot." — SecureClaw. All security-critical enforcement is in the service/runtime layer (code), never solely in the prompt layer (constitutions).

## Appendix D: Question Protocols Quick Reference

**Problem Representation (Planner):**
1. Restate the core problem
2. Distinguish asked (X) vs. needed (Y) vs. underlying need (Z)
3. Identify hard constraints
4. Define success criteria
5. Classify problem type
6. Detect contradictions
7. Select search strategy

**Metacognitive Self-Check (All Services):**
- What am I confident about? Uncertain about?
- What am I assuming without verification?
- How could this output be wrong?

**Anti-Hallucination (All Services):**
- Dense training data or sparse?
- Pattern-completing or directly knowing?
- Know WHY or just THAT?
- Tag: [VERIFIED] [INFERRED] [ESTIMATED] [UNCERTAIN] [UNKNOWN]

**Anti-Sycophancy (All Services):**
- Strongest argument against my conclusion?
- Would I say this if user believed the opposite?
- Agreeing because evidence or because comfortable?

**Adversarial Review (Verification Service):**
- What assumptions haven't been verified?
- What would have to be true for this to be wrong?
- What's the strongest counterargument?
- Which claims are most likely wrong and why?

**Coherence Maintenance (Long-Running Workflows):**
- What is the core problem we're solving?
- What has been established as true?
- Am I still serving the original goal?
- Have I contradicted earlier statements?

---

*This plan is a living document. Each phase should be revisited and adjusted based on what is learned during implementation. It incorporates lessons from the OpenClaw ecosystem (175K+ GitHub stars, 5,700+ community skills, 40K+ security incidents), voice/multimodal technology research (faster-whisper, WhisperX, Kokoro TTS, Pipecat), problem-solving science, question science, LLM failure mode research (H-Neurons), and distributed systems reliability engineering.*
