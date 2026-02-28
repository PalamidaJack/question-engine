# Cutting-Edge Research & Innovation Radar

**Date:** 2026-02-28
**Scope:** Post-improvement-plan research — innovations and architectural patterns not yet integrated into `IMPROVEMENT_PLAN.md`

---

## Table of Contents

1. [Tier 1 — Adopt Now](#tier-1--adopt-now)
2. [Tier 2 — Next Phase](#tier-2--next-phase)
3. [Tier 3 — Watch](#tier-3--watch)
4. [Agent SDK & Framework Patterns](#agent-sdk--framework-patterns)
5. [Synthesized Architecture Map](#synthesized-architecture-map)
6. [Appendix: Source Matrix](#appendix-source-matrix)

---

## Tier 1 — Adopt Now

High-confidence innovations with clear QE integration points. Low risk, proven in production or peer-reviewed.

### 1.1 Adaptive Graph of Thoughts (AGoT)

**What:** Recursive DAG decomposition of complex reasoning tasks. Unlike linear chain-of-thought, AGoT dynamically builds and refines a directed acyclic graph where each node is a reasoning step that can branch, merge, or loop.

**Results:** +32.4% accuracy over chain-of-thought on complex multi-step tasks. Particularly strong on tasks requiring backtracking or synthesis of multiple evidence streams.

**QE Mapping:**
- **Goal decomposition** — Replace the current linear subtask expansion in the GoalManager with AGoT-style recursive decomposition. Each subtask becomes a DAG node; dependencies become edges.
- **Claim reconciliation** — When merging contradictory claims, build a reasoning graph that weighs evidence from multiple sources before producing a merged claim.
- **Implementation surface:** `src/qe/kernel/goal_manager.py`, new `src/qe/reasoning/agoт.py` module

**Effort:** Medium. Requires a graph data structure for reasoning steps and an LLM prompt template that instructs the model to produce structured DAG output.

---

### 1.2 Test-Time Compute Scaling

**What:** Instead of using a larger model, allocate more inference-time compute (retries, verification passes, self-consistency sampling) to a smaller model. A 7B parameter model with 100x compute budget approximates 70B model quality.

**Results:** Empirically validated across multiple benchmarks. The scaling curve is roughly logarithmic — diminishing returns past ~50x, but the first 10x is nearly free in cost terms when using a cheap model.

**QE Mapping:**
- **Adaptive LLM routing** — For high-stakes operations (claim creation, goal decomposition), run the cheap model N times with self-consistency voting instead of one expensive call. The Router already selects models; add a "compute budget" dimension.
- **Confidence calibration** — Use variance across N samples as a confidence signal. If 8/10 runs agree, confidence is high. If 5/10 disagree, escalate to a larger model.
- **Implementation surface:** `src/qe/services/llm_router.py`, new `src/qe/reasoning/compute_scaling.py`

**Effort:** Low-Medium. Self-consistency voting is a wrapper around existing LLM calls. Main cost is latency management (parallel sampling helps).

---

### 1.3 Parallel Guardrails with Tripwire Cancellation

**What (from OpenAI Agents SDK):** Run input/output guardrails as parallel coroutines alongside the main agent task. If any guardrail fires, it triggers a "tripwire" that cancels the in-flight agent work immediately — no wasted tokens.

**Results:** Near-zero latency overhead for safety checks (they run concurrently). Immediate cancellation prevents cost waste on toxic/unsafe outputs.

**QE Mapping:**
- **Bus-level guardrails** — Attach guardrail coroutines to the envelope pipeline. When an envelope enters a service, guardrails and the service handler run in parallel. Tripwire fires → envelope gets rejected before the LLM call completes.
- **Tool call gating** — The existing `ToolGate` concept in the plan can use tripwire cancellation to abort tool execution mid-flight if a guardrail detects policy violation.
- **Implementation surface:** `src/qe/bus.py` (parallel dispatch), `src/qe/security/guardrails.py` (new)

**Effort:** Low. `asyncio.gather` with cancellation is straightforward. The design work is in defining guardrail policies.

---

### 1.4 MAGMA Multi-Graph Memory

**What:** A memory architecture using four co-indexed graphs: semantic (concept similarity), temporal (event ordering), causal (cause→effect), and entity (who/what relationships). Queries fan out across all four graphs and merge results.

**Results:** 95% token reduction in context windows compared to naive "dump all history" approaches. Retrieval relevance significantly higher because queries are matched across multiple dimensions.

**QE Mapping:**
- **Belief ledger enhancement** — The claim store already has entity relationships (subject→predicate→object). Add temporal ordering (claim timestamps), causal links (claim A led to claim B via a reasoning chain), and semantic similarity (embedding-based).
- **Agent memory** — Instead of each genome carrying full conversation history, give agents a MAGMA-style memory interface that retrieves relevant context across all four graph types.
- **Implementation surface:** `src/qe/substrate.py` (extend claim retrieval), new `src/qe/memory/magma.py`

**Effort:** High. Requires embedding infrastructure, graph storage, and multi-graph query fusion. But the payoff (95% context reduction) is enormous for cost optimization.

---

### 1.5 Lazy Tool Loading (MCP November 2025 Spec)

**What:** The latest MCP spec supports lazy tool registration — servers advertise tool names/descriptions but don't load full schemas until a tool is actually invoked. Combined with a "Tool Search Tool" (meta-tool that searches available tools by description), agents see only 3-5 relevant tools per turn instead of 50+.

**Results:** 85-95% context reduction in tool descriptions. Matches the OpenClaw "Tool Discovery" pattern already in the improvement plan, but now with protocol-level support.

**QE Mapping:**
- **Already partially planned** — Phase 5 in the improvement plan has Tool Discovery. This research confirms the approach and provides the MCP protocol backing.
- **New insight:** MCP's `listTools` can be paginated. The meta-tool should use semantic search over tool descriptions, not just name matching.
- **Implementation surface:** `src/qe/tools/discovery.py` (already planned), MCP bridge integration

**Effort:** Low (incremental to planned work). The MCP spec does the heavy lifting.

---

### 1.6 DSPy Prompt Optimization

**What:** DSPy treats prompts as programs with typed Signatures (input→output contracts). The GEPA optimizer automatically tunes prompt wording, few-shot examples, and chain-of-thought structure by hill-climbing on a scoring function.

**Results:** GEPA optimizer improved task accuracy from 37.5% to 80% on complex extraction tasks — without changing the model or the code, only the prompt.

**QE Mapping:**
- **Genome prompt tuning** — Each genome's system prompt and extraction templates could be expressed as DSPy Signatures. Run the optimizer against a held-out validation set of claims to auto-tune prompts.
- **Claim extraction quality** — The extractor service prompt is the highest-leverage optimization target. DSPy can find prompt formulations that produce more accurate, better-calibrated claims.
- **Implementation surface:** New `src/qe/optimization/prompt_tuner.py`, genome TOML format extension for Signature definitions

**Effort:** Medium. Requires a scoring function (how good is this claim extraction?) and a validation dataset. DSPy itself is a pip install.

---

### 1.7 A2A Protocol (Agent-to-Agent)

**What:** Google-initiated, now Linux Foundation-governed protocol for agent interoperability. 150+ organizations participating. Defines: Agent Cards (capability advertisement), Task lifecycle (submitted→working→completed/failed), streaming via SSE, push notifications.

**Results:** Emerging standard. Not yet widely deployed but has critical mass of backers. Complementary to MCP (which is tool-focused; A2A is agent-focused).

**QE Mapping:**
- **External agent federation** — QE genomes could expose A2A Agent Cards, allowing external agents to discover and delegate tasks to QE services. QE could also consume external A2A agents as if they were genomes.
- **Multi-instance QE** — Two QE installations could federate via A2A, sharing specialized genomes across organizational boundaries.
- **Implementation surface:** New `src/qe/protocols/a2a.py`, Agent Card generation from genome TOML

**Effort:** Medium-High. Protocol implementation + security implications of cross-boundary agent communication. Good candidate for Phase 13+ timeframe.

---

## Tier 2 — Next Phase

Promising innovations that need more validation or have higher integration complexity. Target for after core platform stabilizes.

### 2.1 MetaQA Hallucination Detection

**What:** Detects hallucinations by applying metamorphic prompt mutations — rephrasing the same question N ways and checking if the model gives consistent answers. Inconsistency signals hallucination. Works on ANY model (black-box, no access to logits needed).

**Results:** High detection rates across multiple benchmarks. The key insight is that factual knowledge is robust to paraphrasing; hallucinations are not.

**QE Mapping:**
- **Claim verification** — Before a claim enters the belief ledger, run MetaQA: rephrase the source observation 3-5 ways, extract claims from each rephrasing, check consistency. Inconsistent claims get flagged or downweighted.
- **Confidence calibration** — MetaQA consistency score becomes a factor in claim confidence.
- **Implementation surface:** `src/qe/services/claim_extractor.py` (verification step), new `src/qe/verification/metaqa.py`

**Effort:** Medium. Requires N additional LLM calls per claim (cost multiplier). Use selectively — only for high-stakes claims or when confidence is borderline.

---

### 2.2 CLAP (Cross-Layer Attention Probing)

**What:** Per-claim hallucination scoring by probing attention patterns across transformer layers. Unlike MetaQA (black-box), CLAP requires access to model internals (attention weights). Produces a per-token or per-claim hallucination probability.

**Results:** More granular than MetaQA — can identify WHICH specific claim in a multi-claim extraction is hallucinated, not just that something is wrong.

**QE Mapping:**
- **Fine-grained claim scoring** — If using an open-weight model (Llama, Mistral), CLAP can score each extracted claim individually. Useful for the Bayesian confidence layer.
- **Limitation:** Only works with models that expose attention weights. Not applicable to API-only models (Claude, GPT).
- **Implementation surface:** New `src/qe/verification/clap.py`, requires model serving with attention output enabled

**Effort:** High. Requires running open-weight models with attention extraction. Good research bet for Phase 8.5 (Bayesian confidence).

---

### 2.3 LangGraph Typed Reducers

**What:** LangGraph 1.0 introduced typed reducers — functions that define how state merges when parallel branches converge. Instead of "last write wins," you define semantic merge logic (e.g., union for sets, append for lists, max for confidence scores).

**Results:** Eliminates a class of bugs in parallel agent systems where concurrent state updates silently overwrite each other.

**QE Mapping:**
- **Bus state convergence** — When multiple genomes produce claims about the same entity simultaneously, a typed reducer defines the merge semantics (e.g., keep highest confidence, union of evidence, weighted average).
- **Goal subtask merging** — When parallel subtasks complete, their results need deterministic merge into the parent goal state.
- **Implementation surface:** `src/qe/bus.py` (envelope merge logic), `src/qe/kernel/goal_manager.py` (subtask result aggregation)

**Effort:** Low-Medium. The concept is simple (define merge functions per state field). The work is in identifying all merge points and defining correct semantics.

---

### 2.4 MCP Tasks Primitive

**What:** The MCP November 2025 spec added a Tasks primitive — async lifecycle management for long-running tool operations. States: `submitted → working → completed | failed | cancelled`. Supports progress streaming and resumption.

**Results:** Enables tools that take minutes/hours (web scraping, document processing, external API calls) without blocking the agent loop.

**QE Mapping:**
- **Long-running tool support** — Some QE tools (web research, document ingestion, goal execution) are inherently long-running. MCP Tasks gives them a lifecycle that the genome can poll or subscribe to.
- **HIL integration** — Human-in-the-loop approvals are naturally long-running tasks. MCP Tasks lifecycle maps cleanly to the existing HIL queue.
- **Implementation surface:** `src/qe/tools/mcp_bridge.py` (already planned), HIL queue migration

**Effort:** Medium. Requires async task tracking infrastructure. Good synergy with existing HIL implementation.

---

### 2.5 Absolute Zero Reasoner (AZR)

**What:** Self-generated training curricula — the model creates its own training problems, solves them, verifies solutions, and uses the results to improve. No external training data needed. NeurIPS 2025 Spotlight paper.

**Results:** Matches or exceeds models trained on curated datasets for code generation and mathematical reasoning tasks.

**QE Mapping:**
- **Self-improving genomes** — A genome could generate test cases for its own claim extraction, verify them against known-good claims, and refine its approach. This is the "self-improvement" capability in Phase 14.
- **Bootstrapping new domains** — When QE encounters a new domain with no training data, AZR-style self-training could bootstrap competence.
- **Implementation surface:** New `src/qe/optimization/azr.py`, genome self-evaluation framework

**Effort:** High. Research-grade technique. Good fit for Phase 14 (⚗️ Research Bet) if the Go/No-Go criteria are met.

---

## Tier 3 — Watch

Interesting but too early or too niche for QE right now. Monitor for maturation.

### 3.1 Nested Learning

**What:** Training smaller "inner" models to handle routine subtasks, while the larger "outer" model handles novel/complex cases. The inner model learns from the outer model's outputs over time.

**QE Relevance:** Could allow QE to progressively replace expensive LLM calls with fine-tuned small models for routine claim extraction patterns.

**Status:** Research-stage. No production implementations yet. Watch for tooling that makes this practical.

---

### 3.2 Speculative Decoding for Agents

**What:** Run a small draft model to generate candidate actions, then use a large model to verify/correct. If the draft is correct (often 70-80% of the time), you get large-model quality at small-model cost.

**QE Relevance:** Could reduce costs for the LLM Router — draft with a 7B model, verify with Claude/GPT only when the draft model is uncertain.

**Status:** Well-understood for text generation, but applying it to agentic tool-use decisions is still experimental.

---

### 3.3 Flow of Thoughts (FoT)

**What:** Collaborative reasoning across multiple model instances that share intermediate "thought" tokens via a shared buffer. Like pair programming but for LLMs.

**QE Relevance:** Multiple genomes could share reasoning context when working on related subtasks of the same goal.

**Status:** Interesting research direction. No practical frameworks yet.

---

### 3.4 Verification via Computational Graphs

**What:** Express verification logic as a computational graph (like a circuit) where each node is a checkable assertion. The graph structure makes it possible to formally verify that a reasoning chain is sound.

**QE Relevance:** Could provide formal verification for claim derivation chains — prove that claim C logically follows from claims A and B.

**Status:** Early research. Potentially high value for trust/auditability but years from practical tooling.

---

## Agent SDK & Framework Patterns

Architectural patterns extracted from production agent frameworks (as of February 2026) that are directly applicable to QE's design.

### 4.1 Anthropic Claude Code SDK

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Subagent context isolation** | Each spawned agent gets a fresh context window. Parent passes only the task description, not full history. | Genome isolation — each genome invocation should get only relevant context, not the full bus history. |
| **In-process MCP servers** | Tools can be MCP servers running in the same process. No network overhead for local tools. | QE tools that are Python functions can use in-process MCP, avoiding serialization overhead. |
| **Tool Search Tool** | A meta-tool that searches available tools by description. Agent sees only search results, not all tools. | Direct validation of the Tool Discovery pattern in Phase 5. |
| **Foreground/background agents** | Foreground blocks until result; background notifies on completion. | Goal subtasks: blocking subtasks (need result before proceeding) vs. fire-and-forget research tasks. |
| **Resume with full context** | Agents can be resumed by ID, continuing with their full prior context preserved. | Genome invocation continuity — resume a paused genome without re-establishing context. |

---

### 4.2 OpenAI Agents SDK

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Guardrails as parallel coroutines** | Safety checks run concurrently with agent work. Tripwire cancels in-flight work if guardrail fires. | See Tier 1.3 above — direct adoption target. |
| **Handoffs with history compression** | When agent A hands off to agent B, the conversation history is compressed to only relevant context. | Genome-to-genome handoff: when one genome delegates to another, compress the envelope history. |
| **Sessions with pluggable backends** | Session state stored in pluggable backends (Redis, Postgres, file). Agent code doesn't know or care. | QE substrate abstraction already does this. Validates the approach. |
| **Typed tool outputs** | Tools return typed objects, not raw strings. The framework handles serialization. | Enforce typed returns from QE tools to prevent the "tool output bloat" problem. |

---

### 4.3 Google Agent Development Kit (ADK)

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Event-driven runtime** | Agents yield event streams, not return values. The runtime processes events asynchronously. | QE's bus IS an event stream. Genomes should yield envelopes, not return results. This validates QE's core architecture. |
| **7 named multi-agent patterns** | Sequential, Parallel, Loop, Delegation, Mixture-of-Agents, Transfer, Debate. Each is a first-class pattern with runtime support. | QE goal decomposition should support these patterns explicitly. Currently only sequential and parallel. |
| **AutoFlow with transfer_to_agent** | Runtime automatically selects which agent to delegate to based on task description. | Dynamic genome routing — the supervisor selects which genome handles a subtask based on its description and the genome's advertised capabilities. |
| **Shared session state with output_key** | Each agent writes to a named key in shared state. Other agents read from those keys. No message passing needed. | Shared substrate state: genome A writes claim to substrate, genome B reads it. The substrate IS the shared state. Validates QE's approach. |

---

### 4.4 LangGraph 1.0

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Typed reducers for state merge** | Define how parallel branch results merge: union, append, max, custom function. | See Tier 2.3 above. |
| **Checkpoint-per-super-step** | Every "super step" (batch of parallel node executions) gets a checkpoint. Enables time-travel debugging and replay. | Goal execution checkpointing: save state after each batch of subtask completions. Enables replay and debugging. |
| **interrupt() / Command(resume=)** | First-class HIL primitives. `interrupt()` pauses execution and yields to human. `Command(resume=value)` continues with human's input. | Direct mapping to QE's HIL system. `interrupt()` = HIL proposal envelope. `Command(resume=)` = HIL approval envelope. |
| **Configurable recursion limits** | Prevent infinite agent loops with hard recursion limits per graph. | Genome invocation depth limits — prevent a genome from recursively spawning sub-genomes indefinitely. |

---

### 4.5 CrewAI Flows

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **@listen decorators (event-bus pattern)** | Methods decorated with `@listen(event)` automatically trigger when that event is published. Declarative event wiring. | Genome subscription: genomes declare which bus topics they listen to via decorators or TOML config. Already the QE model — validates approach. |
| **4-tier memory** | Short-term (current task), Long-term (persistent), Entity (knowledge graph), Contextual (per-task context). | Maps to QE: Short-term = envelope context, Long-term = belief ledger, Entity = claim graph, Contextual = goal context. |
| **A2A async chain** | Agents communicate via A2A protocol with async task handoff. | See Tier 1.7 above. |

---

### 4.6 Microsoft Agent Framework

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Typed-edge graph with conditional routing** | Agent topology is a graph with typed edges. Edge conditions determine which path is taken. | Goal execution graph: subtask dependencies are typed edges. Conditions (claim confidence > threshold) determine whether to proceed or loop. |
| **Middleware interceptors** | Pre/post processing hooks on every agent invocation. Used for logging, auth, rate limiting. | Bus middleware: pre/post hooks on every envelope. Already partially implemented — validates and extends. |

---

### 4.7 Pydantic AI

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Type-safe structured output** | LLM outputs are validated against Pydantic models. If validation fails, the framework retries with the error message. | Claim extraction should return `Claim` Pydantic models directly. If the LLM output doesn't validate, retry with the validation error. |
| **Dependency injection for tools** | Tools receive dependencies (DB connections, API clients) via DI, not global state. | QE tools should receive `Substrate` via DI, not import it globally. Improves testability. |
| **5 levels of multi-agent complexity** | Single agent → tool-calling agent → agent-with-memory → multi-agent → agent swarm. Each level adds exactly one capability. | Progressive genome complexity: start simple, add capabilities incrementally. Don't build the swarm before the single agent works. |

---

### 4.8 DSPy

| Pattern | Description | QE Application |
|---------|-------------|----------------|
| **Programming over prompting** | Define typed Signatures (input→output), not prose prompts. The framework generates optimal prompts. | See Tier 1.6 above. |
| **Automatic prompt optimization** | GEPA optimizer hill-climbs on prompt quality metrics. | See Tier 1.6 above. |
| **Composable modules** | DSPy modules compose like PyTorch modules — chain, branch, loop. | Genome prompt chains could be expressed as DSPy module compositions. |

---

## Synthesized Architecture Map

How these innovations map to QE's existing architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        QE INNOVATION LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  REASONING   │  │   MEMORY     │  │     VERIFICATION       │ │
│  │             │  │              │  │                        │ │
│  │  AGoT DAG   │  │  MAGMA       │  │  MetaQA (black-box)   │ │
│  │  decomp     │  │  4-graph     │  │  CLAP (open-weight)   │ │
│  │             │  │  retrieval   │  │  Self-consistency      │ │
│  │  Test-time  │  │              │  │  voting                │ │
│  │  compute    │  │  95% token   │  │                        │ │
│  │  scaling    │  │  reduction   │  │  Per-claim scoring     │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         │                │                      │               │
│  ┌──────┴────────────────┴──────────────────────┴────────────┐ │
│  │                    BUS + GUARDRAILS                        │ │
│  │                                                            │ │
│  │  Parallel guardrails ──── Tripwire cancellation            │ │
│  │  Typed reducers ────────── State merge semantics           │ │
│  │  Middleware interceptors ─ Pre/post hooks                  │ │
│  │  Checkpoint-per-step ──── Time-travel debugging            │ │
│  └──────┬────────────────────────────────────────┬────────────┘ │
│         │                                        │               │
│  ┌──────┴──────────┐                  ┌──────────┴─────────────┐│
│  │  TOOL SYSTEM    │                  │  OPTIMIZATION          ││
│  │                 │                  │                        ││
│  │  MCP lazy load  │                  │  DSPy prompt tuning    ││
│  │  Tool Search    │                  │  Compute budget mgmt   ││
│  │  MCP Tasks      │                  │  AZR self-training     ││
│  │  (async lifecycle)                 │  Nested learning       ││
│  │  Typed outputs  │                  │  (future)              ││
│  └──────┬──────────┘                  └──────────┬─────────────┘│
│         │                                        │               │
│  ┌──────┴────────────────────────────────────────┴────────────┐ │
│  │                    PROTOCOLS                               │ │
│  │                                                            │ │
│  │  MCP (tool interop) ──────── A2A (agent interop)           │ │
│  │  Agent Cards ─────────────── Task lifecycle                │ │
│  │  Federation ──────────────── Cross-instance QE             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Priority Integration Sequence

| Priority | Innovation | Blocks | Effort | Impact |
|----------|-----------|--------|--------|--------|
| **P0** | Parallel guardrails + tripwire | Nothing | Low | Safety + cost |
| **P0** | Test-time compute scaling | LLM Router | Low-Med | Quality + cost |
| **P1** | DSPy prompt optimization | Validation dataset | Medium | Quality |
| **P1** | Typed reducers | Bus refactor | Low-Med | Correctness |
| **P1** | MCP lazy loading | Phase 5 | Low | Context cost |
| **P2** | MAGMA multi-graph memory | Embedding infra | High | Context cost |
| **P2** | AGoT reasoning | Goal manager | Medium | Quality |
| **P2** | MetaQA verification | Cost budget | Medium | Trust |
| **P3** | A2A protocol | Core stable | Med-High | Extensibility |
| **P3** | MCP Tasks | Phase 5 | Medium | Capability |
| **P3** | AZR self-training | Phase 14 | High | Autonomy |

---

## Appendix: Source Matrix

| Innovation | Origin | Year | Validation Level |
|-----------|--------|------|-----------------|
| AGoT | Academic paper | 2025 | Peer-reviewed, benchmarked |
| Test-time compute scaling | DeepMind / academic | 2024-2025 | Multiple independent replications |
| Parallel guardrails | OpenAI Agents SDK | 2025 | Production (OpenAI internal) |
| MAGMA | Academic paper | 2025 | Peer-reviewed |
| MCP lazy loading | Anthropic MCP spec | Nov 2025 | Specification, early adopters |
| DSPy / GEPA | Stanford NLP | 2024-2025 | Open source, community validated |
| A2A Protocol | Google → Linux Foundation | 2025 | 150+ organizations, specification |
| MetaQA | Academic paper | 2025 | Peer-reviewed, benchmarked |
| CLAP | Academic paper | 2025 | Peer-reviewed |
| LangGraph reducers | LangChain | 2025 | Production (LangGraph 1.0) |
| MCP Tasks | Anthropic MCP spec | Nov 2025 | Specification |
| AZR | Academic paper | 2025 | NeurIPS Spotlight |
| Nested Learning | Academic paper | 2025 | Research-stage |
| Speculative decoding | Multiple | 2024-2025 | Production for text gen, experimental for agents |
| FoT | Academic paper | 2025 | Research-stage |
| TALE-EP / BudgetThinker | Academic | 2025 | Peer-reviewed, 67% cost reduction claim |
