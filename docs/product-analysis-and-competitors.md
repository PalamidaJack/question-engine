# Question Engine OS — Product Analysis & Competitive Landscape

*Last updated: 2026-02-28*

---

## What It Does

Question Engine OS is a **self-hosted, multi-agent AI orchestration platform** that turns natural-language goals into executed, verified results through an automated pipeline. Think of it as an AI operations system that plans, executes, self-heals, and accumulates knowledge over time.

### End-to-End Flow

```
User Input (Telegram/Slack/Email/Webhook/API)
  → Command Classification (goal / question / status)
  → PlannerService decomposes goal into a DAG of typed subtasks
  → Dispatcher walks the DAG, dispatching ready subtasks
  → ExecutorService runs each via LLM (with tool access)
  → VerificationGate validates output (structural/contract/anomaly)
  → RecoveryOrchestrator handles failures (retry/escalate/HIL)
  → Results delivered back via originating channel
  → Knowledge extracted and stored in Belief Ledger
```

### Core Capabilities

| Capability | Details |
|---|---|
| **Goal Decomposition** | LLM-powered planner breaks goals into subtask DAGs with dependencies, execution contracts, and model tier assignments. Caches successful patterns for reuse (85% similarity threshold). |
| **7 Task Types** | research, analysis, fact_check, synthesis, code_execution, web_search, document_generation |
| **Multi-Model Orchestration** | 8+ LLM providers via litellm (OpenAI, Anthropic, Google, Groq, Mistral, Together, Kilo Code, Ollama). 4 model tiers: local → fast → balanced → powerful. |
| **Verification Gates** | 3-layer validation: structural checks (non-empty, well-formed), contract postconditions (e.g., `score >= 0.5`), statistical anomaly detection (2σ outlier flagging). |
| **Self-Healing Recovery** | Failure classification (transient/capability/approach/specification/unrecoverable) with graduated strategies: retry with backoff, model escalation, simplified prompt, checkpoint rollback, human-in-the-loop. Failure KB learns from past errors. |
| **Belief Ledger** | Persistent knowledge base storing structured claims with confidence scores, decay over time (30-day half-life), supersession, and contradiction detection. The system gets smarter over time. |
| **Multi-Channel I/O** | Telegram bot, Slack bot, IMAP email polling, webhooks — with per-user notification routing and quiet hours suppression. |
| **Budget Governance** | Monthly spending limits ($50 default), per-model cost tracking, alerts at 80%/95% thresholds, automatic month reset. |
| **Tool Access** | Web search (SearXNG/Brave/DuckDuckGo), sandboxed code execution, browser automation (Playwright), file ops, document/voice/image ingestion. |
| **MAGMA Multi-Graph Query** | Fan-out queries across 4 weighted dimensions: semantic (0.40), entity (0.25), temporal (0.20), causal (0.15) with fusion scoring. |
| **Symbolic Inference** | Transitive reasoning (A→B + B→C = A→C), aggregate boosting, temporal supersession, inconsistency detection. |
| **Enterprise Reliability** | Event-driven bus with 60+ typed topics, circuit breakers, loop detection, dead letter queues, idempotent processing, backpressure, health monitoring, checkpoint-based crash recovery. |

### REST API (20+ endpoints)

Key endpoints: `/api/submit` (goal submission), `/api/goals` (goal listing/DAG visualization/pause/resume/cancel), `/api/chat` (conversational interface with intent detection), `/api/claims` (belief ledger query), `/api/ingest/*` (document/voice/image ingestion), `/api/setup` (first-run LLM provider configuration), `/ws` (real-time event streaming).

### Service Count: 20+

Orchestration (Planner, Dispatcher, Executor, VerificationGate, Recovery, Checkpoint, HIL), Knowledge Pipeline (Researcher, Validator, FactChecker), Query & Analysis (Query, Chat, Analyst), Content Generation (Coder, Writer), Operational (Doctor, Digest, Monitor, Security, Ingestor, Memory).

---

## Competitive Landscape

### Open-Source Multi-Agent Frameworks

| Competitor | Stars | Description | Key Difference vs QE OS |
|---|---|---|---|
| **AutoGen / AG2** (Microsoft) | ~55K | Conversation-based multi-agent collaboration | No DAG decomposition, no verification gates, no belief ledger. Merging with Semantic Kernel. |
| **CrewAI** | ~25K | Role-based agent teams (Researcher, Writer, etc.) | Role metaphor vs. QE's structured planner/dispatcher/executor. No knowledge accumulation or model escalation. |
| **LangGraph** (LangChain) | ~25K | Low-level graph framework for stateful agents | Building blocks, not a complete system. No belief ledger, verification, or budget tracking. Excellent durable execution. |
| **MS Agent Framework** (Semantic Kernel) | ~27K | Enterprise planner + multi-agent patterns | Closest analog — has plan decomposition. .NET-first. No belief ledger or verification gates. |
| **Strands Agents** (AWS) | Growing | Model-driven agent SDK, LLM handles planning | Model-driven (LLM plans) vs. QE's explicit planner. Deep AWS integration. Supports edge deployment. |
| **Swarms** | Growing | Large-scale swarm orchestration patterns | Focus on swarm patterns (many agents). Less structured quality control. |

### Commercial / Low-Code Platforms

| Competitor | Type | Pricing | Key Difference vs QE OS |
|---|---|---|---|
| **n8n** | Open source + cloud | Free self-hosted / €24-800/mo cloud | General workflow automation + AI. 500+ integrations. Visual builder. Not AI-first. |
| **Dify** | Open source + cloud | Free self-hosted / paid cloud | Visual no-code LLM app builder. Strong RAG engine. No DAG decomposition or verification gates. |
| **Dust** | Commercial SaaS | €29/user/mo | Enterprise no-code agents. SOC2 certified. No structured DAG or belief ledger. |
| **Relevance AI** | Commercial SaaS | From $19/mo | 9,000+ integrations, no-code. Multi-agent but no DAG decomposition. |
| **Beam AI** | Commercial SaaS | From $299/mo | Agents that learn from interactions. Different paradigm — adaptive vs. explicit planning. |

### Enterprise Orchestration

| Competitor | Type | Pricing | Key Difference vs QE OS |
|---|---|---|---|
| **Orkes Conductor** | Commercial (Netflix origin) | $695+/mo | Battle-tested at Netflix scale. RBAC/SSO/audit. General workflow + AI, not AI-first. |
| **Temporal** | Open source + cloud | Consumption-based | Durable execution infrastructure. Language-agnostic. No AI-specific features — could run *under* QE OS. |
| **Amazon Bedrock Agents** | AWS managed service | Pay-per-use | Fully managed. Supervisor/sub-agent model. AWS lock-in. No belief ledger or verification gates. |

### Visual AI Builders

| Competitor | Stars | Key Difference vs QE OS |
|---|---|---|
| **Langflow** | Growing | Visual drag-and-drop. Every workflow deployable as API/MCP server. No DAG decomposition. |
| **Flowise** | ~41K | Node.js-based. Three-tier builder (beginner to advanced). No structured orchestration. |

---

## Feature Comparison Matrix

| Feature | QE OS | CrewAI | LangGraph | AutoGen | MS Agent Framework | Dify | Orkes | n8n |
|---|---|---|---|---|---|---|---|---|
| DAG Task Decomposition | **Yes (core)** | No | Manual | No | Yes (planner) | No | No | No |
| Verification Gates | **Yes** | No | No | No | No | No | No | No |
| Belief Ledger / Confidence | **Yes** | No | No | No | No | No | No | No |
| Model Escalation | **Yes** | No | No | No | No | No | No | No |
| Multi-LLM via litellm | **Yes** | Yes | Yes | Yes | Yes | Yes | No | Yes |
| Event-Driven Bus | **Yes** | No | No | No | No | No | Yes | No |
| Human-in-the-Loop | **Yes** | Limited | Yes | Yes | Yes | No | Yes | Yes |
| Budget/Cost Tracking | **Yes** | No | No | No | No | No | No | No |
| Multi-Channel Input | **Yes** | No | No | No | No | No | No | Yes (integrations) |
| Visual Builder | No | No | No | AutoGen Studio | No | **Yes** | **Yes** | **Yes** |
| Enterprise (RBAC/SSO) | No | Enterprise plan | Enterprise plan | No | Yes | No | **Yes** | No |

---

## Competitive Positioning

### QE OS's Unique Strengths (No competitor has all of these)

1. **Belief Ledger with confidence scoring & decay** — genuinely novel. No competitor natively tracks claims with confidence that decays over time.
2. **Structured DAG decomposition + verification gates** — explicit planner creates typed subtask graphs, outputs are verified before acceptance. Only MS Agent Framework has a comparable planner, but none have verification gates.
3. **Automated model escalation with recovery** — local → fast → balanced → powerful, with failure classification and strategy execution. Unique.
4. **Integrated budget governance** — built-in cost tracking with monthly limits and alerts. Rare among orchestration frameworks.
5. **Multi-channel I/O as first-class** — Telegram/Slack/Email/Webhook natively, not as plugins.

### Where QE OS is Weaker

1. **Community** — AutoGen has 55K stars, CrewAI 25K. QE OS would need to build from scratch.
2. **No visual/no-code interface** — Dify, n8n, Flowise, Langflow all have visual builders.
3. **Enterprise governance gaps** — No RBAC, SSO, or compliance certifications (Orkes, Dust have these).
4. **Scale questions** — SQLite-backed vs. Orkes handling billions of workflows at Netflix scale.
5. **Ecosystem** — n8n has 500+ integrations, Relevance AI has 9,000+. QE OS has ~6 tools.

### Bottom Line

QE OS occupies a unique niche: a **self-hosted, AI-native orchestration system with knowledge accumulation and self-healing**. Its closest competitors are CrewAI and LangGraph in the open-source space, and Orkes Conductor in the enterprise space — but none combine DAG decomposition, verification gates, belief ledger, and model escalation into a single system. The main competitive risk is that the large-community frameworks (CrewAI, LangGraph, AutoGen) add these features incrementally, and that enterprise buyers choose managed services (Bedrock, Orkes) over self-hosted.

---

## Sources

- [CrewAI](https://crewai.com/) — [GitHub](https://github.com/crewAIInc/crewAI) — [Pricing](https://crewai.com/pricing)
- [LangGraph](https://www.langchain.com/langgraph) — [GitHub](https://github.com/langchain-ai/langgraph) — [Pricing](https://www.langchain.com/pricing-langgraph-platform)
- [Microsoft AutoGen](https://github.com/microsoft/autogen) — [AG2 Fork](https://github.com/ag2ai/ag2)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Strands Agents (AWS)](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)
- [Swarms](https://github.com/kyegomez/swarms)
- [n8n](https://n8n.io/) — [Pricing](https://n8n.io/pricing/)
- [Dify](https://dify.ai/) — [GitHub](https://github.com/langgenius/dify)
- [Dust](https://dust.tt/) — [Pricing](https://dust.tt/home/pricing)
- [Relevance AI](https://relevanceai.com/) — [Pricing](https://relevanceai.com/pricing)
- [Beam AI](https://beam.ai/)
- [Orkes Conductor](https://orkes.io/) — [Pricing](https://orkes.io/pricing)
- [Temporal](https://temporal.io/) — [Pricing](https://temporal.io/pricing)
- [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/)
- [Langflow](https://github.com/langflow-ai/langflow)
- [Flowise](https://flowiseai.com/) — [GitHub](https://github.com/FlowiseAI/Flowise)
- [AIMultiple: Top Agentic AI Frameworks 2026](https://aimultiple.com/agentic-frameworks)
- [Deloitte: AI Agent Orchestration Predictions 2026](https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/ai-agent-orchestration.html)
