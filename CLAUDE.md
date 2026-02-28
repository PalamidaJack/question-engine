# Question Engine OS

## Project Overview

Multi-agent orchestration system with a bus-driven architecture. Services communicate via `MemoryBus` with typed topics. Goals are decomposed by a planner, dispatched to executors, and results are delivered back through notification channels.

## Tech Stack

- Python 3.14, FastAPI, Pydantic v2, aiosqlite, litellm
- Virtual env: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
- LLM provider: Kilo Code (OpenRouter-compatible) at `https://kilo.ai/api/openrouter`
- Models: `openai/anthropic/claude-sonnet-4` (balanced), `openai/google/gemini-2.0-flash` (fast)

## Key Directories

- `src/qe/api/app.py` — Main FastAPI app, lifespan, channel wiring, all endpoints
- `src/qe/channels/` — Channel adapters (telegram, slack, email, webhook) + notifications router
- `src/qe/services/` — Planner, Dispatcher, Executor, Validator, Researcher, Doctor, Chat
- `src/qe/bus/` — MemoryBus, event log, bus metrics
- `src/qe/substrate/` — Belief ledger (SQLite), cold storage, goal store, embeddings
- `src/qe/models/` — Pydantic models (Envelope, Claim, GoalState, Genome Blueprint)
- `tests/unit/` — Unit tests (~40 files)
- `tests/integration/` — Integration + E2E tests
- `config.toml` — Runtime config (model tiers, budget, logging)
- `.env` — API keys (gitignored)

## Running Tests & Linting

```bash
.venv/bin/pytest tests/ --timeout=60 -q    # 992 tests, all passing
.venv/bin/ruff check src/ tests/            # all clean
```

## Current State (2026-02-28)

Everything is committed and pushed to `origin/main`. 992 tests pass, ruff clean.

### Recently Completed
- Multi-agent orchestration (planner, dispatcher, executor)
- Kilo Code LLM provider integration with litellm
- Channel adapters (Telegram, Slack, Email, Webhook)
- Channel -> goal -> notification wiring in app.py
- `message_callback` + `_forward_message()` on ChannelAdapter base class
- Telegram/Slack adapters forward messages with command classification (goal/ask/status)
- Per-user notification routing via `GoalState.metadata["origin_user_id"]`
- Command routing: ask -> `queries.asked`, status -> `system.health.check`, default -> `channel.message_received`
- `_on_query_asked` handler (calls `answer_question()`, notifies user)
- `_on_health_check` handler (builds status summary, notifies user)
- 10 channel wiring unit tests in `tests/unit/test_channel_wiring.py`

### Remaining Work

1. **Webhook command routing** — `/api/webhooks/inbound` always publishes to `channel.message_received`. Should respect a `command` field in the payload to route `ask`/`status` commands properly.

2. **Email adapter callback** — `EmailAdapter` needs `message_callback` parameter like Telegram/Slack adapters got.

3. **Integration test for command routing** — Need a live-server test that sends a webhook with `command: "ask"` and verifies the answer notification.

4. **GoalStore metadata round-trip** — Verify `GoalState.metadata` persists through SQLite serialization in `GoalStore`. Could be a silent bug.

5. **Planner genome warning** — Startup logs: `WARNING: Skipping genome planner.toml: PlannerService does not extend BaseService`. The planner genome isn't loading through the supervisor.

Items 1-2 are quick fixes. Item 4 may be a silent bug worth checking first.

## Important Patterns

- Kilo Code requires `litellm.register_model()` for its models — see `_configure_kilocode()` in app.py
- `OPENAI_API_BASE` env var must NOT have `/v1` suffix (Kilo Code path is `/api/openrouter/chat/completions`)
- Setting `OPENAI_API_BASE` at module import time in tests pollutes other tests — always use scoped fixtures with cleanup
- SQLite float values need `pytest.approx()` for comparison
- Bus topics are defined in `src/qe/bus/protocol.py`
