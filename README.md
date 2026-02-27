# Question Engine OS

[![CI](https://github.com/YOUR_USERNAME/question-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/question-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

An operating system for AI agents. Persistent daemon services, deterministic message routing, and a three-layer knowledge substrate.

## Quick Start (Phase 0)

```bash
# Install
git clone https://github.com/YOUR_USERNAME/question-engine.git
cd question-engine
pip install -e ".[dev]"

# Set API keys
cp .env.example .env
# edit .env with your keys

# Start the engine
qe start

# In another terminal: submit an observation
qe submit "SpaceX launched Starship on March 3rd and achieved full orbit"

# Check what was learned
qe claims list
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Scaffold | ✅ | Project structure, installs cleanly |
| Phase 2 — Models | ✅ | Pydantic models pass unit tests |
| Phase 3 — Bus | ✅ | In-memory message bus |
| Phase 4 — Substrate | ✅ | SQLite Belief Ledger + Cold Storage |
| Phase 5 — Runtime | ✅ | BaseService, AutoRouter, ContextManager |
| Phase 6 — Researcher | ✅ | First working service |
| Phase 7 — Kernel | ✅ | Supervisor, genome loading |
| Phase 8 — HIL | ✅ | Human-in-the-loop file queue |
| Phase 9 — CLI | ✅ | qe command-line interface |
| Phase 10 — E2E | ✅ | Submit → claim visible in 30s |

## Architecture

See `question-engine-os-spec.md` for the full architecture spec.

## Development

```bash
# Run tests
pytest tests/unit/ -v    # fast unit tests
pytest tests/integration/ -v  # integration tests (SQLite in /tmp)

# Run pre-commit checks
pre-commit run --all-files

# Check types
mypy src/qe
```

## Configuration

Copy `config.toml` and edit model preferences. Never put API keys in config files — use environment variables.
