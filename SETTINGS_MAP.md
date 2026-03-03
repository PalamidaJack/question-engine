# Question Engine OS Settings Map

This map includes settings currently discoverable in code/config and how they are exposed in UI.

## Legend
- Storage: `config.toml`, `.env`, runtime memory, or localStorage.
- Update Mode: `live`, `restart`, or `client-only`.

## Runtime Settings (`/api/settings`)

| Setting | UI Control | Default | Storage | Update Mode | Notes |
|---|---|---:|---|---|---|
| `budget.monthly_limit_usd` | Numeric input | `50.0` | `config.toml` | live | Applied to budget tracker at save time |
| `budget.alert_at_pct` | Range slider | `0.80` | `config.toml` | live | Applied to budget tracker at save time |
| `runtime.log_level` | Select | `INFO` | `config.toml` | live | Applied via runtime logger update |
| `runtime.hil_timeout_seconds` | Numeric input | `3600` | `config.toml` | live | Consumed by HIL workflows |
| `agent_access.mode` | Select + Full Access checkbox | `balanced` | `config.toml` | live | `strict`, `balanced`, `full` |
| `retrieval.fts_top_k` | Not yet surfaced in dedicated control | `20` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |
| `retrieval.semantic_top_k` | Not yet surfaced in dedicated control | `20` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |
| `retrieval.semantic_min_similarity` | Not yet surfaced in dedicated control | `0.3` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |
| `retrieval.fts_weight` | Not yet surfaced in dedicated control | `0.6` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |
| `retrieval.semantic_weight` | Not yet surfaced in dedicated control | `0.4` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |
| `retrieval.rrf_k` | Not yet surfaced in dedicated control | `60` | `config.toml` | live-on-read | Backend supports; UI pending dedicated field |

## Setup/Reconfigure (`/api/setup/*`)

| Setting | UI Control | Default | Storage | Update Mode | Notes |
|---|---|---:|---|---|---|
| Provider API keys (`*_API_KEY`) | Password inputs | n/a | `.env` | restart-dependent | Managed via setup/reconfigure |
| Provider API base for specific providers | implicit | n/a | `.env` | restart-dependent | Derived during setup save |
| Tier model mapping (`models.fast/balanced/powerful`) | Text inputs | provider defaults | `config.toml` | runtime on next resolution | Set via setup/reconfigure |
| Channel credentials (`TELEGRAM_*`, `SLACK_*`, `EMAIL_*`) | Inputs under channel cards | n/a | `.env` | restart | Channel adapters depend on startup init |

## Feature Flags (`/api/flags*`)

| Setting | UI Control | Default | Storage | Update Mode | Notes |
|---|---|---:|---|---|---|
| Runtime feature flags (`name.enabled`) | Toggle list | defined at runtime | in-memory flag store | live | Toggle via enable/disable endpoints |
| Rollout percentage / targeting | Read-only display (rollout %) | `100` unless defined | in-memory flag store | live | Full targeting editor not yet exposed |

## Local UI Preferences (`localStorage`)

| Setting | UI Control | Default | Storage | Update Mode | Notes |
|---|---|---:|---|---|---|
| `eventFilters[]` | Grouped toggle chips | predefined set | `localStorage` | client-only | Event Bus display filtering |
| `minConfidence` | Slider | `0` | `localStorage` | client-only | Claims display filter |
| `timeRange` | Select | `all` | `localStorage` | client-only | Claims display filter |
| `showSuperseded` | Toggle | `false` | `localStorage` | client-only | Affects claims query request |

## Config-Only Settings Found in `config.toml` (not fully surfaced yet)

| Section | Keys |
|---|---|
| `runtime` | `prefer_local_models` |
| `substrate` | `db_path`, `cold_storage_path` |
| `bus` | `type` (+ optional redis URL) |
| `ingestion` | rss/webhook placeholders |
| `scout` | `enabled`, intervals, thresholds, budgets, topics |
| `guardrails` | content/pii/cost/hallucination toggles and thresholds |
| `a2a` | enable/auth/name/description |
| `otel` | exporter/endpoint/service name |
| `harvest` | enable/polling/thresholds/model behavior/budget/concurrency/timeouts |

## Restart vs Live Summary
- **Live:** budget limits, log level, feature flags, agent access mode.
- **Likely restart-required:** channel credentials and some integration startup wiring.
- **Client-only:** display filters/preferences.

## Import/Export/Reset UI
- Export: settings + display preferences JSON.
- Import: merges JSON payload into current settings/forms.
- Section resets: budget/runtime/agent_access.
- Factory reset: typed confirmation (`RESET`) for local + form defaults.
