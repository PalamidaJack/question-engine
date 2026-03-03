# Question Engine OS UI/UX Decisions

## 1) Product Understanding (Code-Grounded)
The UI is a control plane over a real-time cognitive runtime:
- FastAPI app with multi-router API surface (`system`, `knowledge`, `goals_v2`, `chat`, `telemetry`, `scout`, `mass_intelligence`, `harvest`, `setup`, `guardrails`, `a2a`, `memory_ops`).
- Runtime includes supervisor, substrate, event bus, feature flags, episodic memory, prompt evolution, scout/harvest loops, and optional MCP bridge.
- Main user interaction loops:
  - Chat (WebSocket + progress stream)
  - Goal submission/execution (planner/dispatcher/executor)
  - Knowledge operations (submit/ask/claims/entities)
  - Operational observability (status/health/topology/metrics/profiling/bus)
  - Runtime control (flags, settings, reset-circuit, DLQ replay/purge)

## 2) Frontend Framework
Decision: Keep React 18 (existing) and evolve in-place.
Reasoning:
- Existing dashboard is already React-in-browser and feature-rich.
- Real-time state updates (WS, polling, progressive chat) are already implemented.
- Incremental build avoids migration risk and preserves shipped workflows.

## 3) Styling Approach
Decision: Keep inline style design system primitives in current file and extend consistently.
Reasoning:
- Current app uses strong tokenized palette + typographic system (`IBM Plex Sans/Mono`).
- Existing component primitives (`Badge`, `StatusDot`, `EmptyState`, etc.) are cohesive.
- Consistency and speed outweighed introducing a CSS-in-JS or utility framework midstream.

## 4) State Management
Decision: Component-local state + focused hooks; no external global state library.
Reasoning:
- Data is page-local and mostly independent (claims, entities, flags, settings).
- Realtime updates are event-driven and simple to model with hooks.
- Reduced dependency footprint and operational complexity.

## 5) Data Layer
Decision: Hybrid transport:
- REST for CRUD/config/queries
- WebSocket for event stream and interactive chat pipeline progress
- Polling for status snapshots where acceptable
Reasoning:
- Mirrors backend semantics and minimizes coupling.
- Keeps high-frequency streams off REST.

## 6) Persistence Strategy
Decision:
- Server-authoritative runtime settings via `/api/settings`
- Local UX prefs in `localStorage` (`qe_display_prefs`)
- Import/export bundle as JSON for portability
Reasoning:
- Server settings affect runtime behavior and need backend validation.
- Display preferences are client-only and should not perturb runtime.

## 7) Animation Strategy
Decision: Native CSS keyframes/transitions (`fadeIn`, `pulse`) + semantic motion only.
Reasoning:
- Existing UI uses light, purposeful motion.
- No need for animation library overhead.

## 8) Forms & Validation
Decision:
- Native controlled inputs + explicit validation messaging.
- Sensitive fields remain obscured where applicable.
- Add unsaved-changes guard (`beforeunload`) for settings workflows.
Reasoning:
- Existing forms are straightforward and readable.
- Avoid extra abstraction where domain validation is mostly backend-driven.

## 9) Security Access UX Model
Decision: 3-level `agent_access.mode` surfaced in settings:
- `strict`: web-only tools
- `balanced`: filesystem constrained to workspace
- `full`: full local FS + code execution
Reasoning:
- Provides safe default, practical middle tier, and explicit high-power mode.
- Clear user-facing explanation of blast radius.

## 10) New UX Systems Added
- Command Palette (`⌘K` / `Ctrl+K`) for navigation and common actions.
- Keyboard Shortcut reference modal (`⌘/Ctrl + /`).
- Feature Flag runtime panel (load + toggle via `/api/flags`).
- Import / Export / Reset controls for settings.
- Unsaved changes warning for settings/reconfigure edits.

## 11) Tradeoffs
- Chose in-place enhancement over full framework rewrite to preserve reliability and delivery speed.
- Some requested settings categories depend on backend APIs not yet exposed; UI maps available runtime/config controls now and documents gaps in `SETTINGS_MAP.md`.
