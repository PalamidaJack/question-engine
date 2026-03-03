# Question Engine OS Components

## UI Architecture
Primary UI is implemented in [`src/qe/api/static/index.html`](src/qe/api/static/index.html) as a React app with reusable primitives and tabbed domain views.

## Design Primitives
- `Icon`: SVG icon renderer.
- `Badge`: semantic chip for labels/status.
- `StatusDot`: alive/dead status indicator.
- `Stat`: KPI metric card primitive.
- `EmptyState`: empty list/panel presentation.
- `TopicBadge`: event-topic color mapping.
- `ConfidenceBar`: confidence visualization.

## Domain Components
- `InlineClaimCard`: compact claim display.
- `PipelineProgress`: staged processing visualization for tracked envelopes.
- `ChatMessage`: role-aware message card with claims, confidence, entities, suggestions, and pipeline progress.
- `PipelineFunnel`: event-flow funnel metrics.
- `BudgetPanel`: budget usage and per-model spend.
- `TopicActivity`: event distribution mini-chart.
- `SystemMap`: topology/relationship graph (vis-network).
- `SetupWizard`: first-run provider, tier, channel, and key onboarding.

## Global Runtime UI Systems
- **Navigation shell**: sidebar + service status + tabbed main content.
- **WebSocket bus bridge**: live event stream consumption.
- **Chat WS session**: interactive streaming/progress/interjection loop.
- **Command Palette**:
  - Open with `⌘K / Ctrl+K`
  - Supports navigation and quick actions.
- **Keyboard Shortcut Modal**:
  - Open with `⌘/Ctrl + /`
  - Lists all global shortcuts.

## Screen/Panel Inventory
- `Dashboard`: KPIs + pipeline + budget + recent system activity.
- `System Map`: service topology and relationships.
- `Chat`: live assistant interaction with streamed state.
- `Ask`: query belief ledger.
- `Submit`: submit new observation.
- `Claims`: claim list/search/filter/retract.
- `Entities`: entity inspection.
- `HIL Queue`: approve/reject pending decisions.
- `Event Bus`: event stream with filters.
- `Scout`: innovation proposals, diffs, review actions.
- `Settings`: runtime config, provider/channel reconfigure, flags, access mode, import/export/reset, preferences.

## Settings Panel Subcomponents
- Provider/channel reconfigure accordion.
- Budget controls (`monthly_limit_usd`, `alert_at_pct`).
- Runtime controls (`hil_timeout_seconds`, `log_level`).
- Agent access controls (`strict|balanced|full`).
- Feature flags runtime controls.
- Import/export/reset controls.
- Keyboard shortcuts launcher.
- Event filter and claim display preferences.
- Service control (circuit reset).

## Component State Coverage
The UI handles:
- Loading (boot, per-panel fetch, flags loading)
- Empty (claims/entities/events/flags/services states)
- Populated (normal operation)
- Error (settings save error, flag fetch error, API failures)
- Disabled (save buttons, setup controls, unavailable actions)
- Saving/Pending (setup/settings/reconfigure save and chat typing/progress)
- Stale/Refreshable (manual refresh for flags/scout/status paths)
- Read-only equivalents (non-editable status displays)

## Action Surface (User-triggered)
- Submit observations, ask questions, chat.
- Retract claims.
- Approve/reject HIL and scout proposals.
- Toggle feature flags.
- Save runtime settings.
- Reconfigure providers/tiers/channels.
- Export/import settings bundle.
- Section reset and factory reset (typed confirmation).
- Navigate and execute commands via command palette.
