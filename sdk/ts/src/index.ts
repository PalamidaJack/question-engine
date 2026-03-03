/**
 * Question Engine TypeScript SDK
 *
 * A TypeScript client for Question Engine OS HTTP APIs.
 *
 * @example
 * ```typescript
 * import { QEClient } from 'qe-sdk-ts';
 *
 * const client = new QEClient({
 *   baseUrl: 'http://localhost:8000',
 *   apiKey: 'your-api-key'
 * });
 *
 * const progress = await client.getGoalProgress('goal-123');
 * console.log(`Progress: ${progress.pct_complete}%`);
 * ```
 */

export { QEClient, default } from "./client.js";
export type {
  // Goals
  Goal,
  GoalStatus,
  GoalProgress,
  GoalResult,
  GoalSubmitResponse,
  Subtask,
  GoalDAG,
  DAGNode,
  DAGEdge,
  DAGStatus,
  // Memory (legacy)
  Memory,
  MemoryList,
  Project,
  ProjectList,
  Preference,
  // Memory Operations
  MemorySearchParams,
  MemorySearchResult,
  MemoryTiersStatus,
  ProceduralResult,
  ConsolidationHistory,
  MemoryExport,
  MemoryImportResult,
  // A2A
  AgentCard,
  A2ATaskCreateResponse,
  A2ATask,
  A2AMessageList,
  // Guardrails
  GuardrailsStatus,
  GuardrailsConfigUpdate,
  GuardrailsConfigureResponse,
  // Health & Status
  HealthResponse,
  ReadinessResponse,
  StatusResponse,
  // Feature Flags
  FeatureFlag,
  FeatureFlagList,
  // Client
  QEClientOptions,
  RequestOptions,
} from "./types.js";
