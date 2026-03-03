/**
 * TypeScript type definitions for Question Engine OS API.
 */

// ── Goals ────────────────────────────────────────────────────────────────────

export interface Goal {
  goal_id: string;
  description: string;
  status: GoalStatus;
  progress: GoalProgress;
  subtasks: Subtask[];
}

export type GoalStatus = "pending" | "in_progress" | "completed" | "failed";

export type DAGStatus = GoalStatus | string;

export interface GoalProgress {
  total: number;
  completed: number;
  failed: number;
  pending: number;
  pct_complete: number;
}

export interface Subtask {
  subtask_id: string;
  description: string;
  task_type: string;
  status: string;
  latency_ms: number;
  cost_usd: number;
  model_used: string;
}

export interface GoalDAG {
  nodes: DAGNode[];
  edges: DAGEdge[];
}

export interface DAGNode {
  id: string;
  description: string;
  task_type: string;
  model_tier: string;
  status: DAGStatus;
}

export interface DAGEdge {
  source: string;
  target: string;
}

export interface GoalResult {
  goal_id: string;
  status: GoalStatus;
  result: unknown;
  completed_at?: string;
}

export interface GoalSubmitResponse {
  goal_id: string;
  status: string;
}

// ── Memory (legacy) ──────────────────────────────────────────────────────────

export interface Memory {
  memory_id: string;
  category: string;
  key: string;
  value: string;
  confidence: number;
  source: string;
}

export interface MemoryList {
  memories: Memory[];
}

export interface Project {
  name: string;
  description?: string;
}

export interface ProjectList {
  projects: Project[];
}

export interface Preference {
  memory_id: string;
  key: string;
  value: string;
}

// ── Memory Operations ────────────────────────────────────────────────────────

export interface MemorySearchParams {
  query: string;
  tiers?: string;
  top_k?: number;
  goal_id?: string;
  min_confidence?: number;
}

export interface MemorySearchResult {
  query: string;
  results: {
    episodic?: unknown[];
    beliefs?: unknown[];
    procedural?: {
      templates: unknown[];
      sequences: unknown[];
    };
  };
}

export interface MemoryTiersStatus {
  working: Record<string, unknown>;
  episodic: { hot_entries?: number; warm_entries?: number };
  beliefs: { claim_count?: number | null };
  procedural: { templates?: number; sequences?: number };
}

export interface ProceduralResult {
  templates: unknown[];
  sequences: unknown[];
}

export interface ConsolidationHistory {
  history: unknown[];
}

export interface MemoryExport {
  episodic: unknown[];
  claims: unknown[];
  procedural: {
    templates: unknown[];
    sequences: unknown[];
  };
}

export interface MemoryImportResult {
  episodes_imported: number;
  claims_imported: number;
  procedural_imported: number;
}

// ── A2A (Agent-to-Agent) ─────────────────────────────────────────────────────

export interface AgentCard {
  name: string;
  description: string;
  url: string;
  version: string;
  capabilities: {
    intents: string[];
    tools: string[];
  };
  skills: Array<{ name: string; description: string }>;
}

export interface A2ATaskCreateResponse {
  task_id: string;
  goal_id: string | null;
}

export interface A2ATask {
  task_id: string;
  goal_id: string | null;
}

export interface A2AMessageList {
  messages: Array<{
    sender: string;
    content: Record<string, unknown>;
  }>;
}

// ── Guardrails ───────────────────────────────────────────────────────────────

export interface GuardrailsStatus {
  configured: boolean;
  config: Record<string, unknown> | null;
  rules: string[];
}

export interface GuardrailsConfigUpdate {
  enabled?: boolean;
  content_filter_enabled?: boolean;
  pii_detection_enabled?: boolean;
  cost_guard_enabled?: boolean;
  cost_guard_threshold_usd?: number;
  hallucination_guard_enabled?: boolean;
}

export interface GuardrailsConfigureResponse {
  status: string;
  applied: Record<string, unknown>;
}

// ── Health & Status ──────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  timestamp?: string;
}

export interface ReadinessResponse {
  ready: boolean;
  phases?: Record<string, boolean>;
  [key: string]: unknown;
}

export interface StatusResponse {
  status: string;
  services?: Record<string, unknown>;
  budget?: Record<string, unknown>;
  circuit_breakers?: Record<string, unknown>;
  [key: string]: unknown;
}

// ── Feature Flags ────────────────────────────────────────────────────────────

export interface FeatureFlag {
  name: string;
  enabled: boolean;
  [key: string]: unknown;
}

export interface FeatureFlagList {
  flags: Record<string, boolean>;
  [key: string]: unknown;
}

// ── Client Options ───────────────────────────────────────────────────────────

export interface QEClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
}

export interface RequestOptions {
  timeout?: number;
  headers?: Record<string, string>;
}
