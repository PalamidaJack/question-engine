/**
 * HTTP client for Question Engine OS API.
 */

import type {
  A2AMessageList,
  A2ATask,
  A2ATaskCreateResponse,
  AgentCard,
  ConsolidationHistory,
  FeatureFlag,
  FeatureFlagList,
  GoalDAG,
  GoalProgress,
  GoalResult,
  GoalSubmitResponse,
  GuardrailsConfigUpdate,
  GuardrailsConfigureResponse,
  GuardrailsStatus,
  HealthResponse,
  MemoryExport,
  MemoryImportResult,
  MemoryList,
  MemorySearchParams,
  MemorySearchResult,
  MemoryTiersStatus,
  Preference,
  ProceduralResult,
  ProjectList,
  QEClientOptions,
  ReadinessResponse,
  RequestOptions,
  StatusResponse,
} from "./types.js";

export class QEClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number = 30000;

  constructor(options: QEClientOptions) {
    try {
      new URL(options.baseUrl);
    } catch {
      throw new Error(
        `Invalid baseUrl: '${options.baseUrl}'. Must be a valid URL (e.g., 'http://localhost:8000')`
      );
    }
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...options?.headers,
    };

    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      options?.timeout ?? this.timeout
    );

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (response.status === 204) {
        return null as unknown as T;
      }

      if (!response.ok) {
        const contentType = response.headers.get("content-type") || "";
        let error: string;
        if (contentType.includes("application/json")) {
          try {
            const json = await response.json();
            error = JSON.stringify(json);
          } catch {
            error = await response.text();
          }
        } else {
          error = await response.text();
        }
        throw new Error(`QE API error ${response.status}: ${error}`);
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        return (await response.json()) as T;
      }
      return null as unknown as T;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        const timeoutMs = options?.timeout ?? this.timeout;
        throw new Error(`QE API request timed out after ${timeoutMs}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ── Goals ────────────────────────────────────────────────────────────────

  async submitGoal(description: string): Promise<GoalSubmitResponse> {
    return this.request<GoalSubmitResponse>("POST", "/api/goals", {
      description,
    });
  }

  async getGoalProgress(goalId: string): Promise<GoalProgress> {
    return this.request<GoalProgress>("GET", `/api/goals/${goalId}/progress`);
  }

  async getGoalDAG(goalId: string): Promise<GoalDAG> {
    return this.request<GoalDAG>("GET", `/api/goals/${goalId}/dag`);
  }

  async getGoalResult(goalId: string): Promise<GoalResult> {
    return this.request<GoalResult>("GET", `/api/goals/${goalId}/result`);
  }

  async cancelGoal(goalId: string): Promise<{ status: string }> {
    return this.request<{ status: string }>(
      "POST",
      `/api/goals/${goalId}/cancel`
    );
  }

  // ── Memory (legacy) ──────────────────────────────────────────────────────

  async listMemories(): Promise<MemoryList> {
    return this.request<MemoryList>("GET", "/api/memory");
  }

  async deleteMemory(memoryId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>(
      "DELETE",
      `/api/memory/${memoryId}`
    );
  }

  async listProjects(): Promise<ProjectList> {
    return this.request<ProjectList>("GET", "/api/memory/projects");
  }

  async createProject(
    name: string,
    description?: string
  ): Promise<{ name: string; description?: string }> {
    return this.request("POST", "/api/memory/projects", {
      name,
      description,
    });
  }

  async setPreference(key: string, value: string): Promise<Preference> {
    return this.request<Preference>("POST", "/api/memory/preferences", {
      key,
      value,
    });
  }

  // ── Memory Operations ────────────────────────────────────────────────────

  async searchMemory(params: MemorySearchParams): Promise<MemorySearchResult> {
    const qs = new URLSearchParams();
    qs.set("query", params.query);
    if (params.tiers) qs.set("tiers", params.tiers);
    if (params.top_k !== undefined) qs.set("top_k", String(params.top_k));
    if (params.goal_id) qs.set("goal_id", params.goal_id);
    if (params.min_confidence !== undefined)
      qs.set("min_confidence", String(params.min_confidence));
    return this.request<MemorySearchResult>(
      "GET",
      `/api/memory/search?${qs.toString()}`
    );
  }

  async getMemoryTiers(): Promise<MemoryTiersStatus> {
    return this.request<MemoryTiersStatus>("GET", "/api/memory/tiers");
  }

  async getProceduralMemory(
    domain: string = "general",
    topK: number = 5
  ): Promise<ProceduralResult> {
    return this.request<ProceduralResult>(
      "GET",
      `/api/memory/procedural?domain=${encodeURIComponent(domain)}&top_k=${topK}`
    );
  }

  async getWorkingMemory(goalId: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      "GET",
      `/api/memory/working/${goalId}`
    );
  }

  async getContextDrift(goalId: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      "GET",
      `/api/memory/context-curator/${goalId}`
    );
  }

  async getConsolidationHistory(
    limit: number = 20
  ): Promise<ConsolidationHistory> {
    return this.request<ConsolidationHistory>(
      "GET",
      `/api/memory/consolidation/history?limit=${limit}`
    );
  }

  async exportMemory(): Promise<MemoryExport> {
    return this.request<MemoryExport>("POST", "/api/memory/export");
  }

  async importMemory(data: MemoryExport): Promise<MemoryImportResult> {
    return this.request<MemoryImportResult>("POST", "/api/memory/import", data);
  }

  // ── A2A (Agent-to-Agent) ─────────────────────────────────────────────────

  async getAgentCard(): Promise<AgentCard> {
    return this.request<AgentCard>("GET", "/.well-known/agent.json");
  }

  async createA2ATask(
    payload: Record<string, unknown>
  ): Promise<A2ATaskCreateResponse> {
    return this.request<A2ATaskCreateResponse>(
      "POST",
      "/api/a2a/tasks",
      payload
    );
  }

  async getA2ATask(taskId: string): Promise<A2ATask> {
    return this.request<A2ATask>("GET", `/api/a2a/tasks/${taskId}`);
  }

  async postA2AMessage(
    taskId: string,
    sender: string,
    content: Record<string, unknown>
  ): Promise<{ status: string }> {
    return this.request<{ status: string }>(
      "POST",
      `/api/a2a/tasks/${taskId}/messages`,
      { sender, content }
    );
  }

  async getA2AMessages(taskId: string): Promise<A2AMessageList> {
    return this.request<A2AMessageList>(
      "GET",
      `/api/a2a/tasks/${taskId}/messages`
    );
  }

  // ── Guardrails ───────────────────────────────────────────────────────────

  async getGuardrailsStatus(): Promise<GuardrailsStatus> {
    return this.request<GuardrailsStatus>("GET", "/api/guardrails/status");
  }

  async configureGuardrails(
    config: GuardrailsConfigUpdate
  ): Promise<GuardrailsConfigureResponse> {
    return this.request<GuardrailsConfigureResponse>(
      "POST",
      "/api/guardrails/configure",
      config
    );
  }

  // ── Health & Status ──────────────────────────────────────────────────────

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/api/health");
  }

  async getReadiness(): Promise<ReadinessResponse> {
    return this.request<ReadinessResponse>("GET", "/api/health/ready");
  }

  async getLiveHealth(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("GET", "/api/health/live");
  }

  async getStatus(): Promise<StatusResponse> {
    return this.request<StatusResponse>("GET", "/api/status");
  }

  // ── Feature Flags ────────────────────────────────────────────────────────

  async listFlags(): Promise<FeatureFlagList> {
    return this.request<FeatureFlagList>("GET", "/api/flags");
  }

  async getFlag(flagName: string): Promise<FeatureFlag> {
    return this.request<FeatureFlag>("GET", `/api/flags/${flagName}`);
  }

  async enableFlag(flagName: string): Promise<FeatureFlag> {
    return this.request<FeatureFlag>(
      "POST",
      `/api/flags/${flagName}/enable`
    );
  }

  async disableFlag(flagName: string): Promise<FeatureFlag> {
    return this.request<FeatureFlag>(
      "POST",
      `/api/flags/${flagName}/disable`
    );
  }
}

export default QEClient;
