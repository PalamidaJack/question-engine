import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QEClient } from "../src/client.js";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json" }),
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  } as unknown as Response;
}

function textResponse(text: string, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "text/plain" }),
    json: () => Promise.reject(new Error("not json")),
    text: () => Promise.resolve(text),
  } as unknown as Response;
}

describe("QEClient", () => {
  let client: QEClient;

  beforeEach(() => {
    mockFetch.mockReset();
    client = new QEClient({ baseUrl: "http://localhost:8000" });
  });

  // ── Constructor ────────────────────────────────────────────────────────

  describe("constructor", () => {
    it("rejects invalid URL", () => {
      expect(() => new QEClient({ baseUrl: "not-a-url" })).toThrow(
        "Invalid baseUrl"
      );
    });

    it("strips trailing slash", () => {
      const c = new QEClient({ baseUrl: "http://localhost:8000/" });
      // Verify by making a request
      mockFetch.mockResolvedValueOnce(jsonResponse({ status: "ok" }));
      c.getHealth();
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/health",
        expect.anything()
      );
    });

    it("sets Authorization header when apiKey provided", async () => {
      const c = new QEClient({
        baseUrl: "http://localhost:8000",
        apiKey: "test-key",
      });
      mockFetch.mockResolvedValueOnce(jsonResponse({ status: "ok" }));
      await c.getHealth();
      const [, opts] = mockFetch.mock.calls[0];
      expect(opts.headers["Authorization"]).toBe("Bearer test-key");
    });
  });

  // ── Error handling ─────────────────────────────────────────────────────

  describe("error handling", () => {
    it("throws on non-ok JSON response", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ detail: "not found" }, 404)
      );
      await expect(client.getHealth()).rejects.toThrow("QE API error 404");
    });

    it("throws on non-ok text response", async () => {
      mockFetch.mockResolvedValueOnce(
        textResponse("server error", 500)
      );
      await expect(client.getHealth()).rejects.toThrow("QE API error 500");
    });

    it("throws on timeout", async () => {
      const c = new QEClient({
        baseUrl: "http://localhost:8000",
        timeout: 1,
      });
      mockFetch.mockImplementation(
        () =>
          new Promise((_, reject) => {
            const err = new Error("aborted");
            err.name = "AbortError";
            setTimeout(() => reject(err), 5);
          })
      );
      await expect(c.getHealth()).rejects.toThrow("timed out");
    });

    it("returns null for 204 status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
        headers: new Headers(),
      } as unknown as Response);
      const result = await client.cancelGoal("g1");
      expect(result).toBeNull();
    });
  });

  // ── Goals ──────────────────────────────────────────────────────────────

  describe("goals", () => {
    it("submitGoal sends POST with description", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ goal_id: "g_1", status: "pending" })
      );
      const res = await client.submitGoal("analyze lithium market");
      expect(res.goal_id).toBe("g_1");
      const [url, opts] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/api/goals");
      expect(opts.method).toBe("POST");
      expect(JSON.parse(opts.body)).toEqual({
        description: "analyze lithium market",
      });
    });

    it("getGoalProgress", async () => {
      const progress = {
        total: 4,
        completed: 2,
        failed: 0,
        pending: 2,
        pct_complete: 50.0,
      };
      mockFetch.mockResolvedValueOnce(jsonResponse(progress));
      const res = await client.getGoalProgress("g_1");
      expect(res.pct_complete).toBe(50.0);
      expect(mockFetch.mock.calls[0][0]).toContain("/api/goals/g_1/progress");
    });

    it("getGoalDAG", async () => {
      const dag = { nodes: [{ id: "s1" }], edges: [] };
      mockFetch.mockResolvedValueOnce(jsonResponse(dag));
      const res = await client.getGoalDAG("g_1");
      expect(res.nodes).toHaveLength(1);
    });

    it("getGoalResult", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          goal_id: "g_1",
          status: "completed",
          result: { summary: "done" },
        })
      );
      const res = await client.getGoalResult("g_1");
      expect(res.status).toBe("completed");
    });

    it("cancelGoal", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ status: "cancelled" }));
      const res = await client.cancelGoal("g_1");
      expect(res.status).toBe("cancelled");
      expect(mockFetch.mock.calls[0][1].method).toBe("POST");
    });
  });

  // ── Memory (legacy) ────────────────────────────────────────────────────

  describe("memory (legacy)", () => {
    it("listMemories", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ memories: [{ memory_id: "m1", key: "k" }] })
      );
      const res = await client.listMemories();
      expect(res.memories).toHaveLength(1);
    });

    it("deleteMemory", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ deleted: true }));
      const res = await client.deleteMemory("m1");
      expect(res.deleted).toBe(true);
      expect(mockFetch.mock.calls[0][1].method).toBe("DELETE");
    });

    it("listProjects", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ projects: [{ name: "p1" }] })
      );
      const res = await client.listProjects();
      expect(res.projects).toHaveLength(1);
    });

    it("createProject", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ name: "proj", description: "test" })
      );
      const res = await client.createProject("proj", "test");
      expect(res.name).toBe("proj");
    });

    it("setPreference", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ memory_id: "p1", key: "theme", value: "dark" })
      );
      const res = await client.setPreference("theme", "dark");
      expect(res.key).toBe("theme");
    });
  });

  // ── Memory Operations ──────────────────────────────────────────────────

  describe("memory operations", () => {
    it("searchMemory builds query string", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ query: "test", results: { episodic: [] } })
      );
      await client.searchMemory({
        query: "test",
        tiers: "episodic",
        top_k: 5,
      });
      const url: string = mockFetch.mock.calls[0][0];
      expect(url).toContain("query=test");
      expect(url).toContain("tiers=episodic");
      expect(url).toContain("top_k=5");
    });

    it("getMemoryTiers", async () => {
      const tiers = {
        working: {},
        episodic: { hot_entries: 10 },
        beliefs: { claim_count: 5 },
        procedural: { templates: 3 },
      };
      mockFetch.mockResolvedValueOnce(jsonResponse(tiers));
      const res = await client.getMemoryTiers();
      expect(res.episodic.hot_entries).toBe(10);
    });

    it("getProceduralMemory with defaults", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ templates: [], sequences: [] })
      );
      await client.getProceduralMemory();
      const url: string = mockFetch.mock.calls[0][0];
      expect(url).toContain("domain=general");
      expect(url).toContain("top_k=5");
    });

    it("getWorkingMemory", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ goal_id: "g1", slots: [] })
      );
      const res = await client.getWorkingMemory("g1");
      expect(res.goal_id).toBe("g1");
    });

    it("getContextDrift", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ drifted: false, similarity: 0.9 })
      );
      const res = await client.getContextDrift("g1");
      expect(res.drifted).toBe(false);
    });

    it("getConsolidationHistory", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ history: [] }));
      const res = await client.getConsolidationHistory(10);
      expect(res.history).toEqual([]);
      expect(mockFetch.mock.calls[0][0]).toContain("limit=10");
    });

    it("exportMemory", async () => {
      const data = { episodic: [], claims: [], procedural: { templates: [], sequences: [] } };
      mockFetch.mockResolvedValueOnce(jsonResponse(data));
      const res = await client.exportMemory();
      expect(res.episodic).toEqual([]);
      expect(mockFetch.mock.calls[0][1].method).toBe("POST");
    });

    it("importMemory", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ episodes_imported: 1, claims_imported: 2, procedural_imported: 0 })
      );
      const res = await client.importMemory({
        episodic: [{}],
        claims: [{}, {}],
        procedural: { templates: [], sequences: [] },
      });
      expect(res.episodes_imported).toBe(1);
      expect(res.claims_imported).toBe(2);
    });
  });

  // ── A2A ────────────────────────────────────────────────────────────────

  describe("A2A", () => {
    it("getAgentCard", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          name: "Question Engine",
          version: "1.0",
          capabilities: { intents: [], tools: [] },
          skills: [],
        })
      );
      const card = await client.getAgentCard();
      expect(card.name).toBe("Question Engine");
      expect(mockFetch.mock.calls[0][0]).toContain("/.well-known/agent.json");
    });

    it("createA2ATask", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ task_id: "a2a_abc", goal_id: "g_1" })
      );
      const res = await client.createA2ATask({ description: "test" });
      expect(res.task_id).toBe("a2a_abc");
      expect(res.goal_id).toBe("g_1");
    });

    it("getA2ATask", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ task_id: "a2a_abc", goal_id: "g_1" })
      );
      const res = await client.getA2ATask("a2a_abc");
      expect(res.task_id).toBe("a2a_abc");
    });

    it("postA2AMessage", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ status: "ok" }));
      const res = await client.postA2AMessage("a2a_abc", "ext", {
        text: "hello",
      });
      expect(res.status).toBe("ok");
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.sender).toBe("ext");
      expect(body.content.text).toBe("hello");
    });

    it("getA2AMessages", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ messages: [] }));
      const res = await client.getA2AMessages("a2a_abc");
      expect(res.messages).toEqual([]);
    });
  });

  // ── Guardrails ─────────────────────────────────────────────────────────

  describe("guardrails", () => {
    it("getGuardrailsStatus", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          configured: true,
          config: { enabled: true },
          rules: ["ContentFilter", "CostGuard"],
        })
      );
      const res = await client.getGuardrailsStatus();
      expect(res.configured).toBe(true);
      expect(res.rules).toContain("ContentFilter");
    });

    it("configureGuardrails", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", applied: { pii_detection_enabled: true } })
      );
      const res = await client.configureGuardrails({
        pii_detection_enabled: true,
      });
      expect(res.status).toBe("ok");
      expect(res.applied.pii_detection_enabled).toBe(true);
    });
  });

  // ── Health ─────────────────────────────────────────────────────────────

  describe("health", () => {
    it("getHealth", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", timestamp: "2026-03-03T00:00:00Z" })
      );
      const res = await client.getHealth();
      expect(res.status).toBe("ok");
    });

    it("getReadiness", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ready: true, phases: { substrate: true, bus: true } })
      );
      const res = await client.getReadiness();
      expect(res.ready).toBe(true);
    });

    it("getLiveHealth", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "healthy", services: {} })
      );
      const res = await client.getLiveHealth();
      expect(res.status).toBe("healthy");
    });

    it("getStatus", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "running", budget: { remaining: 10.0 } })
      );
      const res = await client.getStatus();
      expect(res.status).toBe("running");
    });
  });

  // ── Feature Flags ──────────────────────────────────────────────────────

  describe("feature flags", () => {
    it("listFlags", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ flags: { inquiry_mode: true, competitive_arena: false } })
      );
      const res = await client.listFlags();
      expect(res.flags.inquiry_mode).toBe(true);
    });

    it("getFlag", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ name: "inquiry_mode", enabled: true })
      );
      const res = await client.getFlag("inquiry_mode");
      expect(res.enabled).toBe(true);
    });

    it("enableFlag", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ name: "competitive_arena", enabled: true })
      );
      const res = await client.enableFlag("competitive_arena");
      expect(res.enabled).toBe(true);
      expect(mockFetch.mock.calls[0][1].method).toBe("POST");
    });

    it("disableFlag", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ name: "competitive_arena", enabled: false })
      );
      const res = await client.disableFlag("competitive_arena");
      expect(res.enabled).toBe(false);
    });
  });
});
