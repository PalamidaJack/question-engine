/**
 * Example: Basic usage of Question Engine TypeScript SDK
 *
 * Run with: npx tsx examples/basic.ts
 */

import { QEClient } from "../src/index.js";

async function main() {
  const client = new QEClient({
    baseUrl: process.env.QE_BASE_URL || "http://localhost:8000",
    apiKey: process.env.QE_API_KEY,
  });

  console.log("Question Engine SDK Example\n");

  // Health check
  try {
    const health = await client.getHealth();
    console.log(`Health: ${health.status}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`Health check failed: ${msg}`);
    console.log("Make sure the QE server is running.\n");
    return;
  }

  // Agent Card (A2A discovery)
  try {
    const card = await client.getAgentCard();
    console.log(`Agent: ${card.name} v${card.version}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`Agent card error: ${msg}`);
  }

  // Memory tiers
  try {
    const tiers = await client.getMemoryTiers();
    console.log(`\nMemory Tiers:`);
    console.log(`  Episodic: ${tiers.episodic.hot_entries ?? 0} hot entries`);
    console.log(`  Beliefs: ${tiers.beliefs.claim_count ?? 0} claims`);
    console.log(`  Procedural: ${tiers.procedural.templates ?? 0} templates`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\nMemory tiers error: ${msg}`);
  }

  // Feature flags
  try {
    const flags = await client.listFlags();
    console.log(`\nFeature Flags:`);
    for (const [name, enabled] of Object.entries(flags.flags)) {
      console.log(`  ${name}: ${enabled ? "ON" : "OFF"}`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\nFlags error: ${msg}`);
  }

  // Guardrails status
  try {
    const gr = await client.getGuardrailsStatus();
    console.log(`\nGuardrails: ${gr.configured ? "configured" : "not configured"}`);
    if (gr.rules.length > 0) {
      console.log(`  Rules: ${gr.rules.join(", ")}`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\nGuardrails error: ${msg}`);
  }

  // Submit a goal (if QE_GOAL env var is set)
  const goalDesc = process.env.QE_GOAL;
  if (goalDesc) {
    try {
      console.log(`\nSubmitting goal: "${goalDesc}"`);
      const goal = await client.submitGoal(goalDesc);
      console.log(`  Goal ID: ${goal.goal_id}`);
      console.log(`  Status: ${goal.status}`);

      // Poll progress
      const progress = await client.getGoalProgress(goal.goal_id);
      console.log(
        `  Progress: ${progress.completed}/${progress.total} (${progress.pct_complete}%)`
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  Goal error: ${msg}`);
    }
  }

  console.log("\nDone!");
}

main().catch(console.error);
