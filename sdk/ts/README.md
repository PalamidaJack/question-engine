# Question Engine TypeScript SDK

A TypeScript client library for Question Engine OS HTTP APIs.

## Installation

```bash
npm install qe-sdk-ts
```

## Usage

```typescript
import { QEClient } from 'qe-sdk-ts';

const client = new QEClient({
  baseUrl: 'http://localhost:8000',
  apiKey: process.env.QE_API_KEY,
});

// Submit a goal
const goal = await client.submitGoal('Analyze lithium-ion battery market');
console.log(`Goal ${goal.goal_id}: ${goal.status}`);

// Track progress
const progress = await client.getGoalProgress(goal.goal_id);
console.log(`Progress: ${progress.pct_complete}%`);
```

## API

### Goals
- `client.submitGoal(description)` - Submit a new goal
- `client.getGoalProgress(goalId)` - Get progress for a goal
- `client.getGoalDAG(goalId)` - Get task DAG for a goal
- `client.getGoalResult(goalId)` - Get final result of a goal
- `client.cancelGoal(goalId)` - Cancel a running goal

### Memory (Legacy)
- `client.listMemories()` - List all memories
- `client.deleteMemory(memoryId)` - Delete a memory
- `client.listProjects()` - List projects
- `client.createProject(name, description?)` - Create a project
- `client.setPreference(key, value)` - Set a preference

### Memory Operations
- `client.searchMemory(params)` - Cross-tier search (episodic, belief, procedural)
- `client.getMemoryTiers()` - Status of all four memory tiers
- `client.getProceduralMemory(domain?, topK?)` - Get best templates and sequences
- `client.getWorkingMemory(goalId)` - Working memory context for a goal
- `client.getContextDrift(goalId)` - Context curator drift report
- `client.getConsolidationHistory(limit?)` - Knowledge loop consolidation history
- `client.exportMemory()` - Export all memory tiers to JSON
- `client.importMemory(data)` - Import memory from exported JSON

### A2A (Agent-to-Agent)
- `client.getAgentCard()` - Get the agent discovery card
- `client.createA2ATask(payload)` - Create an A2A task
- `client.getA2ATask(taskId)` - Get A2A task status
- `client.postA2AMessage(taskId, sender, content)` - Post a message to a task
- `client.getA2AMessages(taskId)` - Get message history for a task

### Guardrails
- `client.getGuardrailsStatus()` - Get guardrails pipeline status and rules
- `client.configureGuardrails(config)` - Update guardrails configuration

### Health & Status
- `client.getHealth()` - Basic health check
- `client.getReadiness()` - Readiness probe (startup phases)
- `client.getLiveHealth()` - Live health from Doctor service
- `client.getStatus()` - Full engine status (services, budget, circuit breakers)

### Feature Flags
- `client.listFlags()` - List all feature flags
- `client.getFlag(name)` - Get a single flag
- `client.enableFlag(name)` - Enable a flag
- `client.disableFlag(name)` - Disable a flag

## Development

```bash
npm install
npm run build
npm test
```

## Requirements

- Node.js 18+
- Question Engine OS server running
