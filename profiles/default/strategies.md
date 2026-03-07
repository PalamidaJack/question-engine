# Strategies

## Decision Framework

### When to use which tool
- **Simple factual question** -> query_beliefs first, then web_search if insufficient
- **Complex research question** -> deep_research (5-iteration cognitive loop)
- **Multi-faceted question** -> swarm_research or plan_and_execute
- **Critical analysis request** -> reason_about for epistemic/dialectic reasoning
- **Knowledge synthesis** -> crystallize_insights after research completes
- **Background learning** -> consolidate_knowledge to detect patterns and promote beliefs

### Tool Chaining Patterns
- Research -> Crystallize -> Consolidate (standard knowledge pipeline)
- Query beliefs -> Deep research (only if existing knowledge insufficient)
- Swarm research -> Reason about (get multiple perspectives, then analyze critically)
- Plan and execute -> Synthesize (decompose complex goals, combine results)

### Model Selection Guidance
- Quick lookups, structured extraction -> fast model (low latency)
- Complex reasoning, research -> balanced model
- Critical decisions, nuanced analysis -> powerful model
- When unsure -> let the router decide based on task classification

### When to Sandbox
- Code from unvetted sources -> run_isolated
- New/untested MCP servers -> run_isolated
- Arbitrary user code execution -> run_isolated
- Trusted playbooks, own tools -> direct execution
