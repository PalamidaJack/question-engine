"""Workflow node type registry and execution handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

# Type alias for node executor functions
NodeExecutor = Callable[
    [dict[str, Any], dict[str, Any]],
    Awaitable[dict[str, Any]],
]

# Registry of node type -> executor function
_NODE_REGISTRY: dict[str, NodeExecutor] = {}


def register_node(node_type: str):
    """Decorator to register a node executor."""

    def wrapper(func: NodeExecutor):
        _NODE_REGISTRY[node_type] = func
        return func

    return wrapper


def get_executor(node_type: str) -> NodeExecutor | None:
    """Get the executor for a node type."""
    # Direct match
    if node_type in _NODE_REGISTRY:
        return _NODE_REGISTRY[node_type]
    # Check for prefix match (e.g., "tool/query_beliefs" -> "tool")
    prefix = (
        node_type.split("/")[0] if "/" in node_type else None
    )
    if prefix and prefix in _NODE_REGISTRY:
        return _NODE_REGISTRY[prefix]
    return None


# ── Built-in node executors ──


@register_node("entry")
async def exec_entry(
    config: dict, context: dict,
) -> dict:
    """Entry point -- passes through input."""
    return context.get("input", {})


@register_node("exit")
async def exec_exit(
    config: dict, context: dict,
) -> dict:
    """Exit point -- returns accumulated output."""
    return context.get("last_output", {})


@register_node("condition")
async def exec_condition(
    config: dict, context: dict,
) -> dict:
    """Evaluate a condition and return branch decision."""
    field = config.get("field", "")
    op = config.get("op", "==")
    value = config.get("value")

    # Navigate dotted field path in last_output
    data = context.get("last_output", {})
    for part in field.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list):
            try:
                data = (
                    len(data)
                    if part == "length"
                    else data[int(part)]
                )
            except (ValueError, IndexError):
                data = None
        else:
            data = None

    # Evaluate
    result = False
    if op == "==" and data == value:
        result = True
    elif op == "!=" and data != value:
        result = True
    elif op == ">" and data is not None and data > value:
        result = True
    elif op == ">=" and data is not None and data >= value:
        result = True
    elif op == "<" and data is not None and data < value:
        result = True
    elif op == "<=" and data is not None and data <= value:
        result = True
    elif (
        op == "contains"
        and value is not None
        and str(value) in str(data)
    ):
        result = True

    return {"branch": result}


@register_node("delay")
async def exec_delay(
    config: dict, context: dict,
) -> dict:
    """Wait for configured seconds."""
    seconds = config.get("seconds", 1)
    await asyncio.sleep(min(seconds, 300))  # Cap at 5 min
    return context.get("last_output", {})


@register_node("parallel")
async def exec_parallel(
    config: dict, context: dict,
) -> dict:
    """Marker node -- actual parallelism handled by executor."""
    return context.get("last_output", {})


@register_node("merge")
async def exec_merge(
    config: dict, context: dict,
) -> dict:
    """Merge parallel results."""
    parallel_results = context.get("parallel_results", [])
    strategy = config.get("strategy", "list")
    if strategy == "list":
        return {"results": parallel_results}
    elif strategy == "first":
        return (
            parallel_results[0] if parallel_results else {}
        )
    elif strategy == "concat":
        texts = [
            str(r.get("text", r))
            for r in parallel_results
            if r
        ]
        return {"text": "\n\n".join(texts)}
    return {"results": parallel_results}


@register_node("loop")
async def exec_loop(
    config: dict, context: dict,
) -> dict:
    """Marker node -- actual looping handled by executor."""
    return context.get("last_output", {})


@register_node("tool")
async def exec_tool(
    config: dict, context: dict,
) -> dict:
    """Execute a tool via the tool registry.

    Requires 'tool_executor' in context.
    """
    tool_executor = context.get("tool_executor")
    if not tool_executor:
        return {"error": "No tool executor available"}
    tool_name = config.get("tool_name", "")
    if not tool_name and "/" in context.get("node_type", ""):
        tool_name = context["node_type"].split("/", 1)[1]
    params = {**config}
    params.pop("tool_name", None)
    # Merge input from previous node
    last = context.get("last_output", {})
    for k, v in last.items():
        if k not in params:
            params[k] = v
    try:
        result = await tool_executor(tool_name, params)
        return (
            result
            if isinstance(result, dict)
            else {"result": result}
        )
    except Exception as e:
        return {"error": str(e)}


@register_node("llm/chat")
async def exec_llm_chat(
    config: dict, context: dict,
) -> dict:
    """Send to LLM for completion."""
    import litellm

    model = config.get("model", "gpt-4o-mini")
    system = config.get(
        "system_prompt", "You are a helpful assistant.",
    )
    last = context.get("last_output", {})
    user_msg = config.get("prompt", "") or str(last)

    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=config.get("temperature", 0.7),
        )
        text = resp.choices[0].message.content or ""
        return {
            "text": text,
            "model": model,
            "tokens": (
                resp.usage.total_tokens
                if resp.usage
                else 0
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@register_node("transform/extract")
async def exec_extract(
    config: dict, context: dict,
) -> dict:
    """Extract specific fields from the last output."""
    fields = config.get("fields", [])
    data = context.get("last_output", {})
    return {f: data.get(f) for f in fields if f in data}


@register_node("transform/filter")
async def exec_filter(
    config: dict, context: dict,
) -> dict:
    """Filter a list based on criteria."""
    data = context.get("last_output", {})
    items = data.get("results", data.get("items", []))
    field = config.get("field", "")
    op = config.get("op", "==")
    value = config.get("value")

    filtered = []
    for item in items:
        if isinstance(item, dict):
            item_val = item.get(field)
            if op == "==" and item_val == value:
                filtered.append(item)
            elif op == "!=" and item_val != value:
                filtered.append(item)
            elif (
                op == ">"
                and item_val is not None
                and item_val > value
            ):
                filtered.append(item)
            elif (
                op == "contains"
                and value
                and str(value) in str(item_val)
            ):
                filtered.append(item)
    return {"results": filtered, "count": len(filtered)}


@register_node("transform/format")
async def exec_format(
    config: dict, context: dict,
) -> dict:
    """Format output using a template string."""
    template = config.get("template", "{text}")
    data = context.get("last_output", {})
    try:
        text = template.format(**data)
    except (KeyError, IndexError):
        text = template
    return {"text": text}


@register_node("transform/aggregate")
async def exec_aggregate(
    config: dict, context: dict,
) -> dict:
    """Aggregate list items."""
    data = context.get("last_output", {})
    items = data.get("results", data.get("items", []))
    op = config.get("operation", "join")
    if op == "join":
        sep = config.get("separator", "\n")
        return {"text": sep.join(str(i) for i in items)}
    elif op == "count":
        return {"count": len(items)}
    elif op == "first":
        return items[0] if items else {}
    elif op == "last":
        return items[-1] if items else {}
    return {"items": items}


@register_node("human/approval")
async def exec_approval_gate(
    config: dict, context: dict,
) -> dict:
    """Pause for human approval.

    Sets workflow state to 'waiting_approval'.
    """
    return {
        "status": "waiting_approval",
        "message": config.get(
            "message", "Approval required to continue",
        ),
        "data": context.get("last_output", {}),
    }


@register_node("human/input")
async def exec_input_request(
    config: dict, context: dict,
) -> dict:
    """Request input from the user."""
    return {
        "status": "waiting_input",
        "question": config.get(
            "question", "Please provide input",
        ),
        "data": context.get("last_output", {}),
    }
