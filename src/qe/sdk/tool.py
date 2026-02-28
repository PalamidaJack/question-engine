"""SDK decorator for registering custom tools."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from qe.runtime.tools import ToolRegistry, ToolSpec

log = logging.getLogger(__name__)

# Global tool collection for SDK-registered tools
_SDK_TOOLS: list[tuple[ToolSpec, Callable]] = []


def tool(
    name: str,
    requires_capability: str = "",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    description: str = "",
    category: str = "custom",
    timeout_seconds: int = 30,
) -> Callable:
    """Decorator to register a function as a QE tool.

    Usage::

        @tool(
            name="arxiv_search",
            requires_capability="web_search",
            description="Search arXiv for papers",
        )
        async def arxiv_search(query: str, max_results: int = 5):
            ...
    """

    def decorator(
        func: Callable[..., Coroutine],
    ) -> Callable[..., Coroutine]:
        spec = ToolSpec(
            name=name,
            description=description or func.__doc__ or "",
            requires_capability=requires_capability or None,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            category=category,
            timeout_seconds=timeout_seconds,
        )
        _SDK_TOOLS.append((spec, func))
        func._tool_spec = spec  # type: ignore[attr-defined]
        log.debug("sdk.tool_registered name=%s", name)
        return func

    return decorator


def register_sdk_tools(registry: ToolRegistry) -> int:
    """Register all SDK-decorated tools with a registry.

    Returns the number of tools registered.
    """
    count = 0
    for spec, handler in _SDK_TOOLS:
        registry.register(spec, handler)
        count += 1
    return count
