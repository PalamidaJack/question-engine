"""Bootstrap default tool registry with all built-in tools."""

from __future__ import annotations

import logging

from qe.runtime.tool_gate import SecurityPolicy, ToolGate
from qe.runtime.tools import ToolRegistry
from qe.tools import (
    browser_navigate,
    browser_spec,
    code_execute,
    code_execute_spec,
    file_read,
    file_read_spec,
    file_write,
    file_write_spec,
    web_fetch,
    web_fetch_spec,
    web_search,
    web_search_spec,
)

log = logging.getLogger(__name__)

_BUILTIN_TOOLS = [
    (web_search_spec, web_search),
    (web_fetch_spec, web_fetch),
    (file_read_spec, file_read),
    (file_write_spec, file_write),
    (code_execute_spec, code_execute),
    (browser_spec, browser_navigate),
]


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry and register all built-in tools."""
    registry = ToolRegistry()
    for spec, handler in _BUILTIN_TOOLS:
        registry.register(spec, handler)
    log.info(
        "tool_bootstrap.registered count=%d tools=%s",
        len(_BUILTIN_TOOLS),
        [s.name for s, _ in _BUILTIN_TOOLS],
    )
    return registry


def create_default_gate(
    policies: list[SecurityPolicy] | None = None,
) -> ToolGate:
    """Create a ToolGate with optional security policies."""
    return ToolGate(policies=policies)
