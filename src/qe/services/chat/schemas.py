"""Pydantic models for the chat interface."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ChatIntent(StrEnum):
    QUESTION = "question"
    OBSERVATION = "observation"
    COMMAND = "command"
    CONVERSATION = "conversation"


# ── Deprecated (kept for backward compat, no longer used by ChatService) ────


class IntentClassification(BaseModel):
    """LLM-generated intent classification for a user message."""

    intent: ChatIntent = Field(description="The detected intent of the user message")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the classification"
    )
    extracted_query: str = Field(
        description="The core query/observation/command extracted from the message"
    )
    reasoning: str = Field(description="Brief reasoning for the classification")


class CommandAction(StrEnum):
    RETRACT_CLAIM = "retract_claim"
    LIST_CLAIMS = "list_claims"
    LIST_ENTITIES = "list_entities"
    SHOW_ENTITY = "show_entity"
    SHOW_BUDGET = "show_budget"
    HELP = "help"
    UNKNOWN = "unknown"


class CommandParse(BaseModel):
    """Parsed command details."""

    action: CommandAction
    target: str | None = Field(
        default=None, description="Target entity/claim ID if applicable"
    )


# ── Active schemas ──────────────────────────────────────────────────────────


class ChatResponsePayload(BaseModel):
    """Structured response sent back to the chat client."""

    message_id: str
    reply_text: str
    intent: ChatIntent
    claims: list[dict] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)
    confidence: float | None = None
    reasoning: str | None = None
    pipeline_complete: bool = True
    tracking_envelope_id: str | None = None
    error: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    tool_calls_made: list[dict] = Field(default_factory=list)
    cognitive_process_used: bool = False


class ConversationalResponse(BaseModel):
    """LLM-generated conversational reply."""

    reply: str = Field(description="The conversational response to the user")
    suggestions: list[str] = Field(
        default_factory=list,
        description="2-3 short follow-up prompts the user might want to ask next",
    )


# ── Agent Permissions ──────────────────────────────────────────────────────


class PermissionScope(StrEnum):
    WEB_ACCESS = "web_access"
    FILE_SYSTEM = "file_system"
    CODE_EXECUTION = "code_execution"
    KNOWLEDGE_BASE = "knowledge_base"
    RESEARCH = "research"
    REASONING = "reasoning"
    SYSTEM_CONTROL = "system_control"
    COMMUNICATION = "communication"
    MCP_TOOLS = "mcp_tools"


# Scope → registry capability strings
_SCOPE_CAPABILITIES: dict[PermissionScope, set[str]] = {
    PermissionScope.WEB_ACCESS: {"web_search", "web_fetch"},
    PermissionScope.FILE_SYSTEM: {"file_read", "file_write"},
    PermissionScope.CODE_EXECUTION: {"code_execute"},
    PermissionScope.SYSTEM_CONTROL: {"browser_control"},
    PermissionScope.MCP_TOOLS: {"mcp"},
}

# Scope → built-in tool names (from _CHAT_TOOL_SCHEMAS)
_SCOPE_BUILTIN_TOOLS: dict[PermissionScope, set[str]] = {
    PermissionScope.KNOWLEDGE_BASE: {
        "query_beliefs", "list_entities", "get_entity_details",
        "submit_observation", "retract_claim", "get_budget_status",
    },
    PermissionScope.RESEARCH: {"deep_research", "swarm_research", "plan_and_execute"},
    PermissionScope.REASONING: {"reason_about", "crystallize_insights", "consolidate_knowledge"},
    PermissionScope.COMMUNICATION: {"delegate_to_agent"},
}

_DEFAULT_SCOPES: dict[PermissionScope, bool] = {
    PermissionScope.WEB_ACCESS: True,
    PermissionScope.FILE_SYSTEM: True,
    PermissionScope.CODE_EXECUTION: False,
    PermissionScope.KNOWLEDGE_BASE: True,
    PermissionScope.RESEARCH: True,
    PermissionScope.REASONING: True,
    PermissionScope.SYSTEM_CONTROL: False,
    PermissionScope.COMMUNICATION: False,
    PermissionScope.MCP_TOOLS: True,
}

_PRESETS: dict[str, dict[PermissionScope, bool]] = {
    "restricted": {s: s in {PermissionScope.WEB_ACCESS, PermissionScope.KNOWLEDGE_BASE} for s in PermissionScope},
    "standard": {s: s in {
        PermissionScope.WEB_ACCESS, PermissionScope.FILE_SYSTEM,
        PermissionScope.KNOWLEDGE_BASE, PermissionScope.RESEARCH,
        PermissionScope.REASONING, PermissionScope.MCP_TOOLS,
    } for s in PermissionScope},
    "autonomous": {s: True for s in PermissionScope},
}

_ACCESS_MODE_TO_PRESET: dict[str, str] = {
    "strict": "restricted",
    "balanced": "standard",
    "full": "autonomous",
}


class AgentPermissions(BaseModel):
    """Per-session agent permission scopes."""

    scopes: dict[PermissionScope, bool] = Field(default_factory=lambda: dict(_DEFAULT_SCOPES))
    budget_cap_usd: float = 0.0  # 0 = use global limit

    @classmethod
    def from_preset(cls, preset: str) -> "AgentPermissions":
        if preset not in _PRESETS:
            preset = "standard"
        return cls(scopes=dict(_PRESETS[preset]))

    @classmethod
    def from_access_mode(cls, mode: str) -> "AgentPermissions":
        preset = _ACCESS_MODE_TO_PRESET.get(mode, "standard")
        return cls.from_preset(preset)

    def to_capabilities(self) -> set[str]:
        """Convert enabled scopes to ToolGate capability strings."""
        caps: set[str] = {"chat"}
        for scope, enabled in self.scopes.items():
            if enabled and scope in _SCOPE_CAPABILITIES:
                caps |= _SCOPE_CAPABILITIES[scope]
        return caps

    def allowed_builtin_tools(self) -> set[str]:
        """Return set of allowed built-in tool names."""
        tools: set[str] = set()
        for scope, enabled in self.scopes.items():
            if enabled and scope in _SCOPE_BUILTIN_TOOLS:
                tools |= _SCOPE_BUILTIN_TOOLS[scope]
        return tools

    def active_count(self) -> int:
        return sum(1 for v in self.scopes.values() if v)

    def matching_preset(self) -> str | None:
        for name, preset_scopes in _PRESETS.items():
            if self.scopes == preset_scopes:
                return name
        return None
