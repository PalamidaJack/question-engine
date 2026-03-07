"""Pydantic models for visual workflow definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    # Composition
    ENTRY = "entry"
    EXIT = "exit"
    # Tool nodes (prefixed with tool/)
    TOOL = "tool"
    # LLM
    LLM_CHAT = "llm/chat"
    LLM_MULTI = "llm/multi"
    LLM_SUMMARIZE = "llm/summarize"
    # Control
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    MERGE = "merge"
    DELAY = "delay"
    # Memory
    READ_EPISODIC = "memory/read_episodic"
    WRITE_BELIEF = "memory/write_belief"
    QUERY_PROCEDURAL = "memory/query_procedural"
    WORKING_MEMORY = "memory/working_memory"
    # Transform
    EXTRACT = "transform/extract"
    FORMAT = "transform/format"
    FILTER = "transform/filter"
    AGGREGATE = "transform/aggregate"
    # Human
    APPROVAL_GATE = "human/approval"
    INPUT_REQUEST = "human/input"
    REVIEW = "human/review"
    # Composition
    SUB_WORKFLOW = "sub_workflow"


class WorkflowNode(BaseModel):
    id: str
    type: str  # NodeType value or "tool/{tool_name}"
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(
        default_factory=dict,
    )  # x, y for UI


class WorkflowEdge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    label: str = ""
    condition: str | None = None  # "true" or "false"

    model_config = {"populate_by_name": True}


class WorkflowMetadata(BaseModel):
    created_by: str = "user"
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0"


class WorkflowDefinition(BaseModel):
    id: str
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    metadata: WorkflowMetadata = Field(
        default_factory=WorkflowMetadata,
    )

    def get_node(self, node_id: str) -> WorkflowNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_entry_node(self) -> WorkflowNode | None:
        for n in self.nodes:
            if n.type == "entry":
                return n
        return None

    def get_outgoing_edges(
        self, node_id: str,
    ) -> list[WorkflowEdge]:
        return [e for e in self.edges if e.from_node == node_id]

    def get_incoming_edges(
        self, node_id: str,
    ) -> list[WorkflowEdge]:
        return [e for e in self.edges if e.to_node == node_id]
