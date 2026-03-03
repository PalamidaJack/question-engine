"""A2A protocol Pydantic models (Phase 4).
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, List


class AgentCapabilities(BaseModel):
    intents: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)


class AgentSkill(BaseModel):
    name: str
    description: str = ""


class AgentCard(BaseModel):
    name: str = "Question Engine"
    description: str = "Cognitive architecture for autonomous knowledge discovery"
    url: str = ""
    version: str = "1.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = Field(default_factory=list)


class A2AMessage(BaseModel):
    sender: str = "external"
    content: dict[str, Any] = Field(default_factory=dict)


class A2AArtifact(BaseModel):
    id: str
    uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATask(BaseModel):
    id: str
    status: str = "submitted"
    messages: List[A2AMessage] = Field(default_factory=list)
    artifacts: List[A2AArtifact] = Field(default_factory=list)
