"""Tests for CognitiveAgent model."""

from __future__ import annotations

from qe.runtime.cognitive_agent import (
    AgentEpistemicState,
    CognitiveAgent,
    CoreMemory,
)


class TestCoreMemory:
    def test_defaults(self):
        cm = CoreMemory()
        assert cm.learned_patterns == []
        assert cm.domain_expertise == []
        assert cm.tool_preferences == {}


class TestAgentEpistemicState:
    def test_defaults(self):
        state = AgentEpistemicState()
        assert state.known_facts_count == 0
        assert state.confidence_level == 0.5


class TestCognitiveAgent:
    def test_defaults(self):
        agent = CognitiveAgent()
        assert agent.agent_id.startswith("cag_")
        assert agent.status == "idle"
        assert agent.persona == "general researcher"

    def test_update_epistemic_state(self):
        agent = CognitiveAgent()
        agent.update_epistemic_state(
            known_facts=8, known_unknowns=2, active_hypotheses=3
        )
        assert agent.epistemic_state.known_facts_count == 8
        assert agent.epistemic_state.known_unknowns_count == 2
        assert agent.epistemic_state.active_hypotheses_count == 3
        assert agent.epistemic_state.confidence_level == 0.8  # 8/(8+2)

    def test_update_epistemic_state_zero(self):
        agent = CognitiveAgent()
        agent.update_epistemic_state(
            known_facts=0, known_unknowns=0, active_hypotheses=0
        )
        assert agent.epistemic_state.confidence_level == 0.5
