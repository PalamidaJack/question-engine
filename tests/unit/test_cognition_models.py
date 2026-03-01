"""Tests for cognition Pydantic models."""


from qe.models.cognition import (
    AbsenceDetection,
    ActionabilityResult,
    ApproachAssessment,
    ApproachNode,
    AssumptionChallenge,
    CapabilityGap,
    CapabilityProfile,
    Counterargument,
    CrystallizedInsight,
    DialecticReport,
    EpistemicState,
    KnownUnknown,
    MechanismExplanation,
    NoveltyAssessment,
    PerspectiveAnalysis,
    ProvenanceChain,
    ReasoningTrace,
    ReframingResult,
    RootCauseAnalysis,
    RootCauseLink,
    SurpriseDetection,
    ToolCombinationSuggestion,
    UncertaintyAssessment,
)


class TestReasoningTrace:
    def test_default_fields(self):
        t = ReasoningTrace(phase="metacognition", goal_id="g1")
        assert t.trace_id.startswith("trc_")
        assert t.phase == "metacognition"
        assert t.goal_id == "g1"
        assert t.reasoning_steps == []

    def test_serialization(self):
        t = ReasoningTrace(phase="dialectic", goal_id="g1")
        d = t.model_dump()
        assert "trace_id" in d
        assert d["phase"] == "dialectic"


class TestMetacognitorModels:
    def test_capability_profile(self):
        p = CapabilityProfile(
            tool_name="web_search",
            description="Search the web",
            domains=["general"],
            limitations=["no authenticated sites"],
        )
        assert p.tool_name == "web_search"
        assert len(p.limitations) == 1

    def test_capability_gap(self):
        g = CapabilityGap(description="No database access")
        assert g.gap_id.startswith("gap_")
        assert g.severity == "degraded"

    def test_approach_node_defaults(self):
        n = ApproachNode(approach_description="Try web search")
        assert n.status == "untried"
        assert n.parent_id is None
        assert n.children == []

    def test_approach_assessment(self):
        a = ApproachAssessment(
            recommended_approach="Use code execution",
            reasoning="Web search found nothing",
        )
        assert a.estimated_success_probability == 0.5

    def test_tool_combination(self):
        tc = ToolCombinationSuggestion(
            description="Combine search + code",
            tool_sequence=["web_search", "code_exec"],
            reasoning="Compute from public data",
        )
        assert len(tc.tool_sequence) == 2


class TestEpistemicModels:
    def test_absence_detection(self):
        a = AbsenceDetection(
            expected_data="Revenue figures",
            why_expected="Required for valuation",
        )
        assert a.absence_id.startswith("abs_")
        assert a.significance == "medium"

    def test_uncertainty_assessment(self):
        u = UncertaintyAssessment(finding_summary="Revenue grew 20%")
        assert u.confidence_level == "moderate"
        assert u.evidence_quality == "secondary"

    def test_known_unknown(self):
        ku = KnownUnknown(
            question="What is the debt ratio?",
            why_unknown="No access to financials",
        )
        assert ku.unknown_id.startswith("unk_")
        assert ku.importance == "medium"

    def test_surprise_detection(self):
        s = SurpriseDetection(
            finding="Revenue dropped 50%",
            expected_instead="Revenue growth of 20%",
            surprise_magnitude=0.9,
        )
        assert s.surprise_id.startswith("sur_")
        assert s.surprise_magnitude == 0.9

    def test_epistemic_state(self):
        es = EpistemicState(goal_id="g1")
        assert es.known_facts == []
        assert es.known_unknowns == []
        assert es.absences == []
        assert es.surprises == []


class TestDialecticModels:
    def test_counterargument(self):
        c = Counterargument(
            target_claim="Market is undervalued",
            counterargument="P/E ratio is historically high",
        )
        assert c.strength == "moderate"

    def test_perspective_analysis(self):
        p = PerspectiveAnalysis(
            perspective_name="bearish analyst",
            key_observations=["High debt levels"],
        )
        assert len(p.key_observations) == 1

    def test_assumption_challenge(self):
        a = AssumptionChallenge(
            assumption="Growth will continue",
            challenge="Recession risk",
        )
        assert a.is_explicit is True
        assert a.testable is True

    def test_dialectic_report(self):
        r = DialecticReport(original_conclusion="Market is undervalued")
        assert r.revised_confidence == 0.5
        assert r.should_investigate_further is False


class TestPersistenceModels:
    def test_root_cause_link(self):
        link = RootCauseLink(
            level=1,
            question="Why did the search fail?",
            answer="Wrong search terms",
        )
        assert link.confidence == 0.5

    def test_root_cause_analysis(self):
        rca = RootCauseAnalysis(
            failure_summary="Empty search results",
            chain=[
                RootCauseLink(level=1, question="Why?", answer="Bad terms"),
                RootCauseLink(level=2, question="Why?", answer="Domain mismatch"),
                RootCauseLink(level=3, question="Why?", answer="No domain context"),
            ],
            root_cause="No domain context provided",
            lesson_learned="Always include domain context in searches",
        )
        assert len(rca.chain) == 3

    def test_reframing_result(self):
        r = ReframingResult(
            original_framing="Find X directly",
            reframing_strategy="inversion",
            reframed_question="What would imply X?",
            reasoning="Direct search failed",
        )
        assert r.estimated_tractability == 0.5


class TestInsightModels:
    def test_novelty_assessment_defaults(self):
        n = NoveltyAssessment(finding="Some finding")
        assert n.is_novel is False
        assert n.novelty_type == "not_novel"

    def test_mechanism_explanation(self):
        m = MechanismExplanation(
            what_happens="Grid stocks underperform",
            why_it_happens="Sector misclassification",
            how_it_works="Classification groups grid with solar",
        )
        assert m.confidence_in_mechanism == 0.5

    def test_provenance_chain(self):
        p = ProvenanceChain(
            original_question="Why is grid infrastructure mispriced?",
            evidence_items=["ETF performance data"],
            reasoning_steps=["Compared grid vs solar ETFs"],
            insight="Grid mispriced due to classification",
        )
        assert len(p.evidence_items) == 1

    def test_actionability_result(self):
        a = ActionabilityResult(
            score=0.8,
            who_can_act="Portfolio managers",
            what_action="Overweight grid infrastructure ETFs",
            time_horizon="weeks",
        )
        assert a.score == 0.8

    def test_crystallized_insight(self):
        insight = CrystallizedInsight(
            headline="Grid infrastructure mispriced",
            mechanism=MechanismExplanation(
                what_happens="Grid undervalued",
                why_it_happens="Classification issue",
                how_it_works="Lumped with solar",
            ),
            novelty=NoveltyAssessment(
                finding="Grid mispriced",
                is_novel=True,
                novelty_type="new_connection",
            ),
            provenance=ProvenanceChain(
                original_question="Find opportunities",
                insight="Grid mispriced",
            ),
            actionability_score=0.8,
            dialectic_survivor=True,
        )
        assert insight.insight_id.startswith("ins_")
        assert insight.dialectic_survivor is True
        assert insight.confidence == 0.5
