"""InquiryEngine — the 7-phase iterative inquiry loop.

Orchestrates all Phase 1+2 components to answer complex questions through
iterative observation, questioning, investigation, synthesis, and reflection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from qe.errors import QEError, classify_error
from qe.models.cognition import CrystallizedInsight
from qe.services.inquiry.hypothesis import HypothesisManager
from qe.services.inquiry.question_generator import QuestionGenerator
from qe.services.inquiry.schemas import (
    InquiryConfig,
    InquiryResult,
    InquiryState,
    InvestigationResult,
    Question,
    Reflection,
    TerminationReason,
)

log = logging.getLogger(__name__)


class _InquiryRateLimiter:
    """Concurrency + rate limiting for inquiry runs.

    Uses an asyncio.Semaphore for concurrency control and a token bucket
    for rate limiting (RPM / 60 refill rate).
    """

    def __init__(self, max_concurrent: int = 3, rpm: int = 10) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rpm = rpm
        self._tokens = float(rpm)
        self._max_tokens = float(rpm)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()

    def try_acquire_rate(self) -> bool:
        """Non-blocking rate check. Returns True if a token is available."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore


class InquiryEngine:
    """The main 7-phase inquiry loop.

    Phases: Observe → Orient → Question → Prioritize → Investigate → Synthesize → Reflect

    All cognitive components are received via constructor injection.
    """

    def __init__(
        self,
        # Phase 1
        episodic_memory: Any = None,
        context_curator: Any = None,
        # Phase 2
        metacognitor: Any = None,
        epistemic_reasoner: Any = None,
        dialectic_engine: Any = None,
        persistence_engine: Any = None,
        insight_crystallizer: Any = None,
        # Phase 3
        question_generator: QuestionGenerator | None = None,
        hypothesis_manager: HypothesisManager | None = None,
        question_store: Any = None,
        procedural_memory: Any = None,
        # Infrastructure
        tool_registry: Any = None,
        budget_tracker: Any = None,
        bus: Any = None,
        config: InquiryConfig | None = None,
        rate_limiter: _InquiryRateLimiter | None = None,
    ) -> None:
        self._episodic = episodic_memory
        self._curator = context_curator
        self._metacognitor = metacognitor
        self._epistemic = epistemic_reasoner
        self._dialectic = dialectic_engine
        self._persistence = persistence_engine
        self._crystallizer = insight_crystallizer
        self._question_gen = question_generator or QuestionGenerator()
        self._hypothesis_mgr = hypothesis_manager or HypothesisManager()
        self._question_store = question_store
        self._procedural = procedural_memory
        self._tool_registry = tool_registry
        self._budget = budget_tracker
        self._bus = bus
        self._config = config or InquiryConfig()
        self._rate_limiter = rate_limiter

    async def run_inquiry(
        self,
        goal_id: str,
        goal_description: str,
        config: InquiryConfig | None = None,
    ) -> InquiryResult:
        """Execute the full 7-phase inquiry loop."""
        cfg = config or self._config
        start_time = time.monotonic()

        # Rate limiting check
        if self._rate_limiter is not None:
            if not self._rate_limiter.try_acquire_rate():
                return InquiryResult(
                    inquiry_id=f"inq_rl_{goal_id}",
                    goal_id=goal_id,
                    status="failed",
                    termination_reason="rate_limited",
                )

        state = InquiryState(
            goal_id=goal_id,
            goal_description=goal_description,
            config=cfg,
        )

        self._publish("inquiry.started", goal_id, {
            "inquiry_id": state.inquiry_id,
            "goal_id": goal_id,
            "goal": goal_description,
        })

        # Acquire concurrency semaphore if rate limiter is present
        if self._rate_limiter is not None:
            try:
                await asyncio.wait_for(
                    self._rate_limiter.semaphore.acquire(), timeout=30
                )
            except TimeoutError:
                return InquiryResult(
                    inquiry_id=state.inquiry_id,
                    goal_id=goal_id,
                    status="failed",
                    termination_reason="rate_limited",
                )

        try:
            return await asyncio.wait_for(
                self._run_inquiry_loop(state, cfg, goal_id, start_time),
                timeout=cfg.inquiry_timeout_seconds,
            )
        except TimeoutError:
            log.warning(
                "inquiry.timeout inquiry_id=%s timeout=%.1fs",
                state.inquiry_id,
                cfg.inquiry_timeout_seconds,
            )
            return self._finalize(
                state, "timeout", [], start_time, status="completed"
            )
        finally:
            if self._rate_limiter is not None:
                self._rate_limiter.semaphore.release()

    async def _run_inquiry_loop(
        self,
        state: InquiryState,
        cfg: InquiryConfig,
        goal_id: str,
        start_time: float,
    ) -> InquiryResult:
        """Core iteration loop, extracted for timeout wrapping."""
        all_insights: list[CrystallizedInsight] = []
        termination_reason: TerminationReason = "max_iterations"

        try:
            for iteration in range(cfg.max_iterations):
                state.current_iteration = iteration

                # Check budget before each iteration
                if self._budget and self._should_stop_budget(cfg):
                    termination_reason = "budget_exhausted"
                    self._publish("inquiry.budget_warning", goal_id, {
                        "inquiry_id": state.inquiry_id,
                        "iteration": iteration,
                    })
                    break

                # 1. OBSERVE
                state.current_phase = "observe"
                _t0 = time.monotonic()
                context = await self._run_phase_with_retry(
                    "observe", self._phase_observe, state
                )
                self._record_phase_timing(state, "observe", _t0)
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "observe",
                    "iteration": iteration,
                })

                # 2. ORIENT
                state.current_phase = "orient"
                _t0 = time.monotonic()
                await self._run_phase_with_retry(
                    "orient", self._phase_orient, state, context
                )
                self._record_phase_timing(state, "orient", _t0)
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "orient",
                    "iteration": iteration,
                })

                # 3. QUESTION
                state.current_phase = "question"
                _t0 = time.monotonic()
                new_questions = await self._run_phase_with_retry(
                    "question", self._phase_question, state
                )
                state.questions.extend(new_questions)
                await self._persist_questions(state.inquiry_id, new_questions)
                self._record_phase_timing(state, "question", _t0)
                for q in new_questions:
                    self._publish("inquiry.question_generated", goal_id, {
                        "inquiry_id": state.inquiry_id,
                        "question_id": q.question_id,
                        "text": q.text,
                    })
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "question",
                    "iteration": iteration,
                })

                # 4. PRIORITIZE
                state.current_phase = "prioritize"
                pending = [q for q in state.questions if q.status == "pending"]
                if not pending:
                    termination_reason = "all_questions_answered"
                    break
                _t0 = time.monotonic()
                prioritized = await self._run_phase_with_retry(
                    "prioritize", self._phase_prioritize, state, pending
                )
                self._record_phase_timing(state, "prioritize", _t0)
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "prioritize",
                    "iteration": iteration,
                })

                # 5. INVESTIGATE top question
                state.current_phase = "investigate"
                top_question = prioritized[0]
                top_question.status = "investigating"
                _t0 = time.monotonic()
                investigation = await self._run_phase_with_retry(
                    "investigate", self._phase_investigate, state, top_question
                )
                state.investigations.append(investigation)
                await self._record_procedural_outcome(top_question, investigation)
                await self._persist_questions(state.inquiry_id, [top_question])
                self._record_phase_timing(state, "investigate", _t0)
                self._publish("inquiry.investigation_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "question_id": top_question.question_id,
                })
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "investigate",
                    "iteration": iteration,
                })

                # 6. SYNTHESIZE
                state.current_phase = "synthesize"
                _t0 = time.monotonic()
                insights = await self._run_phase_with_retry(
                    "synthesize", self._phase_synthesize,
                    state, top_question, investigation,
                )
                self._record_phase_timing(state, "synthesize", _t0)
                all_insights.extend(insights)
                for ins in insights:
                    self._publish("inquiry.insight_generated", goal_id, {
                        "inquiry_id": state.inquiry_id,
                        "insight_id": ins.insight_id,
                        "headline": ins.headline,
                    })
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "synthesize",
                    "iteration": iteration,
                })

                # 7. REFLECT
                state.current_phase = "reflect"
                _t0 = time.monotonic()
                reflection = await self._run_phase_with_retry(
                    "reflect", self._phase_reflect, state, iteration
                )
                self._record_phase_timing(state, "reflect", _t0)
                state.reflections.append(reflection)
                self._publish("inquiry.phase_completed", goal_id, {
                    "inquiry_id": state.inquiry_id,
                    "goal_id": goal_id,
                    "phase": "reflect",
                    "iteration": iteration,
                    "decision": reflection.decision,
                })

                if reflection.decision == "terminate":
                    # Determine specific reason
                    if state.overall_confidence >= cfg.confidence_threshold:
                        termination_reason = "confidence_met"
                    elif reflection.questions_answered >= len(state.questions) > 0:
                        termination_reason = "all_questions_answered"
                    else:
                        termination_reason = "confidence_met"
                    break

                # Check confidence threshold
                if state.overall_confidence >= cfg.confidence_threshold:
                    termination_reason = "confidence_met"
                    break

        except QEError:
            log.exception(
                "inquiry.structured_error inquiry_id=%s iteration=%d",
                state.inquiry_id,
                state.current_iteration,
            )
            self._publish("inquiry.failed", goal_id, {
                "inquiry_id": state.inquiry_id,
                "iteration": state.current_iteration,
            })
            return self._finalize(
                state, "max_iterations", all_insights, start_time, status="failed"
            )
        except Exception:
            log.exception(
                "inquiry.iteration_error inquiry_id=%s iteration=%d",
                state.inquiry_id,
                state.current_iteration,
            )
            self._publish("inquiry.failed", goal_id, {
                "inquiry_id": state.inquiry_id,
                "iteration": state.current_iteration,
            })
            return self._finalize(
                state, "max_iterations", all_insights, start_time, status="failed"
            )

        return self._finalize(
            state, termination_reason, all_insights, start_time
        )

    # -------------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------------

    async def _phase_observe(self, state: InquiryState) -> dict[str, Any]:
        """Phase 1: Gather context from episodic memory and active hypotheses."""
        context: dict[str, Any] = {}

        # Recall relevant episodes
        if self._episodic is not None:
            try:
                episodes = await self._episodic.recall_for_goal(
                    state.goal_id, top_k=10
                )
                context["episodes"] = episodes
            except Exception:
                log.debug("observe.episodic_recall_failed")
                context["episodes"] = []
        else:
            context["episodes"] = []

        # Get active hypotheses
        try:
            hypotheses = await self._hypothesis_mgr.get_active_hypotheses()
            context["hypotheses"] = hypotheses
        except Exception:
            log.debug("observe.hypothesis_fetch_failed")
            context["hypotheses"] = []

        return context

    async def _phase_orient(
        self, state: InquiryState, context: dict[str, Any]
    ) -> str:
        """Phase 2: Use metacognitor to suggest next approach."""
        # Consult procedural memory for effective templates
        if self._procedural is not None:
            try:
                templates = await self._procedural.get_best_templates(
                    domain=state.config.domain, top_k=3
                )
                if templates:
                    hints = "\n".join(
                        f"- {t.pattern} (success_rate={t.success_rate:.0%})"
                        for t in templates
                    )
                    state.findings_summary += (
                        f"\n[Procedural hints]\n{hints}\n"
                    )
            except Exception:
                log.debug("orient.procedural_memory_failed")

        if self._metacognitor is not None:
            try:
                assessment = await self._metacognitor.suggest_next_approach(
                    goal_id=state.goal_id,
                    goal_description=state.goal_description,
                )
                return assessment.recommended_approach
            except Exception:
                log.debug("orient.metacognitor_failed")
        return "Continue investigation with available tools."

    async def _phase_question(self, state: InquiryState) -> list[Question]:
        """Phase 3: Generate new questions."""
        asked = [q.text for q in state.questions]
        hypotheses = await self._hypothesis_mgr.get_active_hypotheses()
        hyp_summary = "\n".join(
            f"- {h.statement} (p={h.current_probability:.2f})"
            for h in hypotheses
        ) if hypotheses else ""

        epistemic_state: dict[str, Any] = {}
        if self._epistemic is not None:
            try:
                ep_state = self._epistemic.get_epistemic_state(state.goal_id)
                if ep_state is not None:
                    epistemic_state = {
                        "known_facts": len(ep_state.known_facts),
                        "known_unknowns": len(ep_state.known_unknowns),
                        "overall_confidence": ep_state.overall_confidence,
                    }
            except Exception:
                log.debug("question.epistemic_state_failed")

        return await self._question_gen.generate(
            goal=state.goal_description,
            findings_summary=state.findings_summary,
            asked_questions=asked,
            epistemic_state=epistemic_state,
            hypotheses_summary=hyp_summary,
            iteration=state.current_iteration,
            max_iterations=state.config.max_iterations,
            n_questions=state.config.questions_per_iteration,
        )

    async def _phase_prioritize(
        self, state: InquiryState, questions: list[Question]
    ) -> list[Question]:
        """Phase 4: Sort questions by priority score."""
        return await self._question_gen.prioritize(
            state.goal_description, questions
        )

    async def _phase_investigate(
        self, state: InquiryState, question: Question
    ) -> InvestigationResult:
        """Phase 5: Investigate a question using tools."""
        tool_calls: list[dict[str, Any]] = []
        raw_findings = ""

        if self._tool_registry is not None:
            try:
                import litellm as _litellm

                tools = self._tool_registry.get_tool_schemas(
                    {"web_search", "code_exec"}, mode="direct"
                )

                if tools:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                f"You are investigating this question: {question.text}\n"
                                f"Goal: {state.goal_description}\n"
                                f"Use the available tools to find the answer."
                            ),
                        },
                        {"role": "user", "content": question.text},
                    ]

                    for _i in range(state.config.max_tool_calls_per_question):
                        response = await _litellm.acompletion(
                            model=state.config.model_fast,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto",
                        )
                        choice = response.choices[0]
                        if not choice.message.tool_calls:
                            raw_findings = choice.message.content or ""
                            break

                        messages.append(choice.message.model_dump())
                        for tc in choice.message.tool_calls:
                            import json
                            try:
                                params = json.loads(tc.function.arguments)
                            except Exception:
                                params = {}
                            try:
                                result = await self._tool_registry.execute(
                                    tc.function.name, params
                                )
                                result_str = (
                                    json.dumps(result)
                                    if not isinstance(result, str)
                                    else result
                                )
                            except Exception as exc:
                                result_str = f"Error: {exc}"
                            tool_calls.append({
                                "tool": tc.function.name,
                                "params": params,
                                "result": result_str[:500],
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_str,
                            })
                else:
                    raw_findings = "No tools available for investigation."
            except Exception:
                log.exception("investigate.tool_loop_failed question=%s", question.question_id)
                raw_findings = "Investigation failed due to tool error."
        else:
            raw_findings = "No tool registry configured."

        # Mark question as answered with derived confidence
        question.status = "answered"
        question.answer = raw_findings[:2000]
        if not raw_findings:
            question.confidence_in_answer = 0.0
        elif tool_calls:
            # Tools were used — higher confidence based on # of successful calls
            question.confidence_in_answer = min(0.5 + 0.1 * len(tool_calls), 0.9)
        elif self._tool_registry is not None:
            # Tools available but LLM answered directly
            question.confidence_in_answer = 0.6
        else:
            # No tools — low confidence
            question.confidence_in_answer = 0.3

        return InvestigationResult(
            question_id=question.question_id,
            tool_calls=tool_calls,
            raw_findings=raw_findings,
        )

    async def _phase_synthesize(
        self,
        state: InquiryState,
        question: Question,
        investigation: InvestigationResult,
    ) -> list[CrystallizedInsight]:
        """Phase 6: Synthesize findings — epistemic + dialectic + crystallize."""
        insights: list[CrystallizedInsight] = []

        # Update findings summary
        if investigation.raw_findings:
            state.findings_summary += (
                f"\n[Q: {question.text}]\n"
                f"A: {investigation.raw_findings[:500]}\n"
            )

        # Epistemic assessment
        contradictions: list[str] = []
        if self._epistemic is not None:
            try:
                await self._epistemic.assess_uncertainty(
                    state.goal_id,
                    investigation.raw_findings[:500],
                    state.goal_description,
                )
                # Detect surprises
                surprise = await self._epistemic.detect_surprise(
                    state.goal_id,
                    investigation.raw_findings[:500],
                )
                if surprise and surprise.surprise_magnitude > 0.5:
                    contradictions.append(surprise.finding)
            except Exception:
                log.debug("synthesize.epistemic_failed")

        # Dialectic challenge
        if self._dialectic is not None and investigation.raw_findings:
            try:
                report = await self._dialectic.full_dialectic(
                    goal_id=state.goal_id,
                    conclusion=investigation.raw_findings[:500],
                    evidence=question.text,
                    domain=state.config.domain,
                )
                state.overall_confidence = max(
                    state.overall_confidence, report.revised_confidence
                )
            except Exception:
                log.debug("synthesize.dialectic_failed")

        # Knowledge Loop: contradictions → hypotheses
        if contradictions:
            try:
                hyps = await self._hypothesis_mgr.generate_hypotheses(
                    goal=state.goal_description,
                    contradictions=contradictions,
                )
                state.hypotheses_tested += len(hyps)
                for h in hyps:
                    self._publish("inquiry.hypothesis_generated", state.goal_id, {
                        "inquiry_id": state.inquiry_id,
                        "hypothesis_id": h.hypothesis_id,
                        "statement": h.statement,
                    })
                    # Generate falsification questions for next iteration
                    falsification_qs = self._hypothesis_mgr.create_falsification_questions(h)
                    state.questions.extend(falsification_qs)
            except Exception:
                log.debug("synthesize.hypothesis_generation_failed")

        # Crystallize insights
        if self._crystallizer is not None and investigation.raw_findings:
            try:
                insight = await self._crystallizer.crystallize(
                    goal_id=state.goal_id,
                    finding=investigation.raw_findings[:500],
                    domain=state.config.domain,
                    evidence=question.text,
                )
                if insight is not None:
                    insights.append(insight)
            except Exception:
                log.debug("synthesize.crystallize_failed")

        return insights

    async def _phase_reflect(
        self, state: InquiryState, iteration: int
    ) -> Reflection:
        """Phase 7: Reflect on progress — drift, blind spots, termination."""
        answered = sum(1 for q in state.questions if q.status == "answered")
        total = len(state.questions)

        # Drift detection
        drift_score = 0.0
        if self._curator is not None:
            try:
                drift = self._curator.detect_drift(
                    state.goal_id, state.findings_summary[:200]
                )
                if drift is not None:
                    drift_score = drift.similarity
            except Exception:
                log.debug("reflect.drift_detection_failed")

        # Blind spot warning
        if self._epistemic is not None:
            try:
                warning = self._epistemic.get_blind_spot_warning(state.goal_id)
                if warning:
                    log.info(
                        "reflect.blind_spot inquiry=%s warning=%s",
                        state.inquiry_id,
                        warning[:100],
                    )
            except Exception:
                log.debug("reflect.blind_spot_failed")

        # Decision
        on_track = drift_score > 0.5 or drift_score == 0.0
        confidence = state.overall_confidence

        decision = "continue"
        reasoning = f"Answered {answered}/{total} questions. Confidence: {confidence:.2f}"

        if not on_track and drift_score > 0:
            decision = "refocus"
            reasoning += f" Drift detected (similarity={drift_score:.2f})."

            # Persistence Engine: root cause analysis
            if self._persistence is not None:
                try:
                    root_cause = await self._persistence.analyze_root_cause(
                        goal_id=state.goal_id,
                        failure_summary=f"Drift detected at iteration {iteration}",
                        context=state.findings_summary[:500],
                    )
                    reasoning += f" Root cause: {root_cause.root_cause}"
                except Exception:
                    log.debug("reflect.persistence_root_cause_failed")

                try:
                    reframed = await self._persistence.reframe(
                        goal_id=state.goal_id,
                        original_problem=state.goal_description,
                        tried_approaches=state.findings_summary[:200],
                        failure_reason=f"Drift score {drift_score:.2f}",
                    )
                    state.questions.append(
                        Question(text=reframed.reframed_question)
                    )
                except Exception:
                    log.debug("reflect.persistence_reframe_failed")

                try:
                    lessons = self._persistence.get_relevant_lessons(
                        state.goal_description, top_k=2
                    )
                    reasoning += f" ({len(lessons)} relevant lessons found)"
                except Exception:
                    log.debug("reflect.persistence_lessons_failed")
        elif confidence >= state.config.confidence_threshold:
            decision = "terminate"
            reasoning += " Confidence threshold met."
        elif answered >= total and total > 0:
            decision = "terminate"
            reasoning += " All questions answered."

        return Reflection(
            iteration=iteration,
            drift_score=drift_score,
            on_track=on_track,
            confidence_in_progress=confidence,
            decision=decision,
            reasoning=reasoning,
            questions_answered=answered,
        )

    def _finalize(
        self,
        state: InquiryState,
        reason: TerminationReason,
        insights: list[CrystallizedInsight],
        start_time: float,
        status: str = "completed",
    ) -> InquiryResult:
        """Build the final InquiryResult."""
        duration = time.monotonic() - start_time
        answered = sum(1 for q in state.questions if q.status == "answered")

        total_cost = sum(inv.cost_usd for inv in state.investigations)

        # Compute per-phase timing stats
        phase_timing_stats: dict[str, dict[str, float]] = {}
        for phase_name, durations in state.phase_timings.items():
            if durations:
                phase_timing_stats[phase_name] = {
                    "count": float(len(durations)),
                    "total_s": sum(durations),
                    "avg_s": sum(durations) / len(durations),
                    "max_s": max(durations),
                }

        result = InquiryResult(
            inquiry_id=state.inquiry_id,
            goal_id=state.goal_id,
            status=status,
            termination_reason=reason,
            iterations_completed=state.current_iteration + 1,
            total_questions_generated=len(state.questions),
            total_questions_answered=answered,
            findings_summary=state.findings_summary,
            insights=[
                {"insight_id": i.insight_id, "headline": i.headline}
                for i in insights
            ],
            hypotheses_tested=state.hypotheses_tested,
            total_cost_usd=total_cost,
            duration_seconds=duration,
            phase_timings=phase_timing_stats,
            question_tree=state.questions,
        )

        self._publish("inquiry.completed", state.goal_id, {
            "inquiry_id": state.inquiry_id,
            "goal_id": state.goal_id,
            "status": status,
            "iterations": state.current_iteration + 1,
            "insights": len(insights),
            "questions_answered": answered,
        })

        return result

    # -------------------------------------------------------------------
    # Retry / Error recovery
    # -------------------------------------------------------------------

    async def _run_phase_with_retry(
        self, phase_name: str, coro_fn: Any, *args: Any, max_retries: int = 2
    ) -> Any:
        """Run a phase coroutine with structured error retry."""
        for attempt in range(max_retries + 1):
            try:
                return await coro_fn(*args)
            except Exception as exc:
                classified = classify_error(exc)
                if not classified.is_retryable or attempt >= max_retries:
                    raise classified from exc
                log.warning(
                    "inquiry.phase_retry phase=%s attempt=%d/%d delay_ms=%d",
                    phase_name,
                    attempt + 1,
                    max_retries,
                    classified.retry_delay_ms,
                )
                await asyncio.sleep(classified.retry_delay_ms / 1000.0)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _record_phase_timing(
        state: InquiryState, phase: str, start: float
    ) -> None:
        """Append a phase duration to state.phase_timings."""
        elapsed = time.monotonic() - start
        state.phase_timings.setdefault(phase, []).append(elapsed)

    async def _persist_questions(
        self, inquiry_id: str, questions: list[Question]
    ) -> None:
        """Persist questions to QuestionStore if available."""
        if self._question_store is None:
            return
        for q in questions:
            try:
                await self._question_store.save_question(inquiry_id, q)
            except Exception:
                log.debug("persist_question.failed question_id=%s", q.question_id)

    async def _record_procedural_outcome(
        self, question: Question, investigation: InvestigationResult
    ) -> None:
        """Record question template outcome in ProceduralMemory if available."""
        if self._procedural is None:
            return
        try:
            success = bool(investigation.raw_findings and question.confidence_in_answer > 0.3)
            await self._procedural.record_template_outcome(
                template_id=None,
                pattern=question.text[:200],
                question_type=question.question_type,
                success=success,
                info_gain=question.confidence_in_answer,
                domain=self._config.domain,
            )
            if investigation.tool_calls:
                tool_names = [tc.get("tool", "") for tc in investigation.tool_calls]
                await self._procedural.record_sequence_outcome(
                    sequence_id=None,
                    tool_names=tool_names,
                    success=success,
                    cost_usd=investigation.cost_usd,
                    domain=self._config.domain,
                )
        except Exception:
            log.debug("record_procedural.failed question_id=%s", question.question_id)

    def _should_stop_budget(self, cfg: InquiryConfig) -> bool:
        """Check if remaining budget is below hard stop percentage."""
        if self._budget is None:
            return False
        limit = max(self._budget.monthly_limit_usd, 0.01)
        remaining_pct = 1.0 - (self._budget._total_spend / limit)
        return remaining_pct < cfg.budget_hard_stop_pct

    def _publish(self, topic: str, goal_id: str, payload: dict[str, Any]) -> None:
        """Publish a bus event if bus is available."""
        if self._bus is None:
            return
        try:
            from qe.models.envelope import Envelope
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="inquiry_engine",
                    correlation_id=goal_id,
                    payload=payload,
                )
            )
        except Exception:
            log.debug("inquiry.publish_failed topic=%s", topic)
