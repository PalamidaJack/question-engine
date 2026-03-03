"""HarvestService: autonomous knowledge improvement via idle free model capacity.

Runs as a background poll loop (like Scout). Each cycle selects the best harvest
mode based on available models and current knowledge state:

1. premium_sprint   — Route hard questions through temporarily-free powerful models
2. model_profile    — Profile untested free models with standardized test prompts
3. consensus_validate — Multi-model validation of low-confidence claims
4. adversarial_challenge — Red-team high-confidence claims using free models
5. knowledge_gap    — Research registered unknowns using free models
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm

from qe.config import HarvestConfig
from qe.models.envelope import Envelope
from qe.runtime.feature_flags import get_flag_store

log = logging.getLogger(__name__)

# Standardized prompts for model profiling
_PROFILE_PROMPTS: list[dict[str, str]] = [
    {
        "category": "reasoning",
        "prompt": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
            "Explain your reasoning step by step."
        ),
        "expected_answer": "9",
    },
    {
        "category": "factual",
        "prompt": "What is the capital of Australia? Provide only the city name.",
        "expected_answer": "Canberra",
    },
    {
        "category": "creative",
        "prompt": (
            "Write a haiku about artificial intelligence. "
            "Format: three lines with 5-7-5 syllable pattern."
        ),
        "expected_answer": "",  # Creative — scored by structure
    },
    {
        "category": "structured_output",
        "prompt": (
            'Return a JSON object with keys "name", "type", and "count" '
            'describing any fruit. Example: {"name": "apple", "type": "fruit", "count": 3}'
        ),
        "expected_answer": "",  # Scored by valid JSON
    },
    {
        "category": "instruction_following",
        "prompt": (
            "List exactly 3 prime numbers less than 20, separated by commas. "
            "Do not include any other text."
        ),
        "expected_answer": "",  # Scored by format compliance
    },
]


class HarvestService:
    """Autonomous knowledge improvement using idle free model capacity."""

    def __init__(
        self,
        bus,
        substrate,
        discovery,
        mass_executor,
        episodic_memory=None,
        epistemic_reasoner=None,
        procedural_memory=None,
        config: HarvestConfig | None = None,
    ) -> None:
        self._bus = bus
        self._substrate = substrate
        self._discovery = discovery
        self._mass_executor = mass_executor
        self._episodic_memory = episodic_memory
        self._epistemic_reasoner = epistemic_reasoner
        self._procedural_memory = procedural_memory
        self._config = config or HarvestConfig()

        # State
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._cycles_completed = 0
        self._last_cycle_at: datetime | None = None
        self._last_mode: str | None = None
        self._model_profiles: dict[str, dict[str, Any]] = {}
        self._processed_claim_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the harvest service background loop."""
        self._running = True
        await self._maybe_subscribe("models.discovered", self._on_models_discovered)
        await self._maybe_subscribe("models.tiers_updated", self._on_tiers_updated)
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("harvest.started interval=%ds", self._config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the harvest service."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._maybe_unsubscribe("models.discovered", self._on_models_discovered)
        await self._maybe_unsubscribe("models.tiers_updated", self._on_tiers_updated)
        log.info("harvest.stopped cycles=%d", self._cycles_completed)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Background loop: run cycles at configured interval."""
        while self._running:
            try:
                await asyncio.sleep(self._config.poll_interval_seconds)
                if not self._running:
                    break
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("harvest.poll_loop_error")

    async def _run_cycle(self) -> None:
        """Execute one harvest cycle if conditions are met."""
        if not get_flag_store().is_enabled("harvest_service"):
            return

        cycle_id = f"hcyc_{uuid.uuid4().hex[:8]}"
        self._publish("harvest.cycle_started", {"cycle_id": cycle_id})

        start = time.monotonic()
        mode = await self._select_mode()
        if mode is None:
            self._publish("harvest.cycle_completed", {
                "cycle_id": cycle_id,
                "mode": None,
                "result": "no_work",
                "duration_ms": 0,
            })
            return

        result: dict[str, Any] = {"mode": mode}
        try:
            handler = {
                "premium_sprint": self._mode_premium_sprint,
                "model_profile": self._mode_model_profile,
                "consensus_validate": self._mode_consensus_validate,
                "adversarial_challenge": self._mode_adversarial_challenge,
                "knowledge_gap": self._mode_knowledge_gap,
            }[mode]

            result = await asyncio.wait_for(
                handler(), timeout=self._config.cycle_timeout_seconds,
            )
            result["mode"] = mode
        except TimeoutError:
            log.warning("harvest.cycle_timeout mode=%s", mode)
            result = {"mode": mode, "error": "timeout"}
        except Exception:
            log.exception("harvest.cycle_error mode=%s", mode)
            result = {"mode": mode, "error": "exception"}

        elapsed_ms = (time.monotonic() - start) * 1000
        self._cycles_completed += 1
        self._last_cycle_at = datetime.now(UTC)
        self._last_mode = mode

        # Record in procedural memory
        if self._procedural_memory is not None:
            try:
                success = "error" not in result
                await self._procedural_memory.record_sequence_outcome(
                    sequence_id=cycle_id,
                    tool_names=[f"harvest.{mode}"],
                    success=success,
                    cost_usd=0.0,
                    domain="harvest",
                )
            except Exception:
                log.debug("harvest.procedural_record_failed")

        self._publish("harvest.cycle_completed", {
            "cycle_id": cycle_id,
            "duration_ms": round(elapsed_ms, 1),
            **result,
        })

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    async def _select_mode(self) -> str | None:
        """Choose the best harvest mode based on current state.

        Priority: premium_sprint → model_profile → consensus_validate
                  → adversarial_challenge → knowledge_gap
        """
        # 1. Premium sprint — powerful model temporarily free?
        if self._config.premium_sprint_enabled and self._discovery is not None:
            powerful_free = self._discovery.get_available_models(
                tier="powerful", free_only=True,
            )
            if powerful_free:
                return "premium_sprint"

        # 2. Model profile — unprofiled models?
        if self._config.model_profile_enabled and self._discovery is not None:
            all_free = self._discovery.get_available_models(free_only=True)
            unprofiled = [m for m in all_free if m.model_id not in self._model_profiles]
            if unprofiled:
                return "model_profile"

        # 3. Consensus validate — low-confidence claims?
        if self._substrate is not None:
            low_conf = await self._substrate.get_claims(
                min_confidence=0.0,
            )
            low_conf = [
                c for c in low_conf
                if c.confidence < self._config.low_confidence_threshold
                and c.claim_id not in self._processed_claim_ids
            ]
            if low_conf:
                return "consensus_validate"

        # 4. Adversarial challenge — high-confidence claims to red-team?
        if self._substrate is not None:
            high_conf = await self._substrate.get_claims(
                min_confidence=self._config.adversarial_confidence_threshold,
            )
            high_conf = [
                c for c in high_conf
                if c.claim_id not in self._processed_claim_ids
            ]
            if high_conf:
                return "adversarial_challenge"

        # 5. Knowledge gap — registered unknowns?
        if self._epistemic_reasoner is not None:
            for state in self._epistemic_reasoner._states.values():
                if state.known_unknowns:
                    return "knowledge_gap"

        return None

    # ------------------------------------------------------------------
    # Mode: consensus_validate
    # ------------------------------------------------------------------

    async def _mode_consensus_validate(self) -> dict[str, Any]:
        """Validate low-confidence claims via multi-model majority vote."""
        claims = await self._substrate.get_claims(min_confidence=0.0)
        claims = [
            c for c in claims
            if c.confidence < self._config.low_confidence_threshold
            and c.claim_id not in self._processed_claim_ids
        ]
        claims = claims[: self._config.max_claims_per_cycle]

        validated = 0
        for claim in claims:
            self._processed_claim_ids.add(claim.claim_id)
            prompt = (
                f"Evaluate this claim and respond with a JSON object "
                f'{{"verdict": "agree" or "disagree", "confidence": 0.0-1.0, '
                f'"reasoning": "brief explanation"}}.\n\n'
                f"Claim: {claim.subject_entity_id} {claim.predicate} {claim.object_value}\n"
                f"Current confidence: {claim.confidence}"
            )

            try:
                result = await self._mass_executor.quick_query(
                    prompt, max_models=self._config.consensus_model_count,
                )
                consensus = self._compute_consensus(result.responses)

                # Publish observation through pipeline (preserves ledger integrity)
                self._publish_observation(
                    text=(
                        f"Consensus validation of claim '{claim.subject_entity_id} "
                        f"{claim.predicate} {claim.object_value}': "
                        f"{consensus['verdict']} ({consensus['agree_count']}/"
                        f"{consensus['total']} agree, "
                        f"adjusted confidence: {consensus['adjusted_confidence']:.2f})"
                    ),
                    metadata={
                        "harvest_mode": "consensus_validate",
                        "claim_id": claim.claim_id,
                        "original_confidence": claim.confidence,
                        **consensus,
                    },
                )
                self._publish("harvest.claim_validated", {
                    "claim_id": claim.claim_id,
                    **consensus,
                })
                validated += 1
            except Exception:
                log.debug("harvest.consensus_failed claim=%s", claim.claim_id)

        return {"validated_count": validated, "total_claims": len(claims)}

    # ------------------------------------------------------------------
    # Mode: adversarial_challenge
    # ------------------------------------------------------------------

    async def _mode_adversarial_challenge(self) -> dict[str, Any]:
        """Red-team high-confidence claims using best available free model."""
        claims = await self._substrate.get_claims(
            min_confidence=self._config.adversarial_confidence_threshold,
        )
        claims = [
            c for c in claims
            if c.claim_id not in self._processed_claim_ids
        ]
        claims = claims[: self._config.max_claims_per_cycle]

        model = self._pick_best_model()
        if model is None:
            return {"challenged_count": 0, "error": "no_models_available"}

        challenged = 0
        for claim in claims:
            self._processed_claim_ids.add(claim.claim_id)
            prompt = (
                f"You are a rigorous fact-checker. Challenge this claim and find "
                f"any weaknesses, counterexamples, or conditions under which it "
                f"might be false. Respond with JSON: "
                f'{{"challenge_valid": true/false, '
                f'"weaknesses": ["list of weaknesses"], '
                f'"counterexamples": ["list"], '
                f'"revised_confidence": 0.0-1.0}}.\n\n'
                f"Claim: {claim.subject_entity_id} {claim.predicate} "
                f"{claim.object_value}\n"
                f"Current confidence: {claim.confidence}"
            )

            try:
                t0 = time.monotonic()
                response = await litellm.acompletion(
                    model=model.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                content = response.choices[0].message.content or ""

                if self._discovery is not None:
                    self._discovery.record_call(
                        model.model_id, latency_ms, success=True,
                    )

                self._publish_observation(
                    text=(
                        f"Adversarial challenge of claim '{claim.subject_entity_id} "
                        f"{claim.predicate} {claim.object_value}': {content[:200]}"
                    ),
                    metadata={
                        "harvest_mode": "adversarial_challenge",
                        "claim_id": claim.claim_id,
                        "model_id": model.model_id,
                        "challenge_response": content,
                    },
                )
                self._publish("harvest.claim_challenged", {
                    "claim_id": claim.claim_id,
                    "model_id": model.model_id,
                })
                challenged += 1
            except Exception:
                log.debug("harvest.adversarial_failed claim=%s", claim.claim_id)
                if self._discovery is not None:
                    self._discovery.record_call(
                        model.model_id, 0, success=False, error="adversarial_error",
                    )

        return {"challenged_count": challenged, "total_claims": len(claims)}

    # ------------------------------------------------------------------
    # Mode: knowledge_gap
    # ------------------------------------------------------------------

    async def _mode_knowledge_gap(self) -> dict[str, Any]:
        """Research registered unknowns using free models."""
        unknowns: list[tuple[str, Any]] = []
        for goal_id, state in self._epistemic_reasoner._states.items():
            for ku in state.known_unknowns:
                unknowns.append((goal_id, ku))

        # Sort by importance: critical > high > medium > low
        importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        unknowns.sort(key=lambda x: importance_order.get(x[1].importance, 4))
        unknowns = unknowns[: self._config.max_claims_per_cycle]

        explored = 0
        for goal_id, ku in unknowns:
            prompt = (
                f"Research this question thoroughly and provide a well-sourced answer:\n\n"
                f"Question: {ku.question}\n"
                f"Context: {ku.why_unknown}\n\n"
                f"Provide a concise, factual answer."
            )

            try:
                result = await self._mass_executor.quick_query(
                    prompt, max_models=self._config.consensus_model_count,
                )
                successful = [r for r in result.responses if r.success]
                if successful:
                    best = max(successful, key=lambda r: len(r.response))
                    self._publish_observation(
                        text=(
                            f"Knowledge gap research for '{ku.question}': "
                            f"{best.response[:300]}"
                        ),
                        metadata={
                            "harvest_mode": "knowledge_gap",
                            "goal_id": goal_id,
                            "unknown_id": ku.unknown_id,
                            "question": ku.question,
                            "model_count": len(successful),
                        },
                    )
                    self._publish("harvest.gap_explored", {
                        "goal_id": goal_id,
                        "unknown_id": ku.unknown_id,
                        "question": ku.question,
                        "models_responded": len(successful),
                    })
                    explored += 1
            except Exception:
                log.debug("harvest.gap_failed unknown=%s", ku.unknown_id)

        return {"explored_count": explored, "total_unknowns": len(unknowns)}

    # ------------------------------------------------------------------
    # Mode: premium_sprint
    # ------------------------------------------------------------------

    async def _mode_premium_sprint(self) -> dict[str, Any]:
        """Route hard questions through temporarily-free powerful models."""
        powerful_models = self._discovery.get_available_models(
            tier="powerful", free_only=True,
        )
        if not powerful_models:
            return {"sprint_count": 0, "error": "no_powerful_models"}

        model = powerful_models[0]
        targets: list[dict[str, Any]] = []

        # Collect critical unknowns
        if self._epistemic_reasoner is not None:
            for goal_id, state in self._epistemic_reasoner._states.items():
                for ku in state.known_unknowns:
                    if ku.importance in ("critical", "high"):
                        targets.append({
                            "type": "unknown",
                            "goal_id": goal_id,
                            "question": ku.question,
                            "unknown_id": ku.unknown_id,
                        })

        # Collect very-low-confidence claims
        if self._substrate is not None:
            very_low = await self._substrate.get_claims(min_confidence=0.0)
            very_low = [
                c for c in very_low
                if c.confidence < 0.3
                and c.claim_id not in self._processed_claim_ids
            ]
            for c in very_low[:5]:
                targets.append({
                    "type": "claim",
                    "claim_id": c.claim_id,
                    "question": (
                        f"Is this true? {c.subject_entity_id} {c.predicate} "
                        f"{c.object_value} (confidence: {c.confidence})"
                    ),
                })

        targets = targets[: self._config.max_claims_per_cycle]
        sprint_count = 0

        for target in targets:
            prompt = (
                f"You are an expert analyst. Provide a thorough, well-reasoned "
                f"answer to this question:\n\n{target['question']}"
            )
            try:
                t0 = time.monotonic()
                response = await litellm.acompletion(
                    model=model.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=90,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                content = response.choices[0].message.content or ""

                if self._discovery is not None:
                    self._discovery.record_call(
                        model.model_id, latency_ms, success=True,
                    )

                if target["type"] == "claim":
                    self._processed_claim_ids.add(target["claim_id"])

                self._publish_observation(
                    text=(
                        f"Premium sprint ({model.model_id}) on "
                        f"'{target['question'][:100]}': {content[:300]}"
                    ),
                    metadata={
                        "harvest_mode": "premium_sprint",
                        "model_id": model.model_id,
                        "target": target,
                        "response": content,
                    },
                )
                self._publish("harvest.premium_sprint_executed", {
                    "model_id": model.model_id,
                    "target_type": target["type"],
                })
                sprint_count += 1
            except Exception:
                log.debug("harvest.sprint_failed target=%s", target.get("question", "")[:50])
                if self._discovery is not None:
                    self._discovery.record_call(
                        model.model_id, 0, success=False, error="sprint_error",
                    )

        return {"sprint_count": sprint_count, "total_targets": len(targets)}

    # ------------------------------------------------------------------
    # Mode: model_profile
    # ------------------------------------------------------------------

    async def _mode_model_profile(self) -> dict[str, Any]:
        """Profile untested free models with standardized test prompts."""
        all_free = self._discovery.get_available_models(free_only=True)
        unprofiled = [
            m for m in all_free if m.model_id not in self._model_profiles
        ][:3]  # Max 3 models per cycle

        profiled = 0
        for model in unprofiled:
            scores: dict[str, float] = {}
            total_latency = 0.0
            successes = 0

            for test in _PROFILE_PROMPTS:
                try:
                    t0 = time.monotonic()
                    response = await litellm.acompletion(
                        model=model.model_id,
                        messages=[{"role": "user", "content": test["prompt"]}],
                        timeout=30,
                    )
                    latency_ms = (time.monotonic() - t0) * 1000
                    content = response.choices[0].message.content or ""

                    score = self._score_profile_response(
                        test["category"], content, test["expected_answer"],
                    )
                    scores[test["category"]] = score
                    total_latency += latency_ms
                    successes += 1

                    if self._discovery is not None:
                        self._discovery.record_call(
                            model.model_id, latency_ms, success=True,
                        )
                except Exception:
                    scores[test["category"]] = 0.0
                    if self._discovery is not None:
                        self._discovery.record_call(
                            model.model_id, 0, success=False, error="profile_error",
                        )

            profile = {
                "model_id": model.model_id,
                "scores": scores,
                "overall_score": (
                    sum(scores.values()) / len(scores) if scores else 0.0
                ),
                "avg_latency_ms": (
                    total_latency / successes if successes else 0.0
                ),
                "tests_passed": successes,
                "profiled_at": datetime.now(UTC).isoformat(),
            }
            self._model_profiles[model.model_id] = profile

            self._publish("harvest.model_profiled", {
                "model_id": model.model_id,
                "overall_score": profile["overall_score"],
                "tests_passed": successes,
                "avg_latency_ms": profile["avg_latency_ms"],
            })
            profiled += 1

        return {"profiled_count": profiled, "total_unprofiled": len(unprofiled)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_best_model(self):
        """Pick the best available free model based on profiles."""
        if self._discovery is None:
            return None
        models = self._discovery.get_available_models(free_only=True)
        if not models:
            return None

        # Prefer profiled models with high scores
        def sort_key(m):
            profile = self._model_profiles.get(m.model_id)
            if profile:
                return profile["overall_score"]
            return 0.5  # Default score for unprofiled

        models.sort(key=sort_key, reverse=True)
        return models[0]

    @staticmethod
    def _compute_consensus(responses) -> dict[str, Any]:
        """Compute majority vote from model responses."""
        agree_count = 0
        disagree_count = 0
        confidences: list[float] = []

        for r in responses:
            if not r.success:
                continue
            text = r.response.lower()
            # Try JSON parsing first
            try:
                data = json.loads(r.response)
                verdict = data.get("verdict", "").lower()
                conf = float(data.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError, TypeError):
                # Fall back to text matching
                verdict = "agree" if "agree" in text else "disagree"
                conf = 0.5

            if verdict == "agree":
                agree_count += 1
            else:
                disagree_count += 1
            confidences.append(conf)

        total = agree_count + disagree_count
        if total == 0:
            return {
                "verdict": "inconclusive",
                "agree_count": 0,
                "disagree_count": 0,
                "total": 0,
                "adjusted_confidence": 0.0,
            }

        verdict = "agree" if agree_count > disagree_count else "disagree"
        majority_ratio = max(agree_count, disagree_count) / total
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        adjusted = majority_ratio * avg_confidence

        return {
            "verdict": verdict,
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "total": total,
            "adjusted_confidence": round(adjusted, 4),
        }

    @staticmethod
    def _score_profile_response(
        category: str, response: str, expected: str,
    ) -> float:
        """Heuristic scoring for profile test responses."""
        if not response.strip():
            return 0.0

        if category == "reasoning":
            return 1.0 if expected.lower() in response.lower() else 0.2

        if category == "factual":
            return 1.0 if expected.lower() in response.lower() else 0.0

        if category == "creative":
            lines = [ln for ln in response.strip().splitlines() if ln.strip()]
            return 1.0 if len(lines) >= 3 else 0.5 if len(lines) >= 1 else 0.0

        if category == "structured_output":
            try:
                data = json.loads(response.strip())
                if isinstance(data, dict):
                    return 1.0
            except (json.JSONDecodeError, ValueError):
                # Check if JSON is embedded in text
                for line in response.splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            json.loads(line)
                            return 0.7
                        except (json.JSONDecodeError, ValueError):
                            pass
            return 0.0

        if category == "instruction_following":
            # Should be just numbers separated by commas
            stripped = response.strip()
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                return 1.0
            return 0.3 if any(c.isdigit() for c in stripped) else 0.0

        return 0.5

    def _publish(self, topic: str, payload: dict) -> None:
        """Publish an event to the bus."""
        if self._bus:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="harvest_service",
                    payload=payload,
                )
            )

    def _publish_observation(self, text: str, metadata: dict | None = None) -> None:
        """Publish an observation through the structured pipeline."""
        if self._bus:
            self._bus.publish(
                Envelope(
                    topic="observations.structured",
                    source_service_id="harvest_service",
                    payload={"text": text, **(metadata or {})},
                )
            )

    def status(self) -> dict:
        """Return service status."""
        return {
            "running": self._running,
            "cycles_completed": self._cycles_completed,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "last_mode": self._last_mode,
            "poll_interval_seconds": self._config.poll_interval_seconds,
            "models_profiled": len(self._model_profiles),
            "claims_processed": len(self._processed_claim_ids),
        }

    # ------------------------------------------------------------------
    # Bus event handlers
    # ------------------------------------------------------------------

    async def _on_models_discovered(self, envelope: Envelope) -> None:
        """Handle new model discovery — clear stale profiles for gone models."""
        models = envelope.payload.get("models", [])
        if isinstance(models, list):
            active_ids = {
                m.get("model_id") or m for m in models if isinstance(m, (str, dict))
            }
            stale = [mid for mid in self._model_profiles if mid not in active_ids]
            for mid in stale:
                del self._model_profiles[mid]

    async def _on_tiers_updated(self, envelope: Envelope) -> None:
        """Handle tier updates — log for observability."""
        log.debug(
            "harvest.tiers_updated changes=%s",
            envelope.payload.get("changes", []),
        )

    # ------------------------------------------------------------------
    # Bus helpers
    # ------------------------------------------------------------------

    async def _maybe_subscribe(self, topic: str, handler) -> None:
        result = self._bus.subscribe(topic, handler)
        if asyncio.iscoroutine(result):
            await result

    async def _maybe_unsubscribe(self, topic: str, handler) -> None:
        result = self._bus.unsubscribe(topic, handler)
        if asyncio.iscoroutine(result):
            await result
