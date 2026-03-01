"""Symbolic inference engine: derives new claims and detects inconsistencies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class InferredClaim:
    """A claim derived by symbolic inference from existing claims."""

    subject: str
    predicate: str
    object_: str
    confidence: float
    reasoning: str
    source_claim_ids: list[str] = field(default_factory=list)
    inference_type: str = ""


@dataclass
class Inconsistency:
    """A detected logical inconsistency between claims."""

    claim_ids: list[str] = field(default_factory=list)
    description: str = ""
    severity: float = 0.0


def _claim_field(claim: dict, key: str, default: Any = "") -> Any:
    """Safely extract a field from a claim dict."""
    return claim.get(key, default)


class InferenceTemplate(ABC):
    """Base class for inference templates that derive new claims from existing ones."""

    @abstractmethod
    def match(self, claims: list[dict]) -> list[InferredClaim]:
        """Apply this template to a set of claims and return inferred claims."""


class TransitiveTemplate(InferenceTemplate):
    """Transitive inference: A->B, B->C therefore A->C.

    Matches claims where the object of one claim equals the subject of another,
    and both share the same predicate, to derive a transitive conclusion.
    """

    def match(self, claims: list[dict]) -> list[InferredClaim]:
        inferred: list[InferredClaim] = []

        # Index claims by subject for efficient lookup
        by_subject: dict[str, list[dict]] = {}
        for claim in claims:
            subj = _claim_field(claim, "subject_entity_id")
            if subj:
                by_subject.setdefault(subj, []).append(claim)

        for claim_a in claims:
            obj_a = _claim_field(claim_a, "object_value")
            pred_a = _claim_field(claim_a, "predicate")
            subj_a = _claim_field(claim_a, "subject_entity_id")
            id_a = _claim_field(claim_a, "claim_id")

            if not obj_a or not pred_a:
                continue

            # Find claims where subject matches object of claim_a
            # and predicate is the same
            for claim_b in by_subject.get(obj_a, []):
                pred_b = _claim_field(claim_b, "predicate")
                obj_b = _claim_field(claim_b, "object_value")
                id_b = _claim_field(claim_b, "claim_id")

                if pred_b != pred_a:
                    continue
                if not obj_b or obj_b == subj_a:
                    continue  # Avoid trivial cycles

                # Derive: A -> C via B
                conf_a = float(_claim_field(claim_a, "confidence", 0.5))
                conf_b = float(_claim_field(claim_b, "confidence", 0.5))
                combined_confidence = conf_a * conf_b

                inferred.append(InferredClaim(
                    subject=subj_a,
                    predicate=pred_a,
                    object_=obj_b,
                    confidence=combined_confidence,
                    reasoning=(
                        f"Transitive: {subj_a} -{pred_a}-> {obj_a} "
                        f"and {obj_a} -{pred_b}-> {obj_b}, "
                        f"therefore {subj_a} -{pred_a}-> {obj_b}"
                    ),
                    source_claim_ids=[id_a, id_b],
                    inference_type="transitive",
                ))

        return inferred


class AggregateTemplate(InferenceTemplate):
    """Aggregate inference: multiple claims about the same subject -> summary claim.

    When multiple claims share the same subject and predicate, derive a summary
    claim with aggregated confidence.
    """

    def match(self, claims: list[dict]) -> list[InferredClaim]:
        inferred: list[InferredClaim] = []

        # Group claims by (subject, predicate)
        groups: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            subj = _claim_field(claim, "subject_entity_id")
            pred = _claim_field(claim, "predicate")
            if subj and pred:
                groups.setdefault((subj, pred), []).append(claim)

        for (subj, pred), group in groups.items():
            if len(group) < 2:
                continue

            # Collect unique object values and confidences
            values: list[str] = []
            confidences: list[float] = []
            claim_ids: list[str] = []

            for claim in group:
                obj = _claim_field(claim, "object_value")
                conf = float(_claim_field(claim, "confidence", 0.5))
                cid = _claim_field(claim, "claim_id")
                if obj:
                    values.append(obj)
                    confidences.append(conf)
                if cid:
                    claim_ids.append(cid)

            if not values:
                continue

            # Use the most common value, or highest-confidence value
            value_counts: dict[str, float] = {}
            for val, conf in zip(values, confidences, strict=True):
                value_counts[val] = value_counts.get(val, 0.0) + conf

            best_value = max(value_counts, key=lambda v: value_counts[v])

            # Aggregate confidence: average of supporting confidences
            supporting = [
                c for v, c in zip(values, confidences, strict=True)
                if v == best_value
            ]
            avg_confidence = sum(supporting) / len(supporting)
            # Boost slightly for having multiple sources, cap at 0.99
            boosted = min(avg_confidence * (1.0 + 0.05 * (len(supporting) - 1)), 0.99)

            inferred.append(InferredClaim(
                subject=subj,
                predicate=pred,
                object_=best_value,
                confidence=boosted,
                reasoning=(
                    f"Aggregate: {len(supporting)} claims about "
                    f"{subj}.{pred} agree on '{best_value}' "
                    f"(avg confidence {avg_confidence:.2f})"
                ),
                source_claim_ids=claim_ids,
                inference_type="aggregate",
            ))

        return inferred


class TemporalTemplate(InferenceTemplate):
    """Temporal inference: newer claim supersedes older contradictory claim.

    When two claims share the same subject and predicate but have different
    object values, the newer claim is treated as the current belief.
    """

    def match(self, claims: list[dict]) -> list[InferredClaim]:
        inferred: list[InferredClaim] = []

        # Group claims by (subject, predicate)
        groups: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            subj = _claim_field(claim, "subject_entity_id")
            pred = _claim_field(claim, "predicate")
            if subj and pred:
                groups.setdefault((subj, pred), []).append(claim)

        for (subj, pred), group in groups.items():
            if len(group) < 2:
                continue

            # Sort by created_at, newest first
            def _sort_key(c: dict) -> str:
                ts = _claim_field(c, "created_at", "")
                if isinstance(ts, datetime):
                    return ts.isoformat()
                return str(ts)

            sorted_claims = sorted(group, key=_sort_key, reverse=True)

            newest = sorted_claims[0]
            newest_obj = _claim_field(newest, "object_value")
            newest_id = _claim_field(newest, "claim_id")
            newest_conf = float(_claim_field(newest, "confidence", 0.5))
            newest_ts = _claim_field(newest, "created_at", "")

            # Check if there are older contradictory claims
            for older in sorted_claims[1:]:
                older_obj = _claim_field(older, "object_value")
                older_id = _claim_field(older, "claim_id")

                if older_obj and older_obj != newest_obj:
                    inferred.append(InferredClaim(
                        subject=subj,
                        predicate=pred,
                        object_=newest_obj,
                        confidence=newest_conf,
                        reasoning=(
                            f"Temporal: newer claim (created {newest_ts}) "
                            f"supersedes older claim. "
                            f"'{newest_obj}' replaces '{older_obj}' "
                            f"for {subj}.{pred}"
                        ),
                        source_claim_ids=[newest_id, older_id],
                        inference_type="temporal",
                    ))
                    # Only emit one temporal inference per group
                    break

        return inferred


class HypothesisTemplate(InferenceTemplate):
    """Hypothesis-based inference: hypothesis status affects related claims.

    - Claims with `hypothesis_id` metadata where hypothesis reached confirmed
      (p >= 0.95) → infer high-confidence claim
    - Claims with `hypothesis_id` where hypothesis falsified (p <= 0.05) →
      weaken supporting claims
    """

    def __init__(self, hypotheses: list[dict] | None = None) -> None:
        # hypotheses: list of dicts with hypothesis_id, current_probability, status
        self._hypotheses = {
            h["hypothesis_id"]: h
            for h in (hypotheses or [])
        }

    def match(self, claims: list[dict]) -> list[InferredClaim]:
        inferred: list[InferredClaim] = []

        for claim in claims:
            metadata = _claim_field(claim, "metadata", {})
            if not isinstance(metadata, dict):
                continue

            hyp_id = metadata.get("hypothesis_id")
            if not hyp_id or hyp_id not in self._hypotheses:
                continue

            hyp = self._hypotheses[hyp_id]
            prob = float(hyp.get("current_probability", 0.5))
            status = hyp.get("status", "active")
            claim_id = _claim_field(claim, "claim_id")
            subj = _claim_field(claim, "subject_entity_id")
            pred = _claim_field(claim, "predicate")
            obj = _claim_field(claim, "object_value")
            claim_conf = float(_claim_field(claim, "confidence", 0.5))

            if status == "confirmed" or prob >= 0.95:
                # Hypothesis confirmed → boost claim confidence
                boosted = min(claim_conf * 1.3, 0.99)
                inferred.append(InferredClaim(
                    subject=subj,
                    predicate=pred,
                    object_=obj,
                    confidence=boosted,
                    reasoning=(
                        f"Hypothesis {hyp_id} confirmed (p={prob:.2f}). "
                        f"Boosting claim confidence from {claim_conf:.2f} to {boosted:.2f}."
                    ),
                    source_claim_ids=[claim_id],
                    inference_type="hypothesis_confirmed",
                ))
            elif status == "falsified" or prob <= 0.05:
                # Hypothesis falsified → weaken claim
                weakened = max(claim_conf * 0.5, 0.01)
                inferred.append(InferredClaim(
                    subject=subj,
                    predicate=pred,
                    object_=obj,
                    confidence=weakened,
                    reasoning=(
                        f"Hypothesis {hyp_id} falsified (p={prob:.2f}). "
                        f"Weakening claim confidence from {claim_conf:.2f} to {weakened:.2f}."
                    ),
                    source_claim_ids=[claim_id],
                    inference_type="hypothesis_falsified",
                ))

        return inferred


class SymbolicInferenceEngine:
    """Runs inference templates over claims to derive new knowledge and detect
    inconsistencies."""

    def __init__(self, templates: list[InferenceTemplate] | None = None) -> None:
        if templates is not None:
            self._templates = templates
        else:
            self._templates: list[InferenceTemplate] = [
                TransitiveTemplate(),
                AggregateTemplate(),
                TemporalTemplate(),
            ]

    async def infer(self, claims: list[dict]) -> list[InferredClaim]:
        """Run all inference templates and return derived claims.

        Args:
            claims: list of claim dicts (matching Claim model fields).

        Returns:
            All inferred claims from all templates.
        """
        if not claims:
            return []

        results: list[InferredClaim] = []
        for template in self._templates:
            try:
                inferred = template.match(claims)
                results.extend(inferred)
            except Exception:
                log.exception(
                    "Inference template %s failed",
                    type(template).__name__,
                )
        log.info("Inference produced %d new claims from %d inputs", len(results), len(claims))
        return results

    async def detect_inconsistencies(self, claims: list[dict]) -> list[Inconsistency]:
        """Find contradictions among claims.

        Detects:
        1. Same subject+predicate but different object values (direct contradiction)
        2. Confidence conflicts (high-confidence claims contradicting each other)

        Args:
            claims: list of claim dicts.

        Returns:
            List of detected inconsistencies.
        """
        if not claims:
            return []

        inconsistencies: list[Inconsistency] = []

        # Group by (subject, predicate)
        groups: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            subj = _claim_field(claim, "subject_entity_id")
            pred = _claim_field(claim, "predicate")
            if subj and pred:
                groups.setdefault((subj, pred), []).append(claim)

        for (subj, pred), group in groups.items():
            # Collect distinct object values
            value_claims: dict[str, list[dict]] = {}
            for claim in group:
                obj = _claim_field(claim, "object_value")
                if obj:
                    value_claims.setdefault(obj, []).append(claim)

            if len(value_claims) < 2:
                continue

            # Direct contradiction: multiple distinct values for same subject+predicate
            all_ids: list[str] = []
            all_values: list[str] = []
            max_conf = 0.0
            for val, val_group in value_claims.items():
                all_values.append(val)
                for c in val_group:
                    cid = _claim_field(c, "claim_id")
                    conf = float(_claim_field(c, "confidence", 0.0))
                    if cid:
                        all_ids.append(cid)
                    max_conf = max(max_conf, conf)

            # Severity based on how confident the contradicting claims are
            confidences = []
            for val_group in value_claims.values():
                for c in val_group:
                    confidences.append(float(_claim_field(c, "confidence", 0.0)))

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            severity = min(avg_conf * (len(value_claims) / 2.0), 1.0)

            inconsistencies.append(Inconsistency(
                claim_ids=all_ids,
                description=(
                    f"Contradiction for {subj}.{pred}: "
                    f"conflicting values {all_values}"
                ),
                severity=severity,
            ))

        log.info(
            "Inconsistency detection found %d issues in %d claims",
            len(inconsistencies),
            len(claims),
        )
        return inconsistencies

    def get_inference_chain(self, claim_id: str, claims: list[dict]) -> list[dict]:
        """Trace the provenance chain for a claim.

        Follows source_envelope_ids and superseded_by links to reconstruct
        the chain of evidence leading to a given claim.

        Args:
            claim_id: the claim to trace.
            claims: all available claims.

        Returns:
            Ordered list of claim dicts forming the provenance chain,
            from root sources to the target claim.
        """
        # Build index by claim_id
        by_id: dict[str, dict] = {}
        for claim in claims:
            cid = _claim_field(claim, "claim_id")
            if cid:
                by_id[cid] = claim

        if claim_id not in by_id:
            return []

        # BFS backwards through source_envelope_ids and metadata references
        visited: set[str] = set()
        chain: list[dict] = []
        queue: list[str] = [claim_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            current = by_id.get(current_id)
            if current is None:
                continue

            chain.append(current)

            # Follow source_envelope_ids if they reference other claims
            source_ids = _claim_field(current, "source_envelope_ids", [])
            if isinstance(source_ids, list):
                for sid in source_ids:
                    if sid in by_id and sid not in visited:
                        queue.append(sid)

            # Follow superseded_by chain backwards
            # Check if any other claim was superseded by this one
            for other_id, other in by_id.items():
                if _claim_field(other, "superseded_by") == current_id and other_id not in visited:
                    queue.append(other_id)

            # Check metadata for source_claim_ids (from inferred claims)
            metadata = _claim_field(current, "metadata", {})
            if isinstance(metadata, dict):
                src_ids = metadata.get("source_claim_ids", [])
                if isinstance(src_ids, list):
                    for sid in src_ids:
                        if sid in by_id and sid not in visited:
                            queue.append(sid)

        # Reverse so root sources come first, target claim last
        chain.reverse()
        return chain
