"""Explicit relationship decision policy after calibration and assignment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from mapel_linkage.assignment import AssignmentEdgeBatch, AssignmentResult, pair_digest
from mapel_linkage.configuration.models import DecisionPolicyConfig, OutputField
from mapel_linkage.domain.errors import DecisionPolicyError

RelationshipStatus = Literal["confirmed", "review_required", "unresolved", "no_match"]
_REVIEW_REASON_CODES = frozenset(
    {
        "candidate_search_incomplete",
        "candidate_search_truncated",
        "calibration_invalid",
        "insufficient_data_quality",
        "anchor_conflict",
        "model_disagreement",
        "assignment_conflict",
        "review_no_match_with_plausible_candidate",
        "review_probability_region",
        "unresolved_no_match_margin",
        "unresolved_insufficient_probability",
    }
)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _relationship_identifier(
    *,
    run_id: str,
    source_record_ref: str,
    target_record_ref: str | None,
    relationship_status: RelationshipStatus,
    decision_rule_id: str,
) -> str:
    return _digest(
        {
            "run_id": run_id,
            "source": source_record_ref,
            "target": target_record_ref or "NO_MATCH",
            "status": relationship_status,
            "decision_rule_id": decision_rule_id,
        }
    )


@dataclass(frozen=True, slots=True, repr=False)
class DecisionEvidence:
    source_dataset_id: str
    target_dataset_id: str
    source_record_ref: str = field(repr=False)
    assigned_target_ref: str | None = field(repr=False)
    assigned_pair_digest: str | None = field(repr=False)
    selected_no_match: bool
    calibrated_probability: float | None
    candidate_rank: int | None
    top_probability: float
    second_probability: float
    probability_margin: float
    candidate_search_complete: bool
    candidate_search_truncated: bool
    calibration_valid: bool
    data_quality_sufficient: bool
    anchor_conflict: bool
    model_disagreement: bool
    assignment_changed_top1: bool
    anchor_rule_ids: tuple[str, ...] = ()
    candidate_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for probability in (self.top_probability, self.second_probability, self.probability_margin):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise DecisionPolicyError("ML-DECISION-001", "Decision probabilities are invalid.")
        if self.second_probability > self.top_probability or not math.isclose(
            self.probability_margin,
            max(0.0, self.top_probability - self.second_probability),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise DecisionPolicyError(
                "ML-DECISION-002", "Decision probability margin is inconsistent."
            )
        if self.selected_no_match and any(
            value is not None
            for value in (
                self.assigned_target_ref,
                self.assigned_pair_digest,
                self.calibrated_probability,
                self.candidate_rank,
            )
        ):
            raise DecisionPolicyError(
                "ML-DECISION-003", "No-match evidence contains a real target."
            )
        if not self.selected_no_match and (
            self.assigned_target_ref is None
            or self.assigned_pair_digest is None
            or self.calibrated_probability is None
            or self.candidate_rank is None
        ):
            raise DecisionPolicyError(
                "ML-DECISION-004", "Assigned relationship evidence is incomplete."
            )
        if not self.selected_no_match:
            assert self.assigned_target_ref is not None
            assert self.assigned_pair_digest is not None
            assert self.calibrated_probability is not None
            assert self.candidate_rank is not None
            if (
                self.assigned_pair_digest
                != pair_digest(self.source_record_ref, self.assigned_target_ref)
                or not math.isfinite(self.calibrated_probability)
                or not 0.0 <= self.calibrated_probability <= 1.0
                or self.calibrated_probability > self.top_probability
                or self.candidate_rank < 1
            ):
                raise DecisionPolicyError(
                    "ML-DECISION-007", "Assigned relationship evidence is inconsistent."
                )
        if self.candidate_search_complete and self.candidate_search_truncated:
            raise DecisionPolicyError(
                "ML-DECISION-008", "A truncated candidate search cannot be marked complete."
            )
        if any(not item for item in (*self.anchor_rule_ids, *self.candidate_rule_ids)):
            raise DecisionPolicyError("ML-DECISION-009", "A decision rule identifier is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class RelationshipDecision:
    relationship_id: str
    source_dataset_id: str
    target_dataset_id: str
    source_record_ref: str = field(repr=False)
    target_record_ref: str | None = field(repr=False)
    relationship_status: RelationshipStatus
    model_family: str
    model_version: str
    calibrated_probability: float | None
    candidate_rank: int | None
    probability_margin: float
    decision_rule_id: str
    assignment_method: str
    assignment_constraint: str
    anchor_rule_ids: tuple[str, ...]
    candidate_rule_ids: tuple[str, ...]
    run_id: str
    configuration_digest: str
    feature_schema_digest: str
    non_sensitive_provenance: tuple[tuple[str, str], ...]
    created_at: datetime
    review_reason_codes: tuple[str, ...] = ()
    decision_authority: Literal["policy_classification"] = "policy_classification"
    merge_authority: Literal["none"] = "none"

    def __post_init__(self) -> None:
        digests = (
            self.relationship_id,
            self.configuration_digest,
            self.feature_schema_digest,
        )
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in digests
        ):
            raise DecisionPolicyError("ML-DECISION-012", "Decision provenance is invalid.")
        if not self.run_id or len(self.run_id) > 64:
            raise DecisionPolicyError("ML-DECISION-012", "Decision provenance is invalid.")
        if self.relationship_id != _relationship_identifier(
            run_id=self.run_id,
            source_record_ref=self.source_record_ref,
            target_record_ref=self.target_record_ref,
            relationship_status=self.relationship_status,
            decision_rule_id=self.decision_rule_id,
        ):
            raise DecisionPolicyError("ML-DECISION-017", "Decision identity is inconsistent.")
        if self.relationship_status == "no_match" and any(
            value is not None
            for value in (
                self.target_record_ref,
                self.calibrated_probability,
                self.candidate_rank,
            )
        ):
            raise DecisionPolicyError("ML-DECISION-013", "A no-match decision is inconsistent.")
        has_target = self.target_record_ref is not None
        if (
            has_target != (self.calibrated_probability is not None)
            or has_target != (self.candidate_rank is not None)
            or (self.relationship_status == "confirmed" and not has_target)
        ):
            raise DecisionPolicyError("ML-DECISION-019", "Decision evidence is incomplete.")
        if self.calibrated_probability is not None and (
            not math.isfinite(self.calibrated_probability)
            or not 0.0 <= self.calibrated_probability <= 1.0
        ):
            raise DecisionPolicyError("ML-DECISION-014", "Decision probability is invalid.")
        if (
            not math.isfinite(self.probability_margin)
            or not 0.0 <= self.probability_margin <= 1.0
            or (self.candidate_rank is not None and self.candidate_rank < 1)
        ):
            raise DecisionPolicyError("ML-DECISION-014", "Decision probability is invalid.")
        if (
            len(set(self.review_reason_codes)) != len(self.review_reason_codes)
            or not set(self.review_reason_codes).issubset(_REVIEW_REASON_CODES)
            or (
                self.relationship_status in {"review_required", "unresolved"}
                and not self.review_reason_codes
            )
        ):
            raise DecisionPolicyError("ML-DECISION-015", "Decision review reasons are invalid.")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise DecisionPolicyError("ML-DECISION-016", "Decision timestamp is invalid.")
        expected_provenance_keys = {
            "candidate_search_complete",
            "calibration_valid",
            "assignment_changed_top1",
        }
        provenance = dict(self.non_sensitive_provenance)
        if (
            len(provenance) != len(self.non_sensitive_provenance)
            or set(provenance) != expected_provenance_keys
            or any(value not in {"true", "false"} for value in provenance.values())
        ):
            raise DecisionPolicyError("ML-DECISION-018", "Decision public provenance is invalid.")

    def safe_summary(self) -> dict[str, str | float | int | None]:
        return {
            "relationship_id": self.relationship_id,
            "relationship_status": self.relationship_status,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "calibrated_probability": self.calibrated_probability,
            "candidate_rank": self.candidate_rank,
            "probability_margin": self.probability_margin,
            "decision_rule_id": self.decision_rule_id,
            "assignment_method": self.assignment_method,
            "run_id": self.run_id,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }

    def restricted_mapping(self, permitted_fields: tuple[OutputField, ...]) -> dict[str, object]:
        values: dict[str, object] = {
            "relationship_id": self.relationship_id,
            "source_dataset_id": self.source_dataset_id,
            "target_dataset_id": self.target_dataset_id,
            "source_record_ref": self.source_record_ref,
            "target_record_ref": self.target_record_ref,
            "relationship_status": self.relationship_status,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "calibrated_probability": self.calibrated_probability,
            "candidate_rank": self.candidate_rank,
            "probability_margin": self.probability_margin,
            "decision_rule_id": self.decision_rule_id,
            "assignment_method": self.assignment_method,
            "assignment_constraint": self.assignment_constraint,
            "anchor_rule_ids": list(self.anchor_rule_ids),
            "candidate_rule_ids": list(self.candidate_rule_ids),
            "review_reason_codes": list(self.review_reason_codes),
            "run_id": self.run_id,
            "configuration_digest": self.configuration_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "non_sensitive_provenance": dict(self.non_sensitive_provenance),
            "created_at": self.created_at.isoformat(),
        }
        return {field: values[field] for field in permitted_fields}


class DecisionEvidenceBuilder:
    """Combine assignment and candidate evidence without making decisions."""

    @staticmethod
    def build(
        *,
        candidates: AssignmentEdgeBatch,
        assignment: AssignmentResult,
        source_dataset_id: str,
        target_dataset_id: str,
        anchor_conflict_sources: frozenset[str] = frozenset(),
        model_disagreement_sources: frozenset[str] = frozenset(),
        insufficient_data_sources: frozenset[str] = frozenset(),
        anchor_rules_by_source: dict[str, tuple[str, ...]] | None = None,
        candidate_rules_by_pair_digest: dict[str, tuple[str, ...]] | None = None,
    ) -> tuple[DecisionEvidence, ...]:
        if (
            assignment.source_record_count != len(candidates.source_record_keys)
            or assignment.candidate_pair_count != candidates.candidate_pair_count
        ):
            raise DecisionPolicyError(
                "ML-DECISION-010", "Assignment and candidate evidence coverage differs."
            )
        probabilities_by_source: dict[str, list[tuple[float, str, int, str]]] = {
            source: [] for source in candidates.source_record_keys
        }
        for index, (source, target) in enumerate(candidates.pair_references):
            probabilities_by_source[source].append(
                (
                    float(candidates.probabilities[index]),
                    target,
                    int(candidates.candidate_ranks[index]),
                    candidates.pair_digests[index],
                )
            )
        assignment_by_source = {item.source_record_key: item for item in assignment.assignments}
        if set(assignment_by_source) != set(candidates.source_record_keys):
            raise DecisionPolicyError(
                "ML-DECISION-010", "Assignment and candidate evidence coverage differs."
            )
        candidates_by_digest = {
            digest: (
                source,
                target,
                float(candidates.probabilities[index]),
                int(candidates.candidate_ranks[index]),
            )
            for index, ((source, target), digest) in enumerate(
                zip(candidates.pair_references, candidates.pair_digests, strict=True)
            )
        }
        output: list[DecisionEvidence] = []
        for source in sorted(candidates.source_record_keys):
            ordered = sorted(
                probabilities_by_source[source],
                key=lambda item: (-item[0], item[3]),
            )
            top = ordered[0][0] if ordered else 0.0
            second = ordered[1][0] if len(ordered) > 1 else 0.0
            assigned = assignment_by_source[source]
            if not assigned.selected_no_match:
                candidate = candidates_by_digest.get(assigned.pair_digest or "")
                if candidate is None or candidate != (
                    source,
                    assigned.target_record_key,
                    assigned.calibrated_probability,
                    assigned.candidate_rank,
                ):
                    raise DecisionPolicyError(
                        "ML-DECISION-011",
                        "An assigned edge is not supported by candidate evidence.",
                    )
            pair_rules: tuple[str, ...] = ()
            if assigned.pair_digest is not None and candidate_rules_by_pair_digest is not None:
                pair_rules = candidate_rules_by_pair_digest.get(assigned.pair_digest, ())
            output.append(
                DecisionEvidence(
                    source_dataset_id=source_dataset_id,
                    target_dataset_id=target_dataset_id,
                    source_record_ref=source,
                    assigned_target_ref=assigned.target_record_key,
                    assigned_pair_digest=assigned.pair_digest,
                    selected_no_match=assigned.selected_no_match,
                    calibrated_probability=assigned.calibrated_probability,
                    candidate_rank=assigned.candidate_rank,
                    top_probability=top,
                    second_probability=second,
                    probability_margin=max(0.0, top - second),
                    candidate_search_complete=candidates.candidate_search_complete,
                    candidate_search_truncated=candidates.candidate_search_truncated,
                    calibration_valid=True,
                    data_quality_sufficient=source not in insufficient_data_sources,
                    anchor_conflict=source in anchor_conflict_sources,
                    model_disagreement=source in model_disagreement_sources,
                    assignment_changed_top1=assigned.changed_from_independent_top1,
                    anchor_rule_ids=(
                        ()
                        if anchor_rules_by_source is None
                        else anchor_rules_by_source.get(source, ())
                    ),
                    candidate_rule_ids=pair_rules,
                )
            )
        return tuple(output)


class RelationshipDecisionPolicy:
    """Classify one explicit relationship outcome without merging records."""

    @staticmethod
    def classify(
        evidence: DecisionEvidence,
        policy: DecisionPolicyConfig,
        *,
        model_family: str,
        model_version: str,
        assignment_method: str,
        assignment_constraint: str,
        run_id: str,
        configuration_digest: str,
        feature_schema_digest: str,
        created_at: datetime | None = None,
    ) -> RelationshipDecision:
        reasons: list[str] = []
        if not evidence.candidate_search_complete:
            reasons.append("candidate_search_incomplete")
        if evidence.candidate_search_truncated:
            reasons.append("candidate_search_truncated")
        if not evidence.calibration_valid:
            reasons.append("calibration_invalid")
        if not evidence.data_quality_sufficient:
            reasons.append("insufficient_data_quality")
        if evidence.anchor_conflict:
            reasons.append("anchor_conflict")
        if evidence.model_disagreement:
            reasons.append("model_disagreement")
        if evidence.assignment_changed_top1:
            reasons.append("assignment_conflict")

        structural_failure = any(
            reason in reasons
            for reason in (
                "candidate_search_incomplete",
                "candidate_search_truncated",
                "calibration_invalid",
                "insufficient_data_quality",
            )
        )
        if structural_failure:
            status: RelationshipStatus = "unresolved"
            rule = "unresolved_structural_evidence"
        elif evidence.selected_no_match:
            if (
                evidence.top_probability >= policy.review_required.minimum_probability
                or evidence.assignment_changed_top1
                or evidence.anchor_conflict
                or evidence.model_disagreement
            ):
                status = "review_required"
                rule = "review_no_match_with_plausible_candidate"
            elif evidence.top_probability <= policy.no_match.maximum_top_probability:
                status = "no_match"
                rule = "no_match_explicit_assignment"
            else:
                status = "unresolved"
                rule = "unresolved_no_match_margin"
        else:
            probability = evidence.calibrated_probability
            if probability is None:
                raise DecisionPolicyError(
                    "ML-DECISION-005", "A real assignment lacks probability evidence."
                )
            if (
                evidence.anchor_conflict
                or evidence.model_disagreement
                or evidence.assignment_changed_top1
            ):
                status = "review_required"
                rule = "review_conflicting_evidence"
            elif (
                probability >= policy.confirmed.minimum_probability
                and evidence.probability_margin >= policy.confirmed.minimum_probability_margin
            ):
                status = "confirmed"
                rule = "confirmed_calibrated_assigned_margin"
            elif probability >= policy.review_required.minimum_probability:
                status = "review_required"
                rule = "review_probability_region"
            else:
                status = "unresolved"
                rule = "unresolved_insufficient_probability"

        if status in {"review_required", "unresolved"} and not reasons:
            reasons.append(rule)

        relationship_id = _relationship_identifier(
            run_id=run_id,
            source_record_ref=evidence.source_record_ref,
            target_record_ref=evidence.assigned_target_ref,
            relationship_status=status,
            decision_rule_id=rule,
        )
        return RelationshipDecision(
            relationship_id=relationship_id,
            source_dataset_id=evidence.source_dataset_id,
            target_dataset_id=evidence.target_dataset_id,
            source_record_ref=evidence.source_record_ref,
            target_record_ref=evidence.assigned_target_ref,
            relationship_status=status,
            model_family=model_family,
            model_version=model_version,
            calibrated_probability=evidence.calibrated_probability,
            candidate_rank=evidence.candidate_rank,
            probability_margin=evidence.probability_margin,
            decision_rule_id=rule,
            assignment_method=assignment_method,
            assignment_constraint=assignment_constraint,
            anchor_rule_ids=evidence.anchor_rule_ids,
            candidate_rule_ids=evidence.candidate_rule_ids,
            run_id=run_id,
            configuration_digest=configuration_digest,
            feature_schema_digest=feature_schema_digest,
            non_sensitive_provenance=(
                ("candidate_search_complete", str(evidence.candidate_search_complete).lower()),
                ("calibration_valid", str(evidence.calibration_valid).lower()),
                ("assignment_changed_top1", str(evidence.assignment_changed_top1).lower()),
            ),
            created_at=created_at or datetime.now(UTC),
            review_reason_codes=tuple(reasons),
        )

    @classmethod
    def classify_all(
        cls,
        evidence: tuple[DecisionEvidence, ...],
        policy: DecisionPolicyConfig,
        *,
        model_family: str,
        model_version: str,
        assignment_method: str,
        assignment_constraint: str,
        run_id: str,
        configuration_digest: str,
        feature_schema_digest: str,
        created_at: datetime | None = None,
    ) -> tuple[RelationshipDecision, ...]:
        decisions = tuple(
            cls.classify(
                item,
                policy,
                model_family=model_family,
                model_version=model_version,
                assignment_method=assignment_method,
                assignment_constraint=assignment_constraint,
                run_id=run_id,
                configuration_digest=configuration_digest,
                feature_schema_digest=feature_schema_digest,
                created_at=created_at,
            )
            for item in evidence
        )
        if len(decisions) != len(evidence) or any(
            decision.relationship_status
            not in {"confirmed", "review_required", "unresolved", "no_match"}
            for decision in decisions
        ):
            raise DecisionPolicyError(
                "ML-DECISION-006", "Decision classification is not exhaustive."
            )
        return decisions
