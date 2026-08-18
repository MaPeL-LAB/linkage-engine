from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    ScipyOneToOneAssignmentSolver,
)
from mapel_linkage.configuration.models import DecisionPolicyConfig
from mapel_linkage.decisions import (
    DecisionEvidenceBuilder,
    RelationshipDecision,
    RelationshipDecisionPolicy,
)
from mapel_linkage.domain.errors import DecisionPolicyError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def decision_policy() -> DecisionPolicyConfig:
    return DecisionPolicyConfig.model_validate(
        {
            "confirmed": {
                "minimum_probability": 0.90,
                "minimum_probability_margin": 0.15,
                "require_assignment": True,
                "require_valid_calibration": True,
            },
            "review_required": {"minimum_probability": 0.60},
            "no_match": {
                "maximum_top_probability": 0.20,
                "require_complete_candidate_search": True,
            },
            "unresolved": {"fallback": True},
        }
    )


def candidates() -> AssignmentEdgeBatch:
    pairs = (
        ("confirmed-source", "target-a"),
        ("confirmed-source", "target-b"),
        ("review-source", "target-c"),
        ("review-source", "target-d"),
        ("no-match-source", "target-e"),
        ("unresolved-source", "target-f"),
    )
    return AssignmentEdgeBatch(
        source_record_keys=(
            "confirmed-source",
            "review-source",
            "no-match-source",
            "unresolved-source",
        ),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.98, 0.40, 0.75, 0.70, 0.10, 0.55], dtype=np.float64),
        candidate_ranks=np.asarray([1, 2, 1, 2, 1, 1], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=digest("ranker"),
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )


def classify(*, truncated: bool = False) -> tuple[RelationshipDecision, ...]:
    batch = candidates()
    if truncated:
        batch = replace(
            batch,
            candidate_search_complete=False,
            candidate_search_truncated=True,
        )
    assignment = ScipyOneToOneAssignmentSolver.solve(
        batch,
        AssignmentPlan(solver="scipy_linear_sum_assignment"),
    )
    evidence = DecisionEvidenceBuilder.build(
        candidates=batch,
        assignment=assignment,
        source_dataset_id="source_a",
        target_dataset_id="source_b",
    )
    return RelationshipDecisionPolicy.classify_all(
        evidence,
        decision_policy(),
        model_family="xgboost",
        model_version="v1",
        assignment_method=assignment.solver,
        assignment_constraint=assignment.constraint,
        run_id="a" * 32,
        configuration_digest=digest("configuration"),
        feature_schema_digest=digest("features"),
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def test_policy_emits_all_four_statuses_without_merge_authority() -> None:
    decisions = classify()
    status = {decision.source_record_ref: decision.relationship_status for decision in decisions}
    assert status == {
        "confirmed-source": "confirmed",
        "review-source": "review_required",
        "no-match-source": "no_match",
        "unresolved-source": "unresolved",
    }
    assert all(decision.merge_authority == "none" for decision in decisions)
    assert all(decision.decision_authority == "policy_classification" for decision in decisions)
    assert "confirmed-source" not in repr(decisions[0])


def test_truncated_candidate_search_never_becomes_no_match() -> None:
    decisions = classify(truncated=True)
    assert {decision.relationship_status for decision in decisions} == {"unresolved"}


def test_restricted_mapping_is_exactly_allow_listed() -> None:
    decision = classify()[0]
    mapping = decision.restricted_mapping(
        ("relationship_id", "relationship_status", "calibrated_probability")
    )
    assert set(mapping) == {"relationship_id", "relationship_status", "calibrated_probability"}
    assert "source_record_ref" not in mapping


def test_restricted_mapping_supports_review_reason_allow_list() -> None:
    decision = classify(truncated=True)[0]
    mapping = decision.restricted_mapping(("relationship_id", "review_reason_codes"))
    assert mapping["review_reason_codes"] == [
        "candidate_search_incomplete",
        "candidate_search_truncated",
    ]


def test_decision_rejects_mismatched_identity_and_non_allowlisted_public_provenance() -> None:
    decision = classify()[0]
    with pytest.raises(DecisionPolicyError, match="ML-DECISION-017"):
        replace(decision, relationship_id=digest("unrelated-decision"))
    with pytest.raises(DecisionPolicyError, match="ML-DECISION-018"):
        replace(decision, non_sensitive_provenance=(("private_note", "synthetic-value"),))


def test_confirmed_decision_requires_complete_real_edge_evidence() -> None:
    decision = next(item for item in classify() if item.relationship_status == "confirmed")
    with pytest.raises(DecisionPolicyError, match="ML-DECISION-019"):
        replace(decision, calibrated_probability=None)
