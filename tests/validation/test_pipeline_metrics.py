from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from mapel_linkage.assignment import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
    ScipyOneToOneAssignmentSolver,
)
from mapel_linkage.decisions import RelationshipDecision, RelationshipStatus
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.models.ranking.contracts import RankingScoreBatch
from mapel_linkage.validation import (
    evaluate_assignment,
    evaluate_candidate_retrieval,
    evaluate_decisions,
    evaluate_ranking,
    write_aggregate_validation_report,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_candidate_retrieval_reports_recall_and_set_size() -> None:
    report = evaluate_candidate_retrieval(
        source_record_keys=("s1", "s2", "s3"),
        target_record_keys=("t1", "t2", "t3"),
        candidate_pairs=(("s1", "t1"), ("s1", "t2"), ("s2", "t2")),
        true_pairs=frozenset({("s1", "t1"), ("s2", "t2")}),
        rule_ids_by_pair={
            ("s1", "t1"): ("rule_a",),
            ("s1", "t2"): ("rule_b",),
            ("s2", "t2"): ("rule_a", "rule_b"),
        },
    )
    assert report.candidate_recall == 1.0
    assert report.zero_candidate_source_count == 1
    assert report.maximum_candidate_set_size == 2
    assert report.cartesian_reduction_fraction == pytest.approx(2 / 3)


def test_ranking_metrics_count_missing_true_candidates_in_denominator() -> None:
    pairs = (("s1", "t1"), ("s1", "t2"), ("s2", "t3"), ("s2", "t4"))
    digests = tuple(digest(f"{left}\x00{right}") for left, right in pairs)
    scores = RankingScoreBatch(
        pair_references=pairs,
        pair_digests=digests,
        query_keys=("s1", "s1", "s2", "s2"),
        scores=np.asarray([0.9, 0.1, 0.8, 0.2]),
        ranks=np.asarray([1, 2, 1, 2], dtype=np.int64),
        top_k_membership=np.asarray([1, 1, 1, 1], dtype=np.int64),
        model_id="ranker",
        model_version="v1",
        model_digest=digest("ranker"),
        query_side="source",
        top_k=2,
    )
    report = evaluate_ranking(
        scores=scores,
        true_pair_digests=frozenset({digests[0], digests[3]}),
        eligible_query_keys=("s1", "s2", "s3"),
        k_values=(1, 2),
    )
    assert report.eligible_query_count == 3
    assert report.top1_fraction == 1 / 3
    assert dict(report.recall_at_k) == {1: 1 / 3, 2: 2 / 3}


def assignment_fixture() -> AssignmentResult:
    pairs = (("s1", "t1"), ("s1", "t2"), ("s2", "t1"))
    batch = AssignmentEdgeBatch(
        source_record_keys=("s1", "s2", "s3"),
        pair_references=pairs,
        pair_digests=tuple(digest(f"{left}\x00{right}") for left, right in pairs),
        probabilities=np.asarray([0.95, 0.80, 0.94]),
        candidate_ranks=np.asarray([1, 2, 1], dtype=np.int64),
        source_model_id="model",
        source_model_version="v1",
        calibrator_digest=digest("calibrator"),
        ranking_model_digest=None,
        candidate_search_complete=True,
        candidate_search_truncated=False,
    )
    return ScipyOneToOneAssignmentSolver.solve(
        batch, AssignmentPlan(solver="scipy_linear_sum_assignment")
    )


def test_assignment_metrics_include_no_match_and_constraint_checks() -> None:
    result = assignment_fixture()
    report = evaluate_assignment(
        assignment=result,
        true_target_by_source={"s1": "t2", "s2": "t1", "s3": None},
    )
    assert report.assignment_accuracy == 1.0
    assert report.no_match_accuracy == 1.0
    assert report.constraint_violation_count == 0


def decision(source: str, status: RelationshipStatus) -> RelationshipDecision:
    target = f"target-{source}" if status in {"confirmed", "review_required"} else None
    relationship_id = hashlib.sha256(
        json.dumps(
            {
                "run_id": "a" * 32,
                "source": source,
                "target": target or "NO_MATCH",
                "status": status,
                "decision_rule_id": "rule",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return RelationshipDecision(
        relationship_id=relationship_id,
        source_dataset_id="a",
        target_dataset_id="b",
        source_record_ref=source,
        target_record_ref=target,
        relationship_status=status,
        model_family="xgboost",
        model_version="v1",
        calibrated_probability=0.72 if target is not None else None,
        candidate_rank=1 if target is not None else None,
        probability_margin=0.0,
        decision_rule_id="rule",
        assignment_method="solver",
        assignment_constraint="one_to_one",
        anchor_rule_ids=(),
        candidate_rule_ids=(),
        run_id="a" * 32,
        configuration_digest=digest("config"),
        feature_schema_digest=digest("features"),
        non_sensitive_provenance=(
            ("candidate_search_complete", "true"),
            ("calibration_valid", "true"),
            ("assignment_changed_top1", "false"),
        ),
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        review_reason_codes=(
            ("review_probability_region",)
            if status == "review_required"
            else (("unresolved_insufficient_probability",) if status == "unresolved" else ())
        ),
    )


def test_decision_metrics_and_aggregate_report_are_row_free(tmp_path: Path) -> None:
    report = evaluate_decisions(
        (
            decision("private-a", "confirmed"),
            decision("private-b", "review_required"),
            decision("private-c", "unresolved"),
            decision("private-d", "no_match"),
        )
    )
    assert report.relationship_count == 4
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    path = write_aggregate_validation_report(
        reports={"decisions": report},
        path="artifacts/reports/evaluation.json",
        policy=policy,
    )
    text = path.read_text()
    assert "private-a" not in text
    assert "Synthetic testing establishes" in text


def test_aggregate_report_does_not_follow_predictable_temporary_symlink(
    tmp_path: Path,
) -> None:
    policy = PathPolicy.build(
        project_root=tmp_path,
        configured_input_roots=("data", "private"),
        configured_output_roots=("private", "artifacts"),
    )
    report = evaluate_decisions((decision("synthetic-source", "confirmed"),))
    report_directory = tmp_path / "artifacts" / "reports"
    report_directory.mkdir(parents=True)
    protected = tmp_path / "protected-sentinel.txt"
    protected.write_text("unchanged\n", encoding="utf-8")
    legacy_temporary = report_directory / "evaluation.json.tmp"
    legacy_temporary.symlink_to(protected)

    write_aggregate_validation_report(
        reports={"decisions": report},
        path="artifacts/reports/evaluation.json",
        policy=policy,
    )

    assert protected.read_text(encoding="utf-8") == "unchanged\n"
    assert legacy_temporary.is_symlink()
