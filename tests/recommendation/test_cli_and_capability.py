from __future__ import annotations

import json

import pytest

from mapel_linkage.capabilities import (
    ComponentStatus,
    RuntimeVerificationStatus,
    WorkflowStatus,
    capabilities,
)
from mapel_linkage.cli.main import main
from tests.helpers import EXAMPLE_CONFIG, ROOT


def test_stage1_advisor_capability_is_integrated_without_overstating_later_stages() -> None:
    by_id = {item.capability_id: item for item in capabilities()}

    stage1 = by_id["stage1_linkage_strategy_advisor"]
    assert stage1.component_status is ComponentStatus.IMPLEMENTED
    assert stage1.workflow_status is WorkflowStatus.INTEGRATED
    assert stage1.runtime_verification is RuntimeVerificationStatus.CORE_CI

    registry = by_id["synthetic_benchmark_registry"]
    assert registry.component_status is ComponentStatus.IMPLEMENTED
    assert registry.workflow_status is WorkflowStatus.INTEGRATED

    learned = by_id["linkage_strategy_advisor"]
    assert learned.component_status is ComponentStatus.PLANNED
    assert learned.workflow_status is WorkflowStatus.NOT_INTEGRATED


def test_profile_job_cli_emits_safe_aggregate_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "profile-job",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["profile_stage"] == "preflight"
    assert payload["record_count_band"] == "not_observed"
    assert payload["contains_record_values"] is False
    assert payload["contains_source_field_names"] is False
    rendered = json.dumps(payload, sort_keys=True)
    assert "record_key_a" not in rendered
    assert str(EXAMPLE_CONFIG) not in rendered


def test_recommend_pipeline_cli_abstains_from_empirical_ranking(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "recommend-pipeline",
                "--config",
                str(EXAMPLE_CONFIG),
                "--project-root",
                str(ROOT),
                "--intent",
                "develop_new_recipe",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["coverage_status"] == "structural_only"
    assert payload["abstained_from_empirical_ranking"] is True
    assert payload["empirical_performance_claims"] == "none"
    assert payload["recommendation_authority"] == "advisory_only"
    assert payload["decision_authority"] == "none"
    assert payload["assignment_authority"] == "none"
    assert payload["merge_authority"] == "none"
    assert payload["automatic_promotion"] == "prohibited"
    assert payload["shortlist"]
    rendered = json.dumps(payload, sort_keys=True)
    assert "record_key_a" not in rendered
    assert str(EXAMPLE_CONFIG) not in rendered
