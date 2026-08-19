from __future__ import annotations

from mapel_linkage.capabilities import (
    ComponentStatus,
    RuntimeVerificationStatus,
    WorkflowStatus,
    capabilities,
    capability_matrix_markdown,
    capability_summary,
)
from tests.helpers import ROOT


def test_capability_registry_is_unique_and_never_claims_operational_validation() -> None:
    registered = capabilities()
    identifiers = [item.capability_id for item in registered]

    assert len(identifiers) == len(set(identifiers))
    assert all(
        item.safe_summary()["operational_validation"] == "not_established" for item in registered
    )
    assert all(item.safe_summary()["merge_authority"] == "none" for item in registered)


def test_capability_registry_distinguishes_components_from_workflows() -> None:
    by_id = {item.capability_id: item for item in capabilities()}

    assert by_id["xgboost_pair_classifier"].workflow_status is WorkflowStatus.INTEGRATED
    assert by_id["adjudication_audit_ledger"].component_status is ComponentStatus.IMPLEMENTED
    assert by_id["adjudication_audit_ledger"].workflow_status is WorkflowStatus.INTEGRATED
    assert by_id["link_and_dedupe"].component_status is ComponentStatus.IMPLEMENTED
    assert by_id["approved_recipe_inference"].component_status is ComponentStatus.IMPLEMENTED
    assert by_id["approved_recipe_inference"].workflow_status is WorkflowStatus.INTEGRATED
    assert by_id["splink_native_model_lifecycle"].component_status is ComponentStatus.PARTIAL
    assert by_id["splink_native_model_lifecycle"].workflow_status is WorkflowStatus.NOT_INTEGRATED


def test_optional_model_capabilities_require_all_models_ci() -> None:
    by_id = {item.capability_id: item for item in capabilities()}

    for capability_id in (
        "lightgbm_pair_classifier",
        "lightgbm_candidate_ranker",
        "pytorch_tabular_matcher",
    ):
        assert by_id[capability_id].runtime_verification is RuntimeVerificationStatus.ALL_MODELS_CI


def test_capability_summary_is_aggregate_only() -> None:
    summary = capability_summary()

    assert summary["capability_count"] == len(capabilities())
    assert summary["integrated_synthetic_workflow"] == "two_source_link_only_one_to_one"
    assert summary["operational_validation"] == "not_established"
    assert summary["merge_authority"] == "none"
    assert "record" not in repr(summary).lower()


def test_generated_capability_matrix_is_current() -> None:
    committed = (ROOT / "docs" / "CAPABILITY_MATRIX.md").read_text(encoding="utf-8")

    assert committed == capability_matrix_markdown()
