"""Auditable capability status for Linkage Engine.

The registry deliberately separates code presence, workflow integration, runtime
verification, and operational validation. A component must not be described as an
integrated platform workflow merely because source files and unit tests exist.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ComponentStatus(StrEnum):
    """Implementation state of a bounded component."""

    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    PLANNED = "planned"


class WorkflowStatus(StrEnum):
    """Whether a component is reachable from an approved orchestrated workflow."""

    INTEGRATED = "workflow_integrated"
    COMPONENT_ONLY = "component_only"
    NOT_INTEGRATED = "not_integrated"


class RuntimeVerificationStatus(StrEnum):
    """CI envelope that executes the component's runtime path."""

    CORE_CI = "core_ci"
    ALL_MODELS_CI = "all_models_ci"
    NOT_VERIFIED = "not_verified"


@dataclass(frozen=True, slots=True)
class Capability:
    """A privacy-safe capability declaration."""

    capability_id: str
    title: str
    milestone: str
    component_status: ComponentStatus
    workflow_status: WorkflowStatus
    runtime_verification: RuntimeVerificationStatus
    notes: str

    def safe_summary(self) -> dict[str, str]:
        """Return machine-readable status without row-level or local material."""
        return {
            "capability_id": self.capability_id,
            "title": self.title,
            "milestone": self.milestone,
            "component_status": self.component_status.value,
            "workflow_status": self.workflow_status.value,
            "runtime_verification": self.runtime_verification.value,
            "operational_validation": "not_established",
            "decision_authority": "none",
            "merge_authority": "none",
            "notes": self.notes,
        }


_CAPABILITIES: Final[tuple[Capability, ...]] = (
    Capability(
        "fellegi_sunter_reference",
        "Fellegi-Sunter reference model",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Package-owned deterministic reference oracle; evidence-only with no decision authority.",
    ),
    Capability(
        "xgboost_pair_classifier",
        "XGBoost pair classifier",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Eligible verified labels only; uncalibrated scores remain evidence-only.",
    ),
    Capability(
        "xgboost_candidate_ranker",
        "XGBoost candidate ranker",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Ranking-only authority; it cannot emit a relationship status.",
    ),
    Capability(
        "sigmoid_calibration",
        "Sigmoid probability calibration",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Fits only on the protected calibration partition.",
    ),
    Capability(
        "isotonic_calibration",
        "Isotonic probability calibration",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Fits only on the protected calibration partition.",
    ),
    Capability(
        "beta_calibration",
        "Beta probability calibration",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Configurable alternative using the protected calibration partition.",
    ),
    Capability(
        "one_to_one_assignment",
        "One-to-one assignment with explicit no-match",
        "M2",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "OR-Tools is the primary solver; SciPy provides a small-problem reference.",
    ),
    Capability(
        "adjudication_audit_ledger",
        "Adjudication audit ledger, consensus, and label promotion",
        "M3",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Immutable append-only audit ledger, multi-reviewer consensus, and "
        "label promotion workflow integrated.",
    ),
    Capability(
        "active_learning_queue",
        "Active-learning review ordering",
        "M3",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Active-learning review ordering across uncertainty, margin, committee, and hybrid modes.",
    ),
    Capability(
        "many_to_one_assignment",
        "Many-to-one assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Greedy many-to-one assignment is CLI-integrated only in the exact generated-synthetic "
        "I1C link_only combination; operational dispatch is not established.",
    ),
    Capability(
        "one_to_many_assignment",
        "One-to-many assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Greedy one-to-many assignment is CLI-integrated only in the exact generated-synthetic "
        "I1C link_only combination; operational dispatch is not established.",
    ),
    Capability(
        "unconstrained_assignment",
        "Unconstrained assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Threshold-based unconstrained assignment is CLI-integrated only for the exact "
        "generated-synthetic I1C link_only and dedupe_only combinations.",
    ),
    Capability(
        "single_source_deduplication",
        "Single-source deduplication",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Canonical same-table pairs and aggregate clustering are CLI-integrated only for the "
        "exact generated-synthetic I1C dedupe_only and link_and_dedupe combinations.",
    ),
    Capability(
        "link_and_dedupe",
        "Combined link-and-dedupe mode",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Two-source linkage plus two intra-source clustering surfaces is CLI-integrated only "
        "for generated-synthetic I1C link_and_dedupe with one_to_one assignment.",
    ),
    Capability(
        "configuration_driven_linkage_modes",
        "Configuration-driven synthetic linkage modes",
        "I1C",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Generated-synthetic CLI dispatch is allow-listed to link_only with many_to_one, "
        "one_to_many, or unconstrained assignment; dedupe_only with unconstrained assignment; "
        "and link_and_dedupe with one_to_one assignment. Operational validation is not "
        "established, no arbitrary or real-data mode dispatch is authorized, and strict "
        "least-privilege attestation data-access isolation is not established.",
    ),
    Capability(
        "lightgbm_pair_classifier",
        "LightGBM pair-classifier challenger",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Configured synthetic tournament, protected selection/calibration, persisted reload, "
        "and recipe-bound replay execute in all-models CI.",
    ),
    Capability(
        "lightgbm_candidate_ranker",
        "LightGBM candidate ranker",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Configured source-query execution is recipe-replayable; target-query candidates are "
        "trained and reported but cannot be silently reinterpreted for source assignment.",
    ),
    Capability(
        "stacking_ensemble",
        "Stacking ensemble meta-learner",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Protected meta-model workflow, tournament selection, and out-of-fold stacking integrated.",
    ),
    Capability(
        "pytorch_tabular_matcher",
        "PyTorch tabular pair matcher",
        "M6",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Configured deterministic CPU training, protected tournament selection, persisted reload, "
        "and recipe-bound replay are integrated; it has no raw-text or identity authority.",
    ),
    Capability(
        "multi_source_entity_resolution",
        "Multi-source entity resolution",
        "M7",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Multi-source N-dataset entity resolution and global crosswalk workflow integrated.",
    ),
    Capability(
        "correlation_clustering",
        "Cannot-link correlation clustering",
        "M7",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Strict cannot-link enforcement and violation reporting are implemented.",
    ),
    Capability(
        "constrained_agglomerative_clustering",
        "Constrained agglomerative clustering",
        "M7",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Cluster merges preserve cannot-link and configured capacity boundaries.",
    ),
    Capability(
        "bcubed_cluster_metrics",
        "BCubed cluster evaluation metrics",
        "M7",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "BCubed precision, recall, F1, purity, and constraint diagnostics are available.",
    ),
    Capability(
        "splink_native_model_lifecycle",
        "Full native Splink model lifecycle",
        "I1",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Pinned Splink fit, canonical JSON reload, bounded candidate parity, and scoring are "
        "integrated as uncalibrated evidence only; operational validity is not established.",
    ),
    Capability(
        "configuration_driven_model_portfolio",
        "Configuration-driven all-model portfolio",
        "I1B",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Generated-synthetic native Splink baseline plus configured XGBoost, LightGBM, PyTorch, "
        "stacking, and ranking candidates with group-protected OOF evidence, validation-only "
        "selection, calibration-only fitting, locked-test evaluation, strict artifact reload, "
        "and disjoint recipe-bound replay; operational validity is not established.",
    ),
    Capability(
        "approved_recipe_inference",
        "Approved train-approve-infer workflow",
        "I1",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "The immutable recipe approval contract and approved recipe inference workflow integrated.",
    ),
    Capability(
        "stage1_linkage_strategy_advisor",
        "Stage-1 Linkage Strategy Advisor",
        "I2A",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Configuration-only profiling, hard eligibility, structural Pareto shortlisting, "
        "transparent explanations, and explicit empirical abstention.",
    ),
    Capability(
        "synthetic_benchmark_registry",
        "Synthetic Benchmark Registry",
        "B1",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Stable seed-v1 plus a versioned 64-family/280-instance advisor-v2 design, three "
        "truth-safe real benchmark adapters, prospective family partitions, deterministic "
        "shards, and append-only resume controls; the diagnostic v1 corpus completed with "
        "retained failures and corrected execution-v2 evidence remains pending.",
    ),
    Capability(
        "stage2_similarity_advisor",
        "Stage-2 Similarity & Coverage Advisor",
        "I2B",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Nearest scenario family retrieval, weighted distance computation, "
        "out-of-distribution thresholding, empirical performance distribution aggregation, "
        "and strict advisory invariants.",
    ),
    Capability(
        "stage3_meta_ranking_advisor",
        "Stage-3 Learned Meta-Ranking Advisor",
        "I2C",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Learned meta-regressor with family-disjoint fit, conformal interval calibration, "
        "locked evaluation, true-mechanism OOD exclusion, scenario-replicate-complete adapter "
        "gating, and similarity fallback.",
    ),
    Capability(
        "linkage_strategy_advisor",
        "Evidence-Based Linkage Strategy Advisor",
        "I2D",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Snapshot-bound active planning plus separately approved, digest-bound advisor-corpus "
        "shard execution and append-only evidence checks; diagnostic v1 evidence is retained, "
        "while corrected v2 advisor validation and operational validity remain unestablished.",
    ),
)


def capabilities() -> tuple[Capability, ...]:
    """Return the immutable capability registry."""
    return _CAPABILITIES


def capability_summary() -> dict[str, object]:
    """Return aggregate status counts suitable for CLI output."""
    component_counts = Counter(item.component_status.value for item in _CAPABILITIES)
    workflow_counts = Counter(item.workflow_status.value for item in _CAPABILITIES)
    runtime_counts = Counter(item.runtime_verification.value for item in _CAPABILITIES)
    return {
        "capability_count": len(_CAPABILITIES),
        "component_status_counts": dict(sorted(component_counts.items())),
        "workflow_status_counts": dict(sorted(workflow_counts.items())),
        "runtime_verification_counts": dict(sorted(runtime_counts.items())),
        "integrated_synthetic_workflow": "two_source_link_only_one_to_one_configured_portfolio",
        "integrated_synthetic_linkage_mode_combinations": (
            "link_only+many_to_one",
            "link_only+one_to_many",
            "link_only+unconstrained",
            "dedupe_only+unconstrained",
            "link_and_dedupe+one_to_one",
        ),
        "operational_validation": "not_established",
        "decision_authority": "explicit_policy_only",
        "merge_authority": "none",
    }


def capability_matrix_markdown() -> str:
    """Render the normative capability matrix deterministically."""
    lines = [
        "# Capability Matrix",
        "",
        "This matrix distinguishes four questions that must not be conflated:",
        "",
        "1. Is a bounded component present in source code?",
        "2. Is it reachable from an approved configuration-driven workflow?",
        "3. Does CI execute its real runtime dependency path?",
        "4. Has it been validated for operational use on an approved population?",
        "",
        "No current capability has established operational validation.",
        "",
        "| Capability | Milestone | Component | Workflow | Runtime verification | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for item in _CAPABILITIES:
        lines.append(
            "| "
            f"`{item.capability_id}` | {item.milestone} | {item.component_status.value} | "
            f"{item.workflow_status.value} | {item.runtime_verification.value} | "
            f"{item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Current integrated workflow",
            "",
            "The legacy complete configuration-driven workflow remains bounded to",
            "generated-synthetic two-source `link_only`, `one_to_one` execution. I1B adds the",
            "configured all-model portfolio path within that same boundary.",
            "I1C separately allow-lists exactly `link_only` with `many_to_one`, `one_to_many`,",
            "or `unconstrained`; `dedupe_only` with `unconstrained`; and `link_and_dedupe` with",
            "`one_to_one`. It is generated-synthetic only, operational validity is not",
            "established, strict least-privilege attestation data-access isolation is not",
            "established, and no arbitrary M3-M7, multi-source, or real-data dispatch is implied.",
            "",
            "## Test reporting",
            "",
            "CI must report collected, passed, failed, and skipped counts separately. A collected",
            "test is not described as passed when its optional runtime dependency was unavailable.",
            "The dedicated all-models CI job installs LightGBM and PyTorch and fails when any test",
            "is skipped.",
            "",
            "## Authority boundary",
            "",
            "- candidate retrieval does not decide identity;",
            "- pair and ranking models remain evidence-only;",
            "- assignment selects compatible edges but does not classify relationships;",
            "- only the explicit decision policy emits relationship status;",
            "- no capability has silent merge or master-record authority;",
            "- synthetic testing does not establish operational linkage validity.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "Capability",
    "ComponentStatus",
    "RuntimeVerificationStatus",
    "WorkflowStatus",
    "capabilities",
    "capability_matrix_markdown",
    "capability_summary",
]
