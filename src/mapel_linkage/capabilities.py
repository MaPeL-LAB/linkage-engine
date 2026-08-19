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
        "Package-owned scoring path; the full native Splink lifecycle remains partial.",
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
        "adjudication_lifecycle",
        "Adjudication import, disagreement, and label promotion",
        "M3",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Implemented below the CLI boundary; no automatic retraining is permitted.",
    ),
    Capability(
        "active_learning_queue",
        "Active-learning review ordering",
        "M3",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Ordering authority only; unknown pairs are not silently relabelled.",
    ),
    Capability(
        "many_to_one_assignment",
        "Many-to-one assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Solver and invariant tests exist; general orchestration is pending.",
    ),
    Capability(
        "one_to_many_assignment",
        "One-to-many assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Solver and invariant tests exist; general orchestration is pending.",
    ),
    Capability(
        "unconstrained_assignment",
        "Unconstrained assignment",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Pairwise selection exists without bypassing the decision-policy boundary.",
    ),
    Capability(
        "single_source_deduplication",
        "Single-source deduplication",
        "M4",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Pair canonicalisation and clustering safeguards exist; CLI integration is pending.",
    ),
    Capability(
        "link_and_dedupe",
        "Combined link-and-dedupe mode",
        "M4",
        ComponentStatus.PARTIAL,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Required primitives exist, but no complete combined execution path exists.",
    ),
    Capability(
        "lightgbm_pair_classifier",
        "LightGBM pair-classifier challenger",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Optional dependency; dedicated all-models CI must execute the runtime path.",
    ),
    Capability(
        "lightgbm_candidate_ranker",
        "LightGBM candidate ranker",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Optional dependency; dedicated all-models CI must execute the runtime path.",
    ),
    Capability(
        "stacking_ensemble",
        "Stacking ensemble meta-learner",
        "M5",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "A protected meta-model workflow and portfolio configuration remain pending.",
    ),
    Capability(
        "pytorch_tabular_matcher",
        "PyTorch tabular pair matcher",
        "M6",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.ALL_MODELS_CI,
        "Optional feature-based challenger; it has no raw-text or identity authority.",
    ),
    Capability(
        "multi_source_entity_resolution",
        "Multi-source entity resolution",
        "M7",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Resolver accepts source-aware evidence, but N-source orchestration is pending.",
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
        ComponentStatus.PARTIAL,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Settings compilation and candidate parity exist; native training/persistence is pending.",
    ),
    Capability(
        "approved_recipe_inference",
        "Approved train-approve-infer workflow",
        "I1",
        ComponentStatus.PARTIAL,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "The immutable recipe approval contract exists; new-data inference is pending.",
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
        ComponentStatus.PARTIAL,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Versioned aggregate family, instance, replicate, run, failure, and snapshot "
        "contracts exist; corpus generation and portfolio execution remain pending.",
    ),
    Capability(
        "linkage_strategy_advisor",
        "Evidence-Based Linkage Strategy Advisor",
        "I2B-I2D",
        ComponentStatus.PLANNED,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.NOT_VERIFIED,
        "Nearest-family retrieval, OOD detection, learned meta-ranking, and active "
        "benchmark planning remain evidence-gated future stages.",
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
        "integrated_synthetic_workflow": "two_source_link_only_one_to_one",
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
            "The only complete configuration-driven row-level orchestrator is the",
            "generated-synthetic two-source `link_only`, `one_to_one` workflow.",
            "M3 through M7 contain substantive",
            "components, but their general CLI and artifact-to-artifact orchestration remains an",
            "integration milestone.",
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
