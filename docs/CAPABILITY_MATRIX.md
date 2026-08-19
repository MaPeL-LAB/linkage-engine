# Capability Matrix

This matrix distinguishes four questions that must not be conflated:

1. Is a bounded component present in source code?
2. Is it reachable from an approved configuration-driven workflow?
3. Does CI execute its real runtime dependency path?
4. Has it been validated for operational use on an approved population?

No current capability has established operational validation.

| Capability | Milestone | Component | Workflow | Runtime verification | Notes |
|---|---|---|---|---|---|
| `fellegi_sunter_reference` | M2 | implemented | workflow_integrated | core_ci | Package-owned scoring path; the full native Splink lifecycle remains partial. |
| `xgboost_pair_classifier` | M2 | implemented | workflow_integrated | core_ci | Eligible verified labels only; uncalibrated scores remain evidence-only. |
| `xgboost_candidate_ranker` | M2 | implemented | workflow_integrated | core_ci | Ranking-only authority; it cannot emit a relationship status. |
| `sigmoid_calibration` | M2 | implemented | workflow_integrated | core_ci | Fits only on the protected calibration partition. |
| `isotonic_calibration` | M2 | implemented | workflow_integrated | core_ci | Fits only on the protected calibration partition. |
| `beta_calibration` | M5 | implemented | workflow_integrated | core_ci | Configurable alternative using the protected calibration partition. |
| `one_to_one_assignment` | M2 | implemented | workflow_integrated | core_ci | OR-Tools is the primary solver; SciPy provides a small-problem reference. |
| `adjudication_lifecycle` | M3 | implemented | component_only | core_ci | Implemented below the CLI boundary; no automatic retraining is permitted. |
| `active_learning_queue` | M3 | implemented | component_only | core_ci | Ordering authority only; unknown pairs are not silently relabelled. |
| `many_to_one_assignment` | M4 | implemented | component_only | core_ci | Solver and invariant tests exist; general orchestration is pending. |
| `one_to_many_assignment` | M4 | implemented | component_only | core_ci | Solver and invariant tests exist; general orchestration is pending. |
| `unconstrained_assignment` | M4 | implemented | component_only | core_ci | Pairwise selection exists without bypassing the decision-policy boundary. |
| `single_source_deduplication` | M4 | implemented | component_only | core_ci | Pair canonicalisation and clustering safeguards exist; CLI integration is pending. |
| `link_and_dedupe` | M4 | partial | not_integrated | core_ci | Required primitives exist, but no complete combined execution path exists. |
| `lightgbm_pair_classifier` | M5 | implemented | component_only | all_models_ci | Optional dependency; dedicated all-models CI must execute the runtime path. |
| `lightgbm_candidate_ranker` | M5 | implemented | component_only | all_models_ci | Optional dependency; dedicated all-models CI must execute the runtime path. |
| `stacking_ensemble` | M5 | implemented | component_only | core_ci | A protected meta-model workflow and portfolio configuration remain pending. |
| `pytorch_tabular_matcher` | M6 | implemented | component_only | all_models_ci | Optional feature-based challenger; it has no raw-text or identity authority. |
| `multi_source_entity_resolution` | M7 | implemented | component_only | core_ci | Resolver accepts source-aware evidence, but N-source orchestration is pending. |
| `correlation_clustering` | M7 | implemented | component_only | core_ci | Strict cannot-link enforcement and violation reporting are implemented. |
| `constrained_agglomerative_clustering` | M7 | implemented | component_only | core_ci | Cluster merges preserve cannot-link and configured capacity boundaries. |
| `bcubed_cluster_metrics` | M7 | implemented | component_only | core_ci | BCubed precision, recall, F1, purity, and constraint diagnostics are available. |
| `splink_native_model_lifecycle` | I1 | partial | not_integrated | core_ci | Settings compilation and candidate parity exist; native training/persistence is pending. |
| `approved_recipe_inference` | I1 | partial | not_integrated | core_ci | The immutable recipe approval contract exists; new-data inference is pending. |
| `linkage_strategy_advisor` | I2 | planned | not_integrated | not_verified | Advisory-only pipeline recommendation with coverage checks and abstention. |

## Current integrated workflow

The only complete configuration-driven row-level orchestrator is the
generated-synthetic two-source `link_only`, `one_to_one` workflow.
M3 through M7 contain substantive
components, but their general CLI and artifact-to-artifact orchestration remains an
integration milestone.

## Test reporting

CI must report collected, passed, failed, and skipped counts separately. A collected
test is not described as passed when its optional runtime dependency was unavailable.
The dedicated all-models CI job installs LightGBM and PyTorch and fails when any test
is skipped.

## Authority boundary

- candidate retrieval does not decide identity;
- pair and ranking models remain evidence-only;
- assignment selects compatible edges but does not classify relationships;
- only the explicit decision policy emits relationship status;
- no capability has silent merge or master-record authority;
- synthetic testing does not establish operational linkage validity.
