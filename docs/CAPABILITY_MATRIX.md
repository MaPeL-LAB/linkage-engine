# Capability Matrix

This matrix distinguishes four questions that must not be conflated:

1. Is a bounded component present in source code?
2. Is it reachable from an approved configuration-driven workflow?
3. Does CI execute its real runtime dependency path?
4. Has it been validated for operational use on an approved population?

No current capability has established operational validation.

| Capability | Milestone | Component | Workflow | Runtime verification | Notes |
|---|---|---|---|---|---|
| `fellegi_sunter_reference` | M2 | implemented | workflow_integrated | core_ci | Package-owned deterministic reference oracle; evidence-only with no decision authority. |
| `xgboost_pair_classifier` | M2 | implemented | workflow_integrated | core_ci | Eligible verified labels only; uncalibrated scores remain evidence-only. |
| `xgboost_candidate_ranker` | M2 | implemented | workflow_integrated | core_ci | Ranking-only authority; it cannot emit a relationship status. |
| `sigmoid_calibration` | M2 | implemented | workflow_integrated | core_ci | Fits only on the protected calibration partition. |
| `isotonic_calibration` | M2 | implemented | workflow_integrated | core_ci | Fits only on the protected calibration partition. |
| `beta_calibration` | M5 | implemented | workflow_integrated | core_ci | Configurable alternative using the protected calibration partition. |
| `one_to_one_assignment` | M2 | implemented | workflow_integrated | core_ci | OR-Tools is the primary solver; SciPy provides a small-problem reference. |
| `adjudication_audit_ledger` | M3 | implemented | workflow_integrated | core_ci | Immutable append-only audit ledger, multi-reviewer consensus, and label promotion workflow integrated. |
| `active_learning_queue` | M3 | implemented | workflow_integrated | core_ci | Active-learning review ordering across uncertainty, margin, committee, and hybrid modes. |
| `many_to_one_assignment` | M4 | implemented | workflow_integrated | core_ci | Greedy many-to-one assignment is CLI-integrated only in the exact generated-synthetic I1C link_only combination; operational dispatch is not established. |
| `one_to_many_assignment` | M4 | implemented | workflow_integrated | core_ci | Greedy one-to-many assignment is CLI-integrated only in the exact generated-synthetic I1C link_only combination; operational dispatch is not established. |
| `unconstrained_assignment` | M4 | implemented | workflow_integrated | core_ci | Threshold-based unconstrained assignment is CLI-integrated only for the exact generated-synthetic I1C link_only and dedupe_only combinations. |
| `single_source_deduplication` | M4 | implemented | workflow_integrated | core_ci | Canonical same-table pairs and aggregate clustering are CLI-integrated only for the exact generated-synthetic I1C dedupe_only and link_and_dedupe combinations. |
| `link_and_dedupe` | M4 | implemented | workflow_integrated | core_ci | Two-source linkage plus two intra-source clustering surfaces is CLI-integrated only for generated-synthetic I1C link_and_dedupe with one_to_one assignment. |
| `configuration_driven_linkage_modes` | I1C | implemented | workflow_integrated | core_ci | Generated-synthetic CLI dispatch is allow-listed to link_only with many_to_one, one_to_many, or unconstrained assignment; dedupe_only with unconstrained assignment; and link_and_dedupe with one_to_one assignment. Operational validation is not established, no arbitrary or real-data mode dispatch is authorized, and strict least-privilege attestation data-access isolation is not established. |
| `lightgbm_pair_classifier` | M5 | implemented | workflow_integrated | all_models_ci | Configured synthetic tournament, protected selection/calibration, persisted reload, and recipe-bound replay execute in all-models CI. |
| `lightgbm_candidate_ranker` | M5 | implemented | workflow_integrated | all_models_ci | Configured source-query execution is recipe-replayable; target-query candidates are trained and reported but cannot be silently reinterpreted for source assignment. |
| `stacking_ensemble` | M5 | implemented | workflow_integrated | core_ci | Protected meta-model workflow, tournament selection, and out-of-fold stacking integrated. |
| `pytorch_tabular_matcher` | M6 | implemented | workflow_integrated | all_models_ci | Configured deterministic CPU training, protected tournament selection, persisted reload, and recipe-bound replay are integrated; it has no raw-text or identity authority. |
| `multi_source_entity_resolution` | M7 | implemented | workflow_integrated | core_ci | Multi-source N-dataset entity resolution and global crosswalk workflow integrated. |
| `correlation_clustering` | M7 | implemented | component_only | core_ci | Strict cannot-link enforcement and violation reporting are implemented. |
| `constrained_agglomerative_clustering` | M7 | implemented | component_only | core_ci | Cluster merges preserve cannot-link and configured capacity boundaries. |
| `bcubed_cluster_metrics` | M7 | implemented | component_only | core_ci | BCubed precision, recall, F1, purity, and constraint diagnostics are available. |
| `splink_native_model_lifecycle` | I1 | implemented | workflow_integrated | core_ci | Pinned Splink fit, canonical JSON reload, bounded candidate parity, and scoring are integrated as uncalibrated evidence only; operational validity is not established. |
| `configuration_driven_model_portfolio` | I1B | implemented | workflow_integrated | all_models_ci | Generated-synthetic native Splink baseline plus configured XGBoost, LightGBM, PyTorch, stacking, and ranking candidates with group-protected OOF evidence, validation-only selection, calibration-only fitting, locked-test evaluation, strict artifact reload, and disjoint recipe-bound replay; operational validity is not established. |
| `approved_recipe_inference` | I1 | implemented | workflow_integrated | core_ci | The immutable recipe approval contract and approved recipe inference workflow integrated. |
| `stage1_linkage_strategy_advisor` | I2A | implemented | workflow_integrated | core_ci | Configuration-only profiling, hard eligibility, structural Pareto shortlisting, transparent explanations, and explicit empirical abstention. |
| `synthetic_benchmark_registry` | B1 | implemented | workflow_integrated | core_ci | Stable seed-v1 plus a versioned 64-family/280-instance advisor-v2 design, three truth-safe real benchmark adapters, prospective family partitions, deterministic shards, and append-only resume controls; corrected execution v2 completed its exact 9,800-run evidence grid, while strategy qualification remains separate. |
| `stage2_similarity_advisor` | I2B | implemented | workflow_integrated | core_ci | Nearest scenario family retrieval, weighted distance computation, out-of-distribution thresholding, empirical performance distribution aggregation, and strict advisory invariants; the first prospective qualification failed the fixed-baseline-improvement and OOD-detection gates. |
| `stage3_meta_ranking_advisor` | I2C | implemented | workflow_integrated | core_ci | Learned meta-regressor with family-disjoint fit, conformal interval calibration, locked evaluation, true-mechanism OOD exclusion, scenario-replicate-complete adapter gating, learned shortlist ordering, and similarity fallback; the first prospective qualification failed improvement and locked interval-coverage gates. |
| `linkage_strategy_advisor` | I2D | implemented | workflow_integrated | core_ci | Snapshot-bound active planning plus separately approved, digest-bound advisor-corpus shard execution, append-only evidence checks, and aggregate qualification artifacts; the first separately approved v3.1 qualification passed every Stage-2 gate but returned not_qualified because Stage 3 failed the fixed regret-improvement gate. Similarity fallback remains mandatory and operational validity remains unestablished. |

## Current integrated workflow

The legacy complete configuration-driven workflow remains bounded to
generated-synthetic two-source `link_only`, `one_to_one` execution. I1B adds the
configured all-model portfolio path within that same boundary.
I1C separately allow-lists exactly `link_only` with `many_to_one`, `one_to_many`,
or `unconstrained`; `dedupe_only` with `unconstrained`; and `link_and_dedupe` with
`one_to_one`. It is generated-synthetic only, operational validity is not
established, strict least-privilege attestation data-access isolation is not
established, and no arbitrary M3-M7, multi-source, or real-data dispatch is implied.

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
