# Roadmap

## Status vocabulary

Linkage Engine reports four independent states:

```text
specified
component implemented
workflow integrated
operationally validated
```

Source-code presence and unit tests do not by themselves make a component reachable from the
configuration-driven CLI. The normative status table is
[`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md).

## Completed integrated foundations

- **M0:** research, architecture, ADRs, governance, bibliography, package shell, and
  publication guard;
- **M1:** strict configuration compilation, path and logging controls, manifests, generated
  schema, and deterministic synthetic generation;
- **M2:** complete generated-synthetic two-source `link_only`, `one_to_one` workflow,
  including canonical preparation, anchors, bounded candidates, comparison features,
  Fellegi-Sunter and XGBoost pair models, validation-only champion selection, sigmoid,
  isotonic and Beta calibration, XGBoost ranking, explicit no-match assignment, four-status
  decisions, restricted review export, evaluation, and deterministic orchestration.

M2 is merged and CI-verified as software behaviour on generated synthetic data. It is not
operationally validated.

## Implemented components awaiting general orchestration

### M3 — adjudication and label lifecycle

Implemented components include bounded review import, disagreement handling, active-learning
ordering, eligibility evaluation, and controlled construction of verified label batches.

Remaining integration work:

- append-only project-level audit persistence;
- explicit CLI import, consensus, promotion, and retraining commands;
- partition-manifest compatibility at the orchestration boundary;
- reviewer-role and protocol configuration;
- no automatic retraining after adjudication.

### M4 — extended linkage modes

Implemented components include many-to-one, one-to-many, unconstrained assignment, and
single-source deduplication primitives.

Remaining integration work:

- full `dedupe_only` runner;
- full `link_and_dedupe` runner;
- mode-specific candidate, comparison, decision, review, and evaluation artifacts;
- configuration-driven dispatch and synthetic end-to-end acceptance tests.

### M5 — broader model portfolio

Implemented components include LightGBM pair classification and ranking, Beta calibration,
and a stacking meta-learner.

Remaining integration work:

- portfolio configuration rather than singular boosted/ranking fields;
- protected out-of-fold base-model predictions for stacking;
- model cards, promotion policy, and immutable selection rationale;
- independent calibration-method selection without locked-test access;
- workflow integration for LightGBM and stacking.

### M6 — optional neural matcher

The feature-based PyTorch matcher and controlled artifacts are implemented.

Remaining integration work:

- complete bounded training configuration;
- portfolio/challenger orchestration;
- calibration and promotion policy;
- device and reproducibility reporting in approved recipe artifacts.

### M7 — multi-source entity resolution

Implemented components include source-aware evidence graphs, correlation clustering,
constrained agglomerative clustering, connected-component baselines, cannot-link enforcement,
crosswalk export, BCubed metrics, purity, pairwise metrics, and violation diagnostics.

Remaining integration work:

- N-source configuration and source-pair recipe planning;
- pairwise calibrated-evidence compatibility checks;
- graph construction from approved pairwise artifacts;
- cluster-level decisions and restricted conflict review;
- end-to-end multi-source synthetic acceptance tests.

## Current cross-cutting target: I1 audit and integration

I1 establishes:

- a package-owned capability registry and generated capability matrix;
- separate core and all-model runtime CI;
- honest collected/passed/skipped test reporting;
- train-versus-infer separation;
- an immutable `PipelineRecipeArtifact`;
- genuine stage boundaries rather than every CLI stage rerunning the complete pipeline;
- general orchestration for M3 through M7;
- shadow-challenger execution with no decision authority.

See
[`architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md`](architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md).

## Next cross-cutting target: I2 Linkage Strategy Advisor

The advisor will use privacy-safe task profiles and aggregate benchmark evidence to recommend
a small, explainable pipeline shortlist. It must support coverage checks, uncertainty,
out-of-distribution detection, and abstention.

Synthetic recommendations are priors only. A local bounded champion-challenger evaluation,
independent calibration, threshold selection, locked testing, and approval remain mandatory.

## M8 — release hardening

- Python and operating-system compatibility matrix;
- runtime and memory benchmarks at increasing synthetic scales;
- security and dependency review;
- artifact migration and API-stability policies;
- complete model cards and error-code catalogue;
- private release and rollback procedures;
- separately approved licence, repository visibility, and package publication decisions.

Privacy-preserving record linkage remains a separate research stream requiring its own threat
model. No milestone implies operational validity or publication without explicit approval.
