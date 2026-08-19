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
and a stacking meta-learner. Plural configuration, bounded portfolio declarations, immutable
stage artifacts, and protected out-of-fold manifests are integrated.

Remaining integration work:

- general portfolio training and artifact-to-artifact stage execution;
- model cards, promotion policy, and immutable selection rationale;
- independent calibration-method selection without locked-test access;
- workflow integration for LightGBM, stacking, and PyTorch challengers.

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

## Current cross-cutting target: I1 general orchestration

I1 establishes:

- immutable artifact-to-artifact stage execution;
- protected portfolio training and out-of-fold stacking evidence;
- separate train, select, calibrate, approve, and infer commands;
- approved-recipe new-data inference;
- shadow challengers with no decision authority;
- general orchestration for M3 through M7.

See
[`architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md`](architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md).

## I2A — Stage-1 Linkage Strategy Advisor

I2A is implemented as an advisory structural workflow:

```text
configuration-only preflight task profile
→ lifecycle-aware hard eligibility rules
→ mandatory Fellegi-Sunter baseline
→ structural Pareto frontier
→ family-diverse bounded shortlist
→ transparent explanations
→ explicit abstention from empirical ranking
```

The advisor fixes recommendation, decision, assignment, merge, and automatic-promotion
authority in immutable contracts. It makes no sensitivity, PPV, calibration, or operational
performance claim without benchmark evidence.

## B1 — synthetic benchmark evidence library

B1 is the next advisor evidence-generation milestone:

- freeze scenario, profile, recipe-fingerprint, metric, failure, and registry schemas;
- implement the designed experimental matrix and scenario-family taxonomy;
- execute the bounded model portfolio across instances and replicates;
- retain successes, failures, timeouts, ineligible recipes, and abstentions;
- populate an aggregate synthetic registry;
- pre-specify held-out scenario families and unseen corruption mechanisms;
- produce a registry coverage and readiness report.

The current repository contains the registry contracts but no populated benchmark corpus.

## I2B — similarity and coverage advisor

Deferred until B1 coverage is sufficient. Scope:

```text
observable feature standardisation
nearest-family retrieval
coverage scoring
out-of-distribution detection
uncertainty and abstention
held-out-family regret and oracle-coverage evaluation
```

Family counts such as 50–100 are planning ranges, not automatic validity gates.

## I2C — learned meta-ranking advisor

Deferred until I2B is independently validated and the recipe-by-family evidence matrix has
adequate overlap. Group-held-out learning-to-rank must outperform transparent retrieval without
violating hard constraints.

## I2D — active benchmark planning

Deferred until learned-advisor uncertainty is calibrated. The planner must target genuine
coverage gaps and demonstrate prospective reduction in uncertainty or recommendation regret.

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
