# Current Limitations

## Evidence and validation boundary

- The complete integrated workflow is validated only as generated-synthetic software
  behaviour.
- No real-data population, subgroup, fairness, calibration, threshold, or operational
  validation has occurred.
- Synthetic metric thresholds in examples and regression guards are mechanical test
  settings, not operational recommendations.
- Operational champion selection, calibration, decision thresholds, and final test
  evaluation require locally approved verified truth.
- Repository visibility does not alter the synthetic-only boundary and is not a substitute
  for data governance.

## Component versus workflow boundary

Substantive M3 through M7 source components are implemented and unit-tested, but the only
complete configuration-driven row-level orchestrator is the two-source `link_only`,
`one_to_one` generated-synthetic workflow.

The following remain integration work rather than absent source code:

- adjudication import, consensus, label promotion, and authorised retraining orchestration;
- complete `dedupe_only` and `link_and_dedupe` runners;
- many-to-one, one-to-many, and unconstrained end-to-end workflows;
- portfolio configuration and promotion for LightGBM, stacking, and PyTorch challengers;
- N-source pairwise-evidence planning, graph construction, cluster decisions, and review;
- approved model-recipe loading and new-data inference.

Code presence must not be reported as workflow completion. See
[`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md).

## Model and calibration limitations

- The package-owned Fellegi-Sunter reference estimator is the integrated scoring path.
  A Splink 4 settings compiler and candidate-parity checks exist, but a full native Splink
  training, persistence, calibration, and reload lifecycle is not integrated.
- Fellegi-Sunter term-frequency adjustment is not enabled in the current reference model.
- XGBoost is the integrated supervised classifier and candidate ranker.
- LightGBM classifier and ranker implementations exist below the orchestration boundary.
- The stacking meta-learner exists, but protected out-of-fold portfolio construction and
  ensemble promotion are not integrated.
- The feature-based PyTorch matcher exists as an optional challenger, but the general
  portfolio and approved-inference workflow is not integrated.
- Sigmoid, isotonic, and Beta calibration are implemented. Each configured run uses one
  method; a protected calibration-method selection protocol remains integration work.
- The rankers use verified binary relevance. Graded adjudication relevance and NDCG-based
  operational selection remain later work.

## Test interpretation

The default core CI may omit optional runtime dependencies for speed. Collected tests that
skip because LightGBM or PyTorch is unavailable are not described as passed.

A dedicated `all-models` CI job installs pinned LightGBM and CPU PyTorch, runs the complete
suite, and fails when any test is skipped. Test reporting must retain separate collected,
passed, failed, and skipped counts.

A green synthetic test suite demonstrates bounded software behaviour. It does not establish
statistical performance on a real population.

## Current CLI limitations

- `status`, `doctor`, `init-local-project`, `validate-config`, and
  `emit-config-schema` provide configuration, environment, and capability functions.
- The row-level CLI requires `--synthetic-demo`.
- `generate-candidates`, `train`, `predict`, `assign`, and `evaluate` currently execute the
  required complete synthetic workflow and return the requested stage summary rather than
  loading reusable stage artifacts.
- A genuine train-versus-infer workflow, immutable approved recipe loading, and shadow
  challenger execution are not yet implemented.
- Operational row-level execution remains blocked by design until the local restricted
  validation and approval workflow exists.

## Candidate and linkage-mode limitations

- Candidate generation supports exact, prefix equality, conjunction, disjunction, and the
  bounded package-owned predicate subset. More advanced blocking remains future work.
- Date-window predicates are available in the typed configuration and anchor-evidence
  layer, but candidate-retrieval coverage and parity must remain explicitly tested.
- Single-source deduplication and extended assignment primitives exist, but complete
  mode-specific candidate, feature, decision, review, and evaluation orchestration is
  pending.
- Multi-source graph solvers consume source-aware evidence, but the platform does not yet
  build a complete N-source graph from approved pairwise pipeline recipes.
- The engine emits relationship or cluster evidence and statuses; it never constructs a
  consolidated master record.

## Engineering and deployment limitations

- Python 3.12 is the only supported runtime in the initial compatibility contract.
- Linux is the authoritative CI environment. macOS and Windows scripts exist, but dedicated
  platform smoke jobs remain release-hardening work.
- Determinism means repeatability inside a recorded software and hardware envelope, not
  universal cross-platform bitwise identity.
- Performance and memory behaviour at operational scale have not been benchmarked.
- Host-level filesystem, process, network, backup, encryption, and access controls remain
  the responsibility of the authorised local environment.
- No licence has been selected, package publication remains blocked, and no release has
  been approved.
- Privacy-preserving record linkage remains a separate threat-modelled research stream.

These limitations must remain visible in reports, model and recipe manifests, user
instructions, and manuscript claims.
