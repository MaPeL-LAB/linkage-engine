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

The complete configuration-driven CLI remains bounded to generated-synthetic, two-source
`link_only`, `one_to_one` execution. Within that boundary, the plural all-model portfolio,
strict artifact reload, recipe binding, and synthetic new-data replay are integrated. M3–M7
contain a mix of integrated workflow APIs and component-only capabilities; the matrix is
authoritative for each item.

The repository does not provide a general operational runner for arbitrary linkage modes,
N-source pairwise-recipe planning, reviewer-role governance, or locally approved retraining.
Automatic retraining and merge authority remain prohibited.

Code presence must not be reported as workflow completion. See
[`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md).

## Model and calibration limitations

- Native Splink 4 is the mandatory configured baseline. Fit, canonical persistence/reload,
  exact candidate parity, score evidence, calibration, locked-test evaluation, and typed
  prepared-data replay are integrated for generated synthetic data.
- Fellegi-Sunter term-frequency adjustment is not enabled in the current reference model.
- XGBoost, LightGBM, and feature-based PyTorch pair candidates execute in the configured
  protected tournament. LightGBM and PyTorch require the all-models runtime.
- Stacking uses OOF predictions grouped by connected source-side entity and household
  evidence. Native Splink is not assigned a false OOF claim and cannot be a stacking base.
- Only source-query ranker artifacts are executable by the source-to-target recipe contract.
  Target-query candidates may be trained and evaluated, but require a distinct
  direction-aware inference contract before execution.
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
- `run-model-portfolio` is synthetic-only and performs tournament, locked-test evaluation,
  strict persistence/reload, and recipe-bound replay in one bounded command. Separate
  operational train/approve/infer commands and shadow-challenger execution remain outside
  this CLI.
- Operational row-level execution remains blocked by design until the local restricted
  validation and approval workflow exists.

## Candidate and linkage-mode limitations

- Candidate generation supports exact, prefix equality, conjunction, disjunction, and the
  bounded package-owned predicate subset. More advanced blocking remains future work.
- Date-window predicates are available in the typed configuration and anchor-evidence
  layer, but candidate-retrieval coverage and parity must remain explicitly tested.
- Extended assignment and deduplication workflows are capability-tracked, but they are not
  dispatched by the bounded all-model CLI.
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
