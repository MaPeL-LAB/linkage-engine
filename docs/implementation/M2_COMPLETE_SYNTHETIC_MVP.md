# M2 Complete Synthetic Vertical Slice

**Status:** Implemented for generated synthetic data only
**Package version:** `0.2.0.dev0`
**Operational validation:** Not established

## Purpose

M2 closes the first complete two-source `link_only`, `one_to_one` software slice.
It connects the already merged configuration, DuckDB, preprocessing, candidate,
comparison, deterministic-anchor, Fellegi–Sunter, and verified-label XGBoost layers
to the remaining calibration, ranking, assignment, relationship-decision, review,
evaluation, orchestration, and command-line stages.

Passing this milestone establishes a reproducible synthetic software workflow. It does
not establish linkage accuracy, calibration, fairness, sensitivity, positive predictive
value, false-link rates, missed-link rates, or operational fitness on a real population.

## Complete synthetic execution path

```text
validated configuration
→ immutable execution plan
→ deterministic synthetic source generation
→ canonical local preprocessing
→ deterministic-anchor evidence
→ bounded candidate generation
→ comparison-feature construction
→ protected entity/household label partitions
→ Fellegi–Sunter evidence scoring
→ verified-label XGBoost pair scoring
→ validation-only champion selection
→ independent sigmoid or isotonic calibration
→ XGBoost candidate ranking
→ one-to-one assignment with explicit no-match
→ confirmed/review_required/unresolved/no_match policy
→ restricted local review queue
→ aggregate synthetic evaluation
→ privacy-safe run and artifact manifests
```

## Champion selection and calibration

Fellegi–Sunter and XGBoost are compared on the protected validation partition only.
The selected model family is recorded in an immutable selection artifact. The test and
calibration partitions cannot select the champion.

The selected model is calibrated on the separate calibration partition using either:

- monotone sigmoid calibration; or
- monotone isotonic calibration.

Calibrator payloads and manifests use package-defined JSON rather than Python pickle.
Payload and manifest digests are checked before application. Calibrated outputs remain
`evidence_only`; they do not acquire threshold, assignment, or merge authority.

Configured relationship thresholds are evaluated on the protected decision partition.
That report is marked `synthetic_benchmark_only`; the locked test partition is not used
to choose or approve thresholds.

## Ranking

The XGBoost ranker groups candidate target records by source-record surrogate and emits:

```text
ranking score
candidate rank
top-K membership
ranker identity and version
model and artifact digests
```

The ranker is fixed to:

```text
decision_authority = ranking_only
relationship_authority = none
```

It cannot emit a relationship status, select a no-match outcome, perform assignment, or
merge records.

## One-to-one assignment

The production synthetic path uses OR-Tools minimum-cost flow. A SciPy dense linear
assignment implementation is retained as a small-problem reference solver.

Every source record receives:

- zero or more real candidate edges; and
- one private dummy no-match edge.

Calibrated probabilities are converted to bounded log-odds utilities and deterministic
integer costs. Stable pair digests break ties without using source values. Successful
one-to-one output must have zero source or target capacity violations.

## Relationship decisions

The decision layer emits exactly one of:

- `confirmed`;
- `review_required`;
- `unresolved`; or
- `no_match`.

`no_match` is permitted only after a complete candidate search and an explicit no-match
assignment. Candidate truncation, invalid calibration, insufficient data quality, or an
incomplete search force `unresolved` rather than being silently interpreted as absence
of identity.

No decision stage creates a consolidated master record. `merge_authority` remains
`none` throughout the pipeline.

## Restricted review queue

`review_required` and `unresolved` relationships are exported to a Git-ignored local
queue. The queue:

- contains package-generated reason codes;
- obeys the configured restricted-output allow-list;
- stores only surrogate references and approved fields;
- writes an aggregate unrestricted manifest without row values;
- does not automatically promote review events to model truth.

The append-only decision import and full adjudication lifecycle remain M3 work.

## Evaluation

The aggregate report contains:

- candidate recall and Cartesian reduction;
- candidate-set-size distribution and blocking-rule contribution;
- pair sensitivity, PPV, false-link and missed-link rates;
- average precision and precision–recall points;
- Brier score, reliability bins, calibration intercept and slope;
- ranking recall@K, top-1 rate, mean reciprocal rank, and true-match rank;
- assignment accuracy, no-match accuracy, changes from independent top-1, and violations;
- relationship-status counts;
- pair performance by missingness pattern and candidate-set-size band;
- protected decision-threshold evidence;
- a versioned mechanical synthetic regression guard.

The aggregate report is row-free and always includes the synthetic-validation warning.

## Functional command line

The following commands are implemented for the generated synthetic workflow:

```text
mapel-linkage doctor --project-root ROOT
mapel-linkage init-local-project --directory ROOT
mapel-linkage generate-candidates --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage train --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage predict --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage assign --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage evaluate --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage run --config CONFIG --project-root ROOT --synthetic-demo
```

The stage commands execute required upstream stages and return only the requested
aggregate stage summary. Omitting `--synthetic-demo` fails safely in the repository
build; real operational data remain a local-only concern.

## Reproducibility and artifacts

The run manifest records package/dependency versions, configuration and registry
digests, random seed, stage counts/digests, selected model, calibrator, ranker,
assignment, and relationship counts. It excludes source paths, columns, identifiers,
record values, candidate pairs, and review values.

Native artifacts are:

```text
XGBoost pair model JSON + manifest
calibrator JSON + manifest
XGBoost ranking model JSON + manifest
restricted relationship JSONL
restricted review JSONL
aggregate evaluation JSON
run manifest JSON
```

Operational artifacts remain under Git-ignored local roots.

## Acceptance gate

M2 is accepted only when the exact review head passes on Python 3.12:

```text
repository privacy/integrity verification
Ruff formatting and linting
strict mypy
pre-commit
all synthetic-only tests
complete deterministic end-to-end run
wheel and source-distribution builds
restricted-distribution inspection
normal pull-request CI
post-merge main CI
```

> Synthetic testing establishes software behaviour only; it does not validate linkage
> accuracy on real populations or systems.
