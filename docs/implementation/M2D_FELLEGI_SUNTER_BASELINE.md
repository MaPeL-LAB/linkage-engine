# M2D Fellegi–Sunter Evidence Baseline

Status: implementation candidate

## Purpose

M2D introduces the first statistical pair-evidence model while preserving the
separation between model evidence and identity decisions.

The implementation has two deliberately separate parts:

1. a deterministic DuckDB-backed reference estimator used as a transparent
   oracle for probability parameters and pair evidence; and
2. a package-owned compiler that translates validated configuration into a
   Splink 4 settings plan without exposing raw SQL or executable configuration.

The reference implementation follows Fellegi–Sunter mixture semantics. It is
not presented as a substitute for the designated Splink production adapter;
it provides a small auditable baseline and parity target.

## Training sequence

The bounded synthetic workflow is:

```text
prepared canonical datasets
→ deterministic random-pair sample
→ M2C comparison features for the random sample
→ smoothed u probabilities
→ aggregate comparison-vector patterns from blocked candidates
→ expectation–maximisation of m probabilities
→ per-level log2 Bayes factors
→ immutable aggregate model artifact
→ local pair-evidence table
```

The random sample uses a deterministic hash order over surrogate record keys
and a configured pair limit. The sampled pairs are used only to estimate the
nonmatch (`u`) comparison-level probabilities.

Expectation–maximisation works on aggregated comparison-vector counts rather
than reading record values into model objects. The configured prior probability
that two random records match is held fixed in this initial reference model.
Additive smoothing prevents zero probabilities and infinite weights.

Reference model version `m2d-reference-v2` also prevents scoring-time numeric transport failures:
learned evidence constants are explicitly cast to DuckDB `DOUBLE`, posterior materialisation uses
a stable two-tail base-2 logistic expression, and aggregate outputs are rejected unless match
weights and probabilities are finite with probabilities inside `[0, 1]`.

## Missingness

An explicit missing comparison level is neutral in the initial baseline:

```text
m_missing = u_missing
log2(m_missing / u_missing) = 0
```

Missingness indicators remain available to later supervised models, where
informative missingness can be evaluated explicitly. Missing values are not
silently interpreted as disagreement.

## Evidence output

The local score table records:

```text
__ml_fs_log2_bayes_factor
__ml_fs_match_weight
__ml_fs_model_probability
__ml_fs_model_id
__ml_fs_model_version
__ml_fs_parameter_digest
__ml_fs_probability_status
__ml_fs_decision_authority
```

Every score is marked:

```text
probability_status = model_posterior_uncalibrated
decision_authority = evidence_only
```

The model posterior is conditional on the configured prior, comparison-level
probabilities, conditional-independence approximation, candidate-generation
process, and synthetic training regime. It is not an independently calibrated
operational probability.

## Splink boundary

`SplinkSettingsPlanCompiler` emits package-owned comparison levels and blocking
rules over generated canonical columns. Project configuration cannot supply:

- Splink objects;
- raw SQL;
- Python callables;
- import paths;
- executable expressions.

The optional runtime dependency is tested by constructing a Splink 4
`SettingsCreator` from the compiled plan. Full runtime parity and model-export
work remain subsequent M2D tasks.

## Privacy and governance boundary

M2D does not:

- use real records in tests or CI;
- retain source values in model artifacts;
- print pair references or probabilities in logs;
- treat anchors as training truth;
- treat an unverified crosswalk as truth;
- calibrate model probabilities;
- select operational thresholds;
- solve assignment;
- emit relationship statuses;
- merge records or create master entities.

## Acceptance tests

M2D tests cover:

- deterministic bounded random-pair sampling;
- normalised smoothed `m` and `u` distributions;
- neutral missing levels;
- stronger evidence for higher-specificity levels in the synthetic fixture;
- deterministic parameter digests;
- score-table candidate coverage;
- extreme finite evidence-weight stability and finite bounded posterior checks;
- evidence-only and uncalibrated status fields;
- pair-budget enforcement before fitting;
- safe errors and object representations;
- package-owned Splink settings construction.

> Synthetic testing establishes software behaviour only. It does not validate
> linkage accuracy, calibration, fairness, sensitivity, positive predictive
> value, false-link rates, missed-link rates, or operational fitness on real
> populations or systems.
