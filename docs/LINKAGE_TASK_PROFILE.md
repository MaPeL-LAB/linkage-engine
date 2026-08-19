# Linkage Task Profile

## Purpose

A linkage task profile is a privacy-safe aggregate description used for advisory eligibility,
benchmark coverage, and later drift analysis. It is not a row-level dataset, a model input table,
or an approval artifact.

## Three profile stages

### Preflight profile

Available from validated configuration and approved aggregate environment facts before row-level
execution:

```text
linkage mode
assignment constraint
dataset role counts
canonical variable type counts
restricted-variable count
transformation, blocking, and comparison counts
candidate-pair budget band
label evidence class
network and remote-access policy
```

Exact row counts, missingness rates, entropy, and uniqueness are not fabricated when they have
not been measured. The initial CLI emits `record_count_band = not_observed`.

### Candidate graph profile

Available after candidate generation:

```text
candidate-pair count band
candidate-set-size bands
zero-candidate rate band
conflict-density band
candidate-budget status
candidate-recall status where eligible truth exists
```

It contains no candidate pairs.

### Evidence profile

Available after selected diagnostic models and assignment stages:

```text
pair-model count
score-margin band
model-disagreement band
calibration state
review-burden band
assignment-change band
```

It contains no score vectors or record references.

## Privacy rules

- global profiles are generated from synthetic scenarios only;
- local profiles remain restricted and Git-ignored;
- source column names and project identifiers are excluded;
- small quantities are suppressed or binned;
- continuous values are quantised when exact precision is unnecessary;
- digests are computed from approved aggregate fields, never raw values;
- a local profile is not transmitted to an external service;
- profile output has no decision, assignment, or merge authority.

## Observable versus latent features

Synthetic scenarios may retain a latent manifest describing true corruption parameters. The
advisor feature set is restricted to values observable at the corresponding profile stage. This
prevents synthetic evaluation from relying on information unavailable in a real project.

## CLI

```text
mapel-linkage profile-job --config CONFIG --project-root ROOT
```

The command performs configuration compilation and emits only the preflight safe summary.
