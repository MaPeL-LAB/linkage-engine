# Plural Model Configuration and Stage Artifacts

## Status

Bounded post-audit integration milestone. This work expands synthetic model-development configuration and provenance contracts; it does not provide an approved operational inference runner.

## Plural model configuration

The existing singular fields remain valid for backward compatibility. `ModelsConfig` additionally supports:

- multiple boosted-tree pair candidates;
- multiple candidate rankers;
- multiple PyTorch challengers;
- stacking ensembles trained only from out-of-fold training predictions;
- a versioned portfolio selection that names the mandatory Fellegi-Sunter baseline, selected pair candidates, selected rankers, challenger budget, and shadow-scoring policy.

Every model ID is unique across pair models, rankers, neural candidates, and ensembles. An enabled stacking candidate may reference only enabled, non-ensemble base models declared in the same project. Portfolio selection cannot use the locked test partition and has no decision or merge authority.

## Stage artifact references

`StageArtifactRef` records aggregate lineage metadata for each immutable stage output:

```text
artifact ID
stage and artifact kind
run and engine version
artifact, configuration, and schema digests
upstream artifact digests
row-level and restricted flags
decision and merge authority
```

Row-level artifacts must be restricted. Public summaries contain counts and digests only; they do not contain paths, rows, identifiers, candidate pairs, or review values.

`StageArtifactLedger` requires every upstream digest to identify an external input or an earlier artifact. This yields an ordered acyclic provenance chain without loading row-level content into the application layer.

## Out-of-fold prediction provenance

`OutOfFoldPredictionManifest` is the required input contract for stacking. It records model, feature, label-authority, split, fold-count, pair-count, and prediction digests while fixing:

```text
partition = training_oof
test_partition_used = false
calibration_partition_used = false
decision_partition_used = false
decision_authority = evidence_only
merge_authority = none
```

The manifest contains no pair references or prediction values.

## Remaining work

1. materialize immutable stage artifacts during portfolio training;
2. generate grouped out-of-fold predictions without entity or household leakage;
3. train eligible pair candidates and stacking only from protected training evidence;
4. select the champion on validation evidence;
5. select calibration without reusing calibrator-fit observations;
6. bind the champion, calibrator, ranker, assignment, and decision policy into an approved recipe;
7. implement a separate recipe-driven inference runner with optional shadow challengers.

Synthetic tests establish software behaviour only. They do not establish operational model validity.
