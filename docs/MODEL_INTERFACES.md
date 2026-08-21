# Model and Pipeline Interfaces

## Interface goals

Backends are replaceable adapters. No model may modify source data, bypass assignment/decision policy, export unapproved fields, or merge entities.

## Shared objects

```python
@dataclass(frozen=True, slots=True)
class TableRef:
    table_name: str
    schema_digest: str
    row_count: int
    contains_row_level_data: bool = True


@dataclass(frozen=True, slots=True)
class ModelArtifactRef:
    artifact_id: str
    model_family: str
    model_version: str
    configuration_digest: str
    feature_schema_digest: str
    manifest_path: Path
    payload_path: Path
```

Paths use `repr=False` in implementation to reduce accidental display.

## CandidateGenerator

Consumes dataset catalog, compiled candidate plan, and run context. Returns a candidate `TableRef` with pair keys, retrieving rule IDs, and candidate-set metadata.

## PairFeatureBuilder

Consumes bounded candidate and canonical dataset tables. M2C implements this contract with `DuckDBComparisonFeatureBuilder`. It returns pair-key/retrieval provenance plus package-generated values, configured level indices, exact indicators, and explicit missingness indicators. Source field values are not copied to the feature output.

## DeterministicAnchorEvaluator

Consumes prepared canonical datasets and the validated deterministic-anchor plan. It returns evidence rows with per-rule left/right uniqueness counts, `evidence_only` authority, and `eligible_as_training_truth = false`. Anchor evidence cannot emit relationship status or bypass later calibrated scoring, assignment, and decision policy.

## PairMatcher

```python
class PairMatcher(Protocol):
    def fit(self, training: TrainingBundle, context: RunContext) -> ModelArtifactRef: ...
    def score(
        self, model: ModelArtifactRef, features: TableRef, context: RunContext
    ) -> TableRef: ...
```

Scores are not automatically calibrated probabilities or decisions.

### M2D Fellegi–Sunter contract

`DuckDBFellegiSunterMatcher` consumes M2C comparison-level features rather than source values. It estimates smoothed nonmatch (`u`) probabilities from a bounded deterministic cross-source sample and match (`m`) probabilities by expectation–maximisation over aggregated comparison vectors. Its immutable public artifact retains only aggregate parameters, digests, counts, convergence metadata, and model provenance.

The corresponding pair table retains surrogate pair references and evidence fields locally, including log2 Bayes factor, match weight, model posterior, model/version identifiers, and the parameter digest. Every M2D result is labelled `model_posterior_uncalibrated` and `evidence_only`.

Reference model version `m2d-reference-v2` casts learned evidence constants explicitly to double
precision, uses a two-tail base-2 logistic expression that avoids positive exponential overflow,
and fails closed unless every aggregate weight and posterior is finite and bounded.

`SplinkSettingsPlanCompiler` translates the validated canonical comparison and blocking
configuration into a package-owned Splink settings plan. The integrated
`SplinkNativeDuckDBMatcher` requires the exact Splink 4.0.16 runtime, fixed safe input
aliases (`mapel_source_a` and `mapel_source_b`) and linker UID (`mapel001`), bounded exact
candidate-pair parity, and additive pseudo-count smoothing over aggregate m/u masses. It
persists only immutable, duplicate-free canonical value-hidden JSON with version,
configuration, schema, model, artifact, and recipe-binding integrity checks, then strictly
reloads before scoring. The package-owned matcher remains the deterministic oracle; native
scores are uncalibrated evidence only, and operational validity is not established.

## CandidateRanker

```python
class CandidateRanker(Protocol):
    def fit(self, training: RankingTrainingBundle, context: RunContext) -> ModelArtifactRef: ...
    def rank(
        self, model: ModelArtifactRef, candidates: TableRef, context: RunContext
    ) -> TableRef: ...
```

Allowed outputs are ranking score, rank, top-K membership, query group, and provenance. Relationship status is prohibited.

## ProbabilityCalibrator

Fits on an independent eligible partition and transforms model scores into probability estimates with an explicit artifact and diagnostics.

## AssignmentSolver

Consumes calibrated candidate edges and a compiled assignment plan. Produces selected real/no-match edges, capacity diagnostics, objective information, and solver provenance.

## DecisionPolicy

Combines anchor evidence, calibrated probability, rank, margin, assignment, retrieval completeness, data-quality state, and thresholds. Returns exactly one supported relationship status.

## ArtifactStore

Writes restricted payloads and unrestricted manifests separately. It verifies paths, digests, schemas, model role, and immutability.

## Safe representation

Domain objects that may reference row-level material must have value-safe `repr`/`str` implementations. No public object exposes a convenient row preview.

## VerifiedLabelBatch

M2E introduces a partition-specific verified-label contract. It retains private
surrogate pair references, binary labels, entity-component digests, household-
component digests, source kind, verification protocol, source digest, and a
canonical label-authority digest.

Only these source kinds are eligible:

```text
synthetic_truth
verified_human_adjudication
verified_gold_standard
```

The contract rejects duplicate/conflicting labels and pair, entity, or
household overlap across protected partitions. An unverified reference cannot
be converted to this interface.

## BoostedFeatureMatrix and BoostedLabelledMatrix

The M2E matrix interface contains package-generated numeric comparison features
and private pair references. Feature values, labels, pair references, and
feature names are excluded from public object representations. Safe summaries
contain aggregate counts and schema/authority digests only.

A labelled matrix records:

```text
partition
label_source_kind
label_authority_digest
selection_digest
positive_count
negative_count
hard_negative_count
```

Training selection is allowed only for the training partition and may use only
verified nonmatches.

## XGBoostPairClassifier

`XGBoostPairClassifier.fit` consumes a verified training matrix and a bounded
`BoostedTreeModelConfig`. It returns a native-JSON `XGBoostModelArtifact` whose
manifest records aggregate provenance and explicit authority limits.

`XGBoostPairClassifier.score` returns a local `TableRef` containing private pair
references and uncalibrated evidence scores. It requires an exact feature-schema
and feature-order match.

`XGBoostPairClassifier.evaluate` accepts only a nontraining labelled matrix and
returns aggregate diagnostics. Its fixed threshold is diagnostic only.

The interface never returns a relationship status and cannot invoke assignment,
calibration, or record merging.


## Complete M2 calibration, ranking, assignment, and decision contracts

### ChampionSelection

Champion selection consumes aggregate validation evidence for at least two model candidates. All candidates must share one validation label-authority digest, partition-manifest digest, and pair count. Selection uses the configured primary metric, a deterministic secondary metric, and stable model identity. The artifact fixes:

```text
test_partition_used = false
calibration_partition_used = false
decision_authority = evidence_only
```

### PairScoreBatch and CalibratorArtifact

A protected score batch aligns private surrogate pairs, pair digests, base-model scores, verified binary labels, source-model identity, feature schema, label authority, and partition manifest. A calibrator may fit only on the calibration partition and only after champion selection.

Sigmoid and isotonic calibrators persist package-defined JSON payloads plus manifests containing source-model, selection, validation-label, calibration-label, partition, dependency-version, and integrity digests. Calibrated scores remain evidence only and have no threshold or assignment authority.

### RankingMatrix and RankingScoreBatch

The ranking adapter groups candidates by one query-side surrogate. Training queries must contain at least two candidates and, by default, verified positive and negative relevance. Outputs contain score, rank, top-K membership, model identity, and artifact digests. Their fixed authorities are:

```text
decision_authority = ranking_only
relationship_authority = none
merge_authority = none
```

### AssignmentEdgeBatch and AssignmentResult

The one-to-one assignment solver receives calibrated candidate edges plus one private no-match option per required source record. It produces selected real/no-match edges, objective and capacity diagnostics, and deterministic solver provenance. Assignment has global selection authority but no relationship-classification or merge authority.

### RelationshipDecision

The explicit policy combines assignment, calibrated probability, margin, rank, retrieval completeness, truncation, calibration state, data quality, model disagreement, and anchor conflict. It emits exactly one status:

```text
confirmed
review_required
unresolved
no_match
```

Incomplete retrieval, truncation, invalid calibration, or insufficient data quality cannot produce `no_match`; they produce `unresolved`. No public interface contains `merge()` or master-record construction.

### SyntheticVerticalSliceRunner

The orchestrator calls existing package-owned stages in order and writes native model/calibrator/ranker artifacts, restricted relationship and review files, aggregate evaluation, and a row-free run manifest. Stage commands and `run` call the same orchestrator rather than maintaining separate implementations.
