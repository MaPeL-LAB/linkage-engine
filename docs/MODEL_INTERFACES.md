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

Consumes candidate and canonical dataset tables. Returns comparison features without source values in unrestricted metadata.

## PairMatcher

```python
class PairMatcher(Protocol):
    def fit(self, training: TrainingBundle, context: RunContext) -> ModelArtifactRef: ...
    def score(self, model: ModelArtifactRef, features: TableRef, context: RunContext) -> TableRef: ...
```

Scores are not automatically calibrated probabilities or decisions.

## CandidateRanker

```python
class CandidateRanker(Protocol):
    def fit(self, training: RankingTrainingBundle, context: RunContext) -> ModelArtifactRef: ...
    def rank(self, model: ModelArtifactRef, candidates: TableRef, context: RunContext) -> TableRef: ...
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
