# Target Architecture

## Architectural principles

1. Configuration is compiled, not interpreted.
2. Source schemas are mapped to canonical variables at the IO boundary.
3. Row-level computation remains in local restricted tables.
4. Candidate retrieval, pair scoring, ranking, calibration, assignment, and decisions are separate authorities.
5. Outputs are deny-by-default and field-whitelisted.
6. Labels are provenance-controlled.
7. Synthetic tests prove behaviour, not operational validity.
8. Backends are adapters behind package-owned protocols.

## Planned package structure

```text
src/mapel_linkage/
├── configuration/          # Pydantic schema, loader, compiler, registries
├── domain/                 # enums, table refs, artifacts, decisions
├── governance/             # privacy, paths, labels, outputs, manifests
├── io/                     # local catalog, DuckDB, schema inspection
├── preprocessing/          # allow-listed canonical transforms
├── candidate_generation/   # safe blocking AST and compilers
├── comparisons/            # comparison functions and feature builder
├── models/
│   ├── fellegi_sunter/
│   ├── boosted/
│   ├── ranking/
│   ├── neural/
│   └── ensemble/
├── calibration/
├── assignment/
├── adjudication/
├── validation/
├── artifacts/
├── pipeline/
└── cli/
```

The current repository contains only the package shell. Planned modules are added milestone by milestone, with tests and contracts preceding full implementation.

## Execution flow

```text
ProjectConfig
  → validation and semantic checks
  → immutable ExecutionPlan
  → local dataset registration
  → run-local surrogate keys
  → canonical normalization
  → deterministic anchor evidence
  → bounded candidate generation
  → comparison features
  → pair model scores
  → candidate ranking
  → probability calibration
  → champion/challenger selection or ensemble
  → constrained assignment with no-match arcs
  → explicit decision policy
  → restricted outputs and review queue
  → aggregate validation and manifests
```

## Internal data references

Pipeline components exchange opaque references rather than row-bearing dataframes:

```python
@dataclass(frozen=True, slots=True)
class TableRef:
    table_name: str
    schema_digest: str
    row_count: int
    contains_row_level_data: bool = True
```

A `TableRef` must not implement row previews, `head()`, or value-rich `repr()` output. Developers explicitly enter the restricted data layer when local inspection is approved.

## Canonical source mapping

Source column names are read only from dataset/variable mapping configuration. The IO layer aliases them to generated internal columns such as:

```text
__ml_record_key
__ml_dataset_id
__ml_pair_key
__ml_v_<digest>
```

Model and orchestration code uses canonical variable IDs and internal columns, never project-specific names.

## Safe DSL compilation

Blocking and comparisons are typed trees:

```text
exact(variable)
prefix_equal(variable, length)
date_window(variable, maximum_days)
all(terms...)
any(terms...)
```

The engine owns compilation to DuckDB SQL and supported Splink rules. Users cannot submit SQL fragments. Identifiers are resolved from the canonical schema and quoted by the compiler.

## Decision record

The canonical relationship contract includes:

```text
relationship_id
source_dataset_id
target_dataset_id
source_record_ref
target_record_ref
relationship_status
model_family
model_version
calibrated_probability
candidate_rank
probability_margin
decision_rule_id
assignment_method
assignment_constraint
anchor_rule_ids
candidate_rule_ids
run_id
configuration_digest
feature_schema_digest
non_sensitive_provenance
created_at
```

Restricted outputs contain only fields whitelisted by configuration and policy.

## Trust boundaries

- configuration parser and compiler;
- path resolution and dataset registration;
- DuckDB connection and SQL execution;
- model fitting, loading, and serialization;
- logging and exception translation;
- artifact storage;
- restricted export;
- adjudication import/export;
- Git and CI.

Detailed controls are in `docs/governance/PRIVACY_THREAT_MODEL.md`.

## Reproducibility envelope

A run manifest records code revision, engine version, configuration digest, schema digest, label snapshot digest, random seed, dependency versions, platform, thread counts, model parameters, candidate rules, split manifest, and artifact digests.

Determinism means repeatability within a declared environment, not universal cross-platform bitwise identity.

## No implicit merge

The public pipeline ends at relationship decisions and review evidence. Entity consolidation, survivorship rules, source precedence, and master-record construction are out of scope until separately designed and approved.
