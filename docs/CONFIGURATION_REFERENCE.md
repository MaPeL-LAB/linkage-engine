# Configuration Reference

**Implementation status:** M1 schema and compiler implemented in `0.1.0.dev1`.

## Normative principles

- YAML or JSON only.
- Schema version is required.
- Unknown and duplicate keys are rejected.
- Configuration is data, not executable code.
- Operations resolve through package-owned allow-list registries.
- Raw SQL and arbitrary Python callables are prohibited.
- Filesystem and outputs are default-deny.
- Configuration values and arbitrary mapping keys are not printed in validation errors.

## Top-level structure

```text
schema_version
project
runtime
privacy
datasets
variables
deterministic_anchors
blocking
comparisons
labels
models
calibration
model_selection
assignment
decision_policy
validation
outputs
```

## Project

Required concepts:

- `project_id`
- `entity_type`
- `linkage_mode`: `link_only`, `dedupe_only`, `link_and_dedupe`, or `multi_source`
- `assignment_constraint`: `one_to_one`, `many_to_one`, `one_to_many`, or `unconstrained`
- `random_seed`

## Datasets

A dataset definition includes a stable config ID, role, local path, format, record-ID source column, and optional source metadata. Source column names are legal here and in variable mappings only.

Remote URI schemes are rejected by default. Paths are resolved and checked against approved roots.

## Variables

A variable definition includes:

- canonical variable ID;
- data type;
- per-dataset source column;
- allow-listed normalisation pipeline;
- missingness policy;
- sensitivity/output metadata.

## Transformation registry

Initial candidates include:

```text
strip
casefold
unicode_normalize
collapse_whitespace
parse_date
numeric_cast
```

Every transform declares supported input/output types and safe parameters.

## Blocking DSL

The public language uses semantic predicates such as:

```yaml
predicate:
  kind: all
  terms:
    - kind: exact
      variable: date_value
    - kind: prefix_equal
      variable: label_text
      length: 2
```

SQL is not part of the public configuration contract.

## Comparisons

Initial comparison families:

```text
exact
jaro_winkler
levenshtein
damerau_levenshtein
qgram
date_difference
numeric_difference
categorical
```

Each comparison declares levels, missingness behaviour, supported types, and output feature schema.

## Labels

The label source and verification metadata determine purpose-specific eligibility. `permit_unverified_crosswalk` is false and should not become a bypass.

## Models

Model sections are discriminated unions. A model selects an approved implementation key and validated parameters; it cannot specify an import path.

### Fellegi–Sunter baseline

The initial statistical baseline uses `implementation: splink_duckdb` and requires a stable model identifier. M2D validates and records:

- `probability_two_random_records_match`;
- `u_max_pairs`;
- `em_max_iterations`;
- `em_convergence`;
- `probability_smoothing`;
- `estimate_u_by_random_sampling: true`;
- `estimate_m_by_em: true`;
- `term_frequency_adjustments: false` for the initial reference model.

The `u_max_pairs` value may not exceed the runtime candidate-pair budget. The initial score output is explicitly `model_posterior_uncalibrated` and `evidence_only`; it cannot satisfy a confirmation rule or bypass calibration, assignment, or the decision policy. Term-frequency adjustment remains a later M2D extension rather than an implied capability.

## Calibration

Calibration declares source model, method, partition, and independence requirement. Supported initial methods are sigmoid and isotonic. Beta calibration is a later challenger.

## Assignment

Assignment declares solver, constraint, utility transform, no-match policy, capacities, and deterministic tie-breaking.

## Decision policy

The policy defines non-overlapping criteria for `confirmed`, `review_required`, `no_match`, and fallback `unresolved`.

## Outputs

`permitted_fields` is an allow-list. Variable values require separate permission. Restricted output directories must resolve under approved local roots.

## Cross-field rejection cases

Validation fails for duplicate IDs, unknown references, type-incompatible operations, unsafe paths/URIs, invalid dataset counts for linkage mode, supervised models without eligible labels, overlapping partitions, invalid threshold ordering, unbounded Cartesian candidates, candidate-budget violations, and reserved internal source columns.

## Machine-readable schema

The loader also rejects YAML merge keys, excessive aliases, excessive nesting or node counts, non-string mapping keys, non-finite JSON numbers, and non-JSON-compatible YAML scalar types.

The normative machine-readable contract is committed at:

```text
schemas/linkage-config.schema.json
```

It is generated directly from `LinkageConfig` by `scripts/generate_config_schema.py`. The test suite compares the committed schema with the live Pydantic schema.

## Compilation result

A valid configuration compiles to an immutable `ExecutionPlan` containing configuration and registry digests, aggregate counts, the random seed, hidden resolved dataset paths, the hidden restricted output directory, and the path policy. `repr()` and `safe_summary()` do not expose paths, source columns, project IDs, or dataset IDs.

## Host path envelope

Configured roots do not authorize themselves. They must also fit inside roots supplied by the host application. The CLI's M1 default envelope is:

```text
input:  data/, private/
output: private/, artifacts/
```

A later deployment may pass different host-approved roots explicitly through trusted Python startup code. Project YAML cannot widen that envelope.
