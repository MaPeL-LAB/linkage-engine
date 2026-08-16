# Configuration Reference

## Normative principles

- YAML or JSON only.
- Schema version is required.
- Unknown keys are rejected.
- Configuration is data, not executable code.
- Operations resolve through package-owned allow-list registries.
- Raw SQL and arbitrary Python callables are prohibited.
- Filesystem and outputs are default-deny.
- Configuration values are not printed in validation errors.

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

The package-generated JSON Schema becomes the normative machine-readable contract once M1 is implemented.
