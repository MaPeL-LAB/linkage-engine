# Configuration Reference

**Implementation status:** complete two-source synthetic MVP contract in `0.2.0.dev0`.

## Normative principles

- YAML or JSON only.
- Schema version is required.
- Unknown and duplicate keys are rejected.
- Configuration is data, not executable code.
- Operations resolve through package-owned allow-list registries.
- Raw SQL, shell commands, import paths, arbitrary Python callables, `eval()`, and `exec()` are prohibited.
- Filesystem access and exported fields are default-deny.
- Configuration values, local paths, source columns, and arbitrary mapping keys are not printed in public validation errors.
- A completed real project configuration is protected local material and must remain Git-ignored.

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

Required fields:

- `project_id`
- `entity_type`
- `linkage_mode`: `link_only`, `dedupe_only`, `link_and_dedupe`, or `multi_source`
- `assignment_constraint`: `one_to_one`, `many_to_one`, `one_to_many`, or `unconstrained`
- `random_seed`

The complete synthetic MVP currently executes `link_only` with `one_to_one` assignment. Other validated enum values are reserved for later mode-specific implementations and fail safely where no runtime implementation exists.

## Runtime and privacy

The runtime section declares DuckDB, deterministic execution, and a hard candidate-pair budget. Run directories are supplied by the trusted host/orchestrator rather than project YAML. The privacy section declares requested local roots and aggregate-only logging. Requested roots must also fall within a separate host-approved path envelope supplied by trusted startup code.

Remote URI schemes, UNC paths, traversal outside approved roots, project-root widening, network access, and public tracebacks are rejected.

## Datasets

A dataset definition includes:

- stable configuration ID;
- role;
- local path;
- approved format;
- source record-ID column.

Current readers support Parquet, CSV, TSV, and newline-delimited JSON. Source column names are legal only in dataset declarations, variable mappings, and protected local truth declarations. They are never embedded in model, assignment, decision, or orchestration logic.

## Variables

A variable definition includes:

- canonical variable ID;
- data type;
- per-dataset source-column mapping;
- allow-listed normalisation pipeline;
- missingness policy;
- restricted-output permission metadata.

Current data types include string, categorical, date, numeric, integer, and Boolean. Every prepared canonical variable receives an explicit missingness indicator.

## Transformation registry

Current allow-listed operations include:

```text
strip
casefold
unicode_normalize
collapse_whitespace
parse_date
numeric_cast
```

Each operation has fixed validated parameters and type compatibility. Unknown operations fail before data access.

## Blocking DSL

The public language uses semantic predicates rather than SQL:

```yaml
predicate:
  kind: all
  terms:
    - kind: exact
      variable: date_attribute
    - kind: prefix_equal
      variable: text_attribute
      length: 2
```

Current predicates include exact equality, prefix equality, date windows, conjunction, and disjunction. Package code compiles the same supported predicate subset to DuckDB and Splink. The synthetic MVP requires exact candidate-pair parity between those paths.

## Comparisons

Current comparison families are:

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

Each comparison declares ordered levels, explicit missingness behaviour, supported variable types, and a deterministic internal feature schema. The final level must be `else`.

## Deterministic anchors

Anchors evaluate exact or bounded package-owned predicates with uniqueness and contradiction evidence. The only supported action is initially:

```text
action: evidence_only
allow_as_training_truth: false
```

An anchor cannot silently create a confirmed relationship or training label.

## Labels

Supported sources are:

```text
synthetic_truth
verified_human_adjudication
verified_gold_standard
unverified_reference
```

Synthetic truth declares entity and optional household grouping columns. Verified adjudication and verified gold-standard sources declare a protected local path plus a protocol version. An unverified reference may be retained as evidence but is statically ineligible for training, validation, calibration, threshold selection, and testing.

The following safeguards are fixed and cannot be disabled:

```yaml
permit_weak_labels_for_training: false
permit_unverified_crosswalk: false
```

## Models

### Fellegi–Sunter

The initial baseline uses `implementation: splink_duckdb` and records:

- prior match probability;
- random-pair budget for `u` estimation;
- EM limits for `m` estimation;
- probability smoothing;
- deterministic seed and artifact digests.

The engine-owned reference estimator produces `model_posterior_uncalibrated` evidence. The Splink adapter compiles the safe configuration and verifies candidate parity. Neither path has decision authority.

### Boosted-tree pair classifier

The initial supervised model uses `implementation: xgboost_classifier` with:

- eligible verified labels only;
- bounded deterministic hard-negative selection;
- fixed single-thread execution;
- bounded estimators, depth, learning rate, subsampling, and training-pair count;
- native JSON model artifacts with integrity digests.

The score remains uncalibrated evidence until the protected calibration stage.

### Candidate ranker

The initial ranker uses `implementation: xgboost_ranker` and declares:

- `query_side`;
- `top_k`;
- verified-label requirement;
- bounded deterministic training controls.

Ranker outputs contain scores, ranks, top-K membership, model identity, and artifact provenance only. They cannot contain relationship statuses, assignment authority, or merge instructions.

### Neural matcher

`pytorch_pair_mlp` remains an optional, disabled future implementation. It is not required for the synthetic MVP.

## Champion–challenger selection

```yaml
model_selection:
  mode: champion_challenger
  selection_partition: validation
  primary_metric: average_precision
  test_partition_may_select_model: false
```

The validation partition compares Fellegi–Sunter and XGBoost aggregate evidence. The test partition cannot select a model. Selection artifacts retain model identities, evidence digests, label authority, partition evidence, metrics, and a deterministic selection digest.

## Calibration

```yaml
calibration:
  method: sigmoid
  source_model: selected_champion
  partition: calibration
  require_independent_partition: true
```

Supported methods are sigmoid and isotonic. Calibration uses only the protected calibration partition and writes native JSON payload/manifest artifacts with tamper detection. A calibrated probability still has `decision_authority: evidence_only` until the separate decision stage.

## Assignment

```yaml
assignment:
  solver: ortools_min_cost_flow
  constraint: one_to_one
  deterministic_tie_breaking: true
  no_match:
    enabled: true
    utility: 0.0
```

The synthetic MVP supports sparse one-to-one assignment with a private no-match option for each source record. SciPy provides a small-problem oracle. Assignment selects a globally compatible real or no-match edge but cannot emit a relationship status.

## Decision policy

The policy contains non-overlapping regions for:

```text
confirmed
review_required
no_match
unresolved
```

`no_match` requires a complete candidate search and an explicit no-match assignment with no plausible candidate. Candidate truncation, invalid calibration, critical data-quality failure, unsupported execution, or incomplete retrieval produces `unresolved`, not `no_match`.

Decision thresholds fitted from the synthetic decision partition are labelled `synthetic_benchmark_only` and are not operational recommendations.

## Validation

The split method is `entity_household_connected_components` with five positive fractions:

```text
training
validation
calibration
decision
test
```

The fractions must sum to one. Entities and households cannot cross partitions. Hard negatives must be verified nonmatches. `candidate_recall_k` must be unique and ascending.

The complete synthetic report includes candidate retrieval, pair discrimination, precision–recall points, calibration, ranking, assignment, decision, missingness-pattern, candidate-set-size, and versioned synthetic regression diagnostics.

## Outputs

`permitted_fields` is a strict allow-list. The supported relationship/review fields are enumerated in the generated JSON Schema, including `review_reason_codes`. Canonical variable values require separate permission in `permitted_variable_values`.

Restricted directories must resolve under approved output roots. Unrestricted manifests contain only aggregate counts, versions, methods, statuses, and digests.

## Cross-field rejection cases

Validation fails for, among other conditions:

- duplicate dataset, variable, rule, comparison, or model IDs;
- unknown references;
- type-incompatible transforms, predicates, comparisons, or levels;
- unsafe paths and URIs;
- invalid dataset counts or roles for linkage mode;
- supervised models without eligible labels;
- calibration of a disabled/unknown source model;
- assignment/project constraint disagreement;
- overlapping decision probability regions;
- invalid protected split fractions;
- candidate or training budgets above the runtime limit;
- output fields or variable values outside their allow-lists;
- reserved internal source columns.

The loader also rejects YAML merge keys, excessive aliases, excessive nesting or node counts, duplicate keys, non-string mapping keys, non-finite JSON numbers, recursive values, and non-JSON-compatible YAML scalars.

## Machine-readable schema

The normative machine-readable contract is:

```text
schemas/linkage-config.schema.json
```

It is generated directly from `LinkageConfig` by `scripts/generate_config_schema.py`. Tests require byte-equivalent semantic parity between the committed schema and the live Pydantic model.

## Compilation result

A valid configuration compiles to an immutable `ExecutionPlan` containing configuration and registry digests, aggregate counts, the seed, hidden resolved dataset paths, the hidden restricted-output directory, and the path policy. Public representations do not expose paths, source columns, project IDs, dataset IDs, or submitted values.

## Local template

Copy the generic worksheet only inside the authorised local environment:

```text
configs/templates/local_project.template.yaml
```

The completed copy belongs under `private/config/` and must never be committed.
