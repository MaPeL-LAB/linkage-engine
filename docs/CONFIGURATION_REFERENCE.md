# Configuration Reference

**Implementation status:** complete two-source synthetic MVP plus bounded I1C synthetic-mode
contract in `0.2.0.dev3`.

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
mode_orchestration
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

The legacy complete synthetic MVP executes `link_only` with `one_to_one` assignment. The
separate I1C synthetic-mode route executes only five exact allow-listed combinations:

```text
link_only + many_to_one
link_only + one_to_many
link_only + unconstrained
dedupe_only + unconstrained
link_and_dedupe + one_to_one
```

Every other combination, including `multi_source`, fails before execution. The I1C route does
not accept real data and does not establish operational validity.

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

### Boosted-tree pair classifiers

Singular `boosted_tree` remains backward compatible. Plural `boosted_trees` accepts bounded
`xgboost_classifier` and `lightgbm_classifier` candidates with:

- eligible verified labels only;
- bounded deterministic hard-negative selection;
- fixed single-thread execution;
- bounded estimators, depth, learning rate, subsampling, and training-pair count;
- native JSON model artifacts with integrity digests.

The score remains uncalibrated evidence until the protected calibration stage.

### Candidate rankers

Singular `ranking` remains backward compatible. Plural `ranking_models` accepts bounded
`xgboost_ranker` and `lightgbm_ranker` candidates and declares:

- `query_side`;
- `top_k`;
- verified-label requirement;
- bounded deterministic training controls.

Ranker outputs contain scores, ranks, top-K membership, model identity, and artifact provenance only. They cannot contain relationship statuses, assignment authority, or merge instructions.
The current source-to-target recipe can execute only `query_side: source`; target-query
artifacts are trained and reported but rejected at this inference boundary.

### Neural matcher

Plural `neural_models` accepts optional `pytorch_pair_mlp` candidates with bounded `epochs`,
`learning_rate`, `weight_decay`, `maximum_training_pairs`, `device: cpu`, `n_threads`, and
`deterministic_mode`. The matcher consumes comparison features only. It has no raw-text,
identity, relationship-decision, or merge authority.

### Stacking and portfolio selection

Plural `ensembles` accepts `stacking_logistic` with explicit replayable supervised
`base_model_ids`. Native Splink cannot be assigned a false OOF claim or used as a generic
feature-matrix stacking base. `models.portfolio` binds the mandatory baseline, enabled pair
and ranking candidate IDs, challenger budget, and no-authority rules under schema version 1.

## Champion–challenger selection

```yaml
model_selection:
  mode: champion_challenger
  selection_partition: validation
  primary_metric: average_precision
  test_partition_may_select_model: false
```

The validation partition compares every eligible configured candidate. The test partition
cannot select a model. Selection artifacts retain model identities, evidence digests, label
authority, partition evidence, metrics, and a deterministic selection digest.

## Calibration

```yaml
calibration:
  method: sigmoid
  source_model: selected_champion
  partition: calibration
  require_independent_partition: true
```

Supported methods are sigmoid, isotonic, and Beta. Calibration uses only the protected
calibration partition and writes native JSON payload/manifest artifacts with tamper detection.
A calibrated probability still has `decision_authority: evidence_only` until the separate
decision stage.

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

## Mode orchestration

I1C mode execution is explicitly enabled and bound to package-owned implementations:

```yaml
mode_orchestration:
  artifact_schema_version: "1"
  implementation: synthetic_mode_v1
  pair_model_id: xgb_pair_classifier
  deduplication:
    algorithm: clique
    minimum_probability: 0.75
    no_match_utility: 0.0
    maximum_cluster_size: 100
    maximum_candidate_edges: 100000
    deterministic_tie_breaking: true
```

`deduplication` is required for `dedupe_only` and `link_and_dedupe`, and forbidden for the
three extended `link_only` combinations. The route also requires seed `20260816`, the enabled
XGBoost pair classifier, calibration bound to that declared pair-model identifier, and zero
assignment no-match utility. Unknown implementations, algorithms, schema versions, modes,
constraints, artifact digests, or provenance bindings fail closed.

Same-source candidate retrieval is deterministic, removes self-pairs, and canonicalises
symmetric pairs before comparison features are built. All three `link_and_dedupe` pair
surfaces share one protected entity/household partition assignment and one combined fitted and
calibrated model authority. Cross-source-only calibration cannot authorise same-source
clustering. Only decision-partition feature evidence is scored for linkage or clustering;
training, validation, calibration, and locked-test pairs are never relationship-decision
inputs. Dedupe-only and link-and-dedupe modes emit aggregate assignment/cluster evidence
without relationship statuses, record values, decision authority, or merge authority.

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
