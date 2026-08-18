# Linkage Engine

`mapel-linkage-engine` is a pre-alpha Python package for configurable probabilistic record linkage, entity resolution, and within-dataset deduplication.

| Item | Canonical value |
|---|---|
| Developer / GitHub organisation | `MaPeL-LAB` |
| Repository | `linkage-engine` |
| Python distribution | `mapel-linkage-engine` |
| Import package | `mapel_linkage` |
| Command-line interface | `mapel-linkage` |
| Initial Python runtime | Python 3.12 |
| Current package version | `0.2.0.dev0` |

The repository name is **`linkage-engine`**. `MaPeL-LAB` identifies the developer and GitHub organisation; it is not part of the repository name.

## Status

Milestones **M0**, **M1**, and the complete **M2 synthetic vertical slice** are implemented. The package now provides strict configuration compilation, local DuckDB preparation, bounded candidate retrieval, comparison and anchor evidence, Fellegi–Sunter and verified-label XGBoost pair models, validation-only champion selection, independent probability calibration, XGBoost candidate ranking, one-to-one assignment with an explicit no-match option, four relationship outcomes, restricted review export, aggregate evaluation, and deterministic orchestration.

The complete workflow is approved for generated synthetic software testing only. It is not validated for operational use, and it does not create or silently merge master records. Real data, completed local configurations, verified operational truth, adjudication records, model artefacts, and outputs remain local and Git-ignored.

## Intended use

The package is designed to support, without study-specific assumptions:

- study-to-population-registry linkage;
- clinic-to-HDSS linkage;
- study-to-study linkage;
- registry-to-clinical-system linkage;
- multi-source entity resolution;
- within-dataset deduplication.

Source column names are accepted only through validated dataset and variable mappings. They are not embedded in model, comparison, assignment, calibration, decision, or orchestration logic.

## Non-negotiable privacy boundary

Only synthetic record-level data may appear in this repository, its documentation, examples, tests, notebooks, issues, pull requests, or continuous integration.

Real participant or operational data, identifiers, project configurations, adjudication records, secrets, model artefacts, candidate pairs, and linkage outputs must remain local under ignored directories. De-identified, hashed, tokenised, masked, sampled, or perturbed real records are **not** considered synthetic for repository purposes.

The package must never print or log record values, source identifiers, secrets, candidate pairs, training examples, or adjudication values. An existing crosswalk is not training truth unless independently verified under the label-provenance policy.

See:

- [`docs/governance/PRIVACY_THREAT_MODEL.md`](docs/governance/PRIVACY_THREAT_MODEL.md)
- [`docs/SYNTHETIC_DATA_POLICY.md`](docs/SYNTHETIC_DATA_POLICY.md)
- [`docs/governance/LABEL_PROVENANCE_POLICY.md`](docs/governance/LABEL_PROVENANCE_POLICY.md)

## Implemented trust and data-preparation boundary

> **Configuration is data, not executable code.**

M1 includes:

- strict, immutable Pydantic models with unknown fields forbidden;
- safe YAML/JSON loading with size, alias, duplicate-key, merge-key, depth, node-count, and scalar controls;
- value-safe validation translation that hides submitted values and arbitrary mapping keys;
- cross-field validation for linkage modes, variables, operations, labels, enabled models, comparison levels, thresholds, partitions, and outputs;
- a package-owned typed blocking/comparison DSL;
- immutable allow-list registries and frozen configuration mappings with no configured import or callable resolution;
- canonical configuration and registry digests;
- project and host path envelopes with remote URI, UNC, traversal, and root-widening protection;
- deny-by-default output fields and restricted variable-value permission;
- typed aggregate-only logging;
- privacy-safe run manifests containing versions, digests, counts, and seeds only;
- deterministic synthetic source generation with separately held truth;
- machine-readable schema at [`schemas/linkage-config.schema.json`](schemas/linkage-config.schema.json).

No configuration may provide raw SQL, a shell command, a module path, a Python callable, `eval()`, `exec()`, or arbitrary executable content.

## Comparison and anchor evidence

`DuckDBComparisonFeatureBuilder` joins only bounded M2A candidate keys to M2B canonical tables. It emits package-generated comparison values, configured level indices, exact-agreement flags, explicit missingness flags, and retrieval provenance. It does not copy source field values into the feature output.

`DuckDBAnchorEvidenceEvaluator` evaluates exact, prefix, date-window, conjunction, and disjunction predicates independently of pair scoring. Its output records aggregate uniqueness evidence and fixes both:

```text
evidence_action = evidence_only
eligible_as_training_truth = false
```

Neither component emits a probability, relationship status, assignment, merged entity, or master record.

## Configured local preparation

`ConfiguredDatasetPreparer` consumes a compiled `ExecutionPlan` and prepares local row-bearing tables without exposing source rows through public interfaces.

The canonical table contract includes:

```text
__ml_dataset_id
__ml_record_key
__ml_v_<stable-variable-digest>
__ml_m_<stable-variable-digest>
```

Original record identifiers are used only to derive deterministic SHA-256 surrogate keys and are not retained as the canonical record reference. Source paths, source column names, identifiers, and values are excluded from public object representations and translated errors. Direct attachment of arbitrary DuckDB databases remains deferred.

## Fellegi–Sunter evidence baseline

M2D adds deterministic bounded random-pair sampling, smoothed `u` estimation,
aggregate-vector expectation–maximisation for `m`, per-level log2 Bayes factors,
and local evidence scoring. The package also compiles validated canonical
configuration into a Splink 4 settings plan.

M2D probabilities are explicitly `model_posterior_uncalibrated` and
`evidence_only`. They cannot confirm identity, select thresholds, perform
assignment, or merge records.

See
[`docs/implementation/M2D_FELLEGI_SUNTER_BASELINE.md`](docs/implementation/M2D_FELLEGI_SUNTER_BASELINE.md).

## Verified-label XGBoost challenger

M2E accepts only synthetic truth, verified adjudication, or an independently
verified gold standard for supervised use. Pair, entity, and household
components cannot cross protected partitions. Unknown pairs remain unknown.

The XGBoost challenger consumes only package-generated comparison and
missingness features. Training is bounded, single-threaded, seeded, and may
prioritise verified hard nonmatches. Native JSON model artifacts retain safe
aggregate manifests and exact feature-schema, label-authority, selection,
parameter, and model digests.

All M2E scores remain `model_score_uncalibrated`, `not_calibrated`, and
`evidence_only`. Diagnostic thresholds have no operational decision authority.

See
[`docs/implementation/M2E_VERIFIED_LABEL_XGBOOST_CHALLENGER.md`](docs/implementation/M2E_VERIFIED_LABEL_XGBOOST_CHALLENGER.md).

## Complete synthetic MVP

The installed synthetic workflow preserves separate authority boundaries:

```text
candidate retrieval → comparison evidence → pair scoring → champion selection
→ calibration → ranking → assignment → relationship decision → review/evaluation
```

Candidate retrieval does not decide identity. Pair and ranking models remain evidence-only. Ranking has no relationship authority. Assignment performs global selection but does not classify relationships. The explicit decision layer alone emits `confirmed`, `review_required`, `unresolved`, or `no_match`, and no stage has merge authority.

See [`docs/implementation/M2_COMPLETE_SYNTHETIC_MVP.md`](docs/implementation/M2_COMPLETE_SYNTHETIC_MVP.md).

## Command line

Environment and configuration commands:

```text
mapel-linkage status
mapel-linkage doctor --project-root ROOT
mapel-linkage init-local-project --directory ROOT
mapel-linkage validate-config --config CONFIG --project-root ROOT
mapel-linkage emit-config-schema --output OUTPUT
```

Complete generated-synthetic workflow commands:

```text
mapel-linkage generate-candidates --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage train --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage predict --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage assign --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage evaluate --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage run --config CONFIG --project-root ROOT --synthetic-demo
```

Stage commands execute their required upstream stages and print aggregate-only summaries. The repository build refuses row-level execution without `--synthetic-demo`; operational records and configurations must remain in the authorised local environment.

## Synthetic generator

`mapel_linkage.synthetic.generate_synthetic_bundle()` creates deterministic generic source tables with:

- source-specific corruption;
- missing values;
- duplicates and one-to-one assignment conflicts;
- source-only no-match records;
- competing candidates;
- truth held in a separate test-only structure.

Generated rows and truth records use value-hiding representations and are never stored in the repository.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/ci-py312.txt -e ".[core,dev]"
python scripts/generate_config_schema.py
ruff format --check .
ruff check .
mypy src tests
pytest
python scripts/verify_repository.py
python -m build
python scripts/verify_repository.py --distribution dist
```

The local bootstrap scripts in `scripts/` install the tested Python 3.12 scientific and development envelope and run the synthetic smoke test.

## Documentation

The documentation index is [`docs/README.md`](docs/README.md). Implementation reports are indexed in [`docs/README.md`](docs/README.md), including M1 through M2E. Research claims use keys from [`docs/references/references.bib`](docs/references/references.bib).

## Validation warning

> **Synthetic testing establishes software behaviour only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.**

## Publication and licence

The distribution is marked `Private :: Do Not Upload`. Publishing, public release, repository visibility changes, and licence selection require explicit MaPeL-LAB approval. See [`docs/governance/LICENSING_DECISION.md`](docs/governance/LICENSING_DECISION.md).
