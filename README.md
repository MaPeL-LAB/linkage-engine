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
| Current package version | `0.1.0.dev3` |

The repository name is **`linkage-engine`**. `MaPeL-LAB` identifies the developer and GitHub organisation; it is not part of the repository name.

## Status

Milestones **M1**, **M2A**, and **M2B** are implemented in the stacked development branches. The package now provides the safe configuration foundation, a local DuckDB data plane, typed bounded candidate generation, and configuration-driven local ingestion with canonical preprocessing.

M2B reads approved local Parquet, CSV, TSV, and newline-delimited JSON sources; maps source columns through validated configuration; creates deterministic surrogate record keys; applies allow-listed normalisation; and materialises canonical local DuckDB tables with explicit missingness indicators.

The package does **not** yet implement comparison-feature construction, deterministic-anchor evidence evaluation, Fellegi–Sunter scoring, supervised matching, calibration, assignment, adjudication processing, or a complete linkage run. It is not validated for operational use.

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

## Command line

Implemented:

```text
mapel-linkage status
mapel-linkage validate-config --config CONFIG --project-root ROOT
mapel-linkage emit-config-schema --output OUTPUT
```

Reserved target interfaces that still return an explicit pre-alpha error:

```text
mapel-linkage generate-candidates --config CONFIG
mapel-linkage train --config CONFIG
mapel-linkage predict --config CONFIG
mapel-linkage assign --config CONFIG
mapel-linkage evaluate --config CONFIG
mapel-linkage run --config CONFIG
```

Successful configuration validation reports only a digest prefix and aggregate counts. It does not print the configuration path, source columns, project ID, dataset IDs, or submitted values.

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
python -m pip install -e ".[dev]"
python scripts/generate_config_schema.py
ruff format --check .
ruff check .
mypy src tests
pytest
python scripts/verify_repository.py
python -m build
python scripts/verify_repository.py --distribution dist
```

Install the planned scientific core only when working on M2:

```bash
python -m pip install -e ".[core]"
```

## Documentation

The documentation index is [`docs/README.md`](docs/README.md). Implementation reports are indexed in [`docs/README.md`](docs/README.md), including M1, M2A, and M2B. Research claims use keys from [`docs/references/references.bib`](docs/references/references.bib).

## Validation warning

> **Synthetic testing establishes software behaviour only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.**

## Publication and licence

The distribution is marked `Private :: Do Not Upload`. Publishing, public release, repository visibility changes, and licence selection require explicit MaPeL-LAB approval. See [`docs/governance/LICENSING_DECISION.md`](docs/governance/LICENSING_DECISION.md).
