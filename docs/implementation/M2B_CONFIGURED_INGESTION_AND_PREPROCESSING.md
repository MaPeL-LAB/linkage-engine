# M2B Configured Ingestion and Canonical Preprocessing

**Status:** Implemented development increment  
**Package version:** `0.1.0.dev3`  
**Authority:** Data preparation only

## Purpose

M2B connects the validated M1 `ExecutionPlan` to the local M2A DuckDB data plane. It prepares heterogeneous local datasets for later candidate retrieval and comparison without embedding study-specific source columns in package code.

## Delivered components

### Local source readers

The internal DuckDB adapter reads only configured columns from approved local files:

- Parquet;
- CSV;
- TSV;
- newline-delimited JSON.

File paths have already passed the M1 host-and-project path policy. Reader SQL is package-owned and uses bound file-path parameters. Source columns are quoted as identifiers and cannot supply SQL expressions.

Direct arbitrary DuckDB source attachment is deliberately deferred because database attachment introduces a broader trust and filesystem surface.

### Canonical variable mapping

For each dataset, configuration maps source columns to generic variable IDs. M2B generates stable internal columns from the variable ID digest:

```text
__ml_v_<digest>    canonical value
__ml_m_<digest>    explicit missingness indicator
```

The same variable ID receives the same internal column name across datasets even when the source column names differ completely.

### Surrogate record keys

The canonical record reference is:

```text
SHA-256(dataset_id + separator + configured_record_identifier)
```

The original record identifier is not copied into the canonical table. Missing and duplicate configured record identifiers fail preparation with stable value-safe errors.

The surrogate is deterministic within the declared configuration and source identity. It is an internal linkage reference, not an anonymisation guarantee and not a licence to export source-derived identifiers.

### Allow-listed normalisation

M2B implements only configuration-schema-approved operations:

- Unicode NFC/NFKC normalisation;
- case folding;
- trimming;
- whitespace collapsing;
- configured date parsing;
- integer and floating-point conversion;
- bounded Boolean conversion.

Invalid values produce `ML-PREP-006` without including the submitted value, source column, or local path.

### Prepared dataset contracts

`PreparedDataset` exposes only:

- dataset ID;
- opaque local `TableRef`;
- immutable variable-to-internal-column mapping;
- immutable missingness-column mapping.

`PreparedDatasetCatalog` provides immutable lookup and aggregate row/dataset counts. Neither object provides row previews.

## Privacy and safety controls

- Real records remain local and Git-ignored.
- Tests and CI generate synthetic inputs only.
- Source paths, source columns, record IDs, and row values are excluded from public representations.
- Backend exceptions are translated to stable `ML-DATA-*` or `ML-PREP-*` errors.
- Configuration remains data, not executable code.
- No raw SQL, callable path, dynamic import, `eval()`, or `exec()` is accepted.
- M2B makes no match, nonmatch, assignment, or merge decision.

## Verification

The M2B test suite covers:

- CSV and JSONL configured ingestion;
- common canonical mappings across renamed source schemas;
- Unicode/text/date/missingness normalisation;
- stable surrogate references;
- duplicate and missing record-ID rejection;
- deferred direct DuckDB attachment;
- value-safe path, column, and row failures;
- absence of source-specific example columns from package logic.

Remote CI additionally verifies Python 3.12, Ruff, strict mypy, pre-commit, the complete synthetic test suite, wheel/sdist builds, and restricted-distribution contents.

## Explicit limitations

M2B does not yet provide:

- streaming or out-of-core preprocessing for very large files;
- direct DuckDB database attachment;
- spreadsheet ingestion;
- comparison-feature construction;
- deterministic-anchor evidence;
- Fellegi–Sunter scoring;
- probability calibration;
- assignment or relationship decisions.

## Next increment

M2C will construct configured comparison features and deterministic-anchor evidence over M2B canonical tables and M2A candidate pairs. Anchors remain evidence-only and ineligible as training truth by default.

> Synthetic testing establishes software behaviour only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.
