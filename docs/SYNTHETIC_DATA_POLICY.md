# Synthetic Data Policy

**M2 status:** `mapel_linkage.synthetic` implements deterministic generic source
generation, separate truth, provenance, corruption, missingness, duplicates,
no-match cases, competitors, and assignment conflicts. The complete synthetic
vertical slice accepts only its package-generated fixture paths and
`synthetic_truth` label authority when `--synthetic-demo` is selected.

## Repository rule

Only generated synthetic record-level data may appear in the repository, tests, examples, documentation, notebooks, issues, pull requests, or CI.

De-identified, masked, hashed, tokenized, sampled, or perturbed real records are not repository-safe synthetic data.

The synthetic-demo flag is not permission to run an arbitrary configuration.
Before any fixture generation or dataset access, orchestration verifies the
exact project-local generated fixture paths, JSONL format, configured record
keys, synthetic label authority, absence of an external label path, and the
configured generator seed. Any mismatch fails closed.

## Generator requirements

The generator records:

- generator version;
- deterministic seed;
- population and source sizes;
- source-specific corruption parameters;
- missingness and duplicate parameters;
- no-match proportions;
- household/group structure;
- generator code revision.

Truth is generated before source corruption and stored separately from model inputs.

## Required corruption families

- insertion, deletion, substitution, and transposition;
- token reordering;
- punctuation, whitespace, case, and Unicode variation;
- date shifts and ambiguous transformations;
- numeric rounding;
- source-specific missingness;
- duplicate records;
- shared household-like values;
- conflicting high-specificity evidence;
- no-match entities;
- near-tied candidates;
- one-to-one assignment conflicts.

## Truth isolation

Truth fields never appear in the candidate, comparison, ranking, or model input schemas. Tests verify that removing the separate truth table does not change inference outputs.

## Sentinel testing

Every privacy test creates unique synthetic sentinel values and identifiers and asserts their absence from logs, errors, manifests, unrestricted output, and distributions.

## Benchmark interpretation

Synthetic benchmark scores are regression-test measurements under declared simulation assumptions. They are not estimates of real-world linkage quality.
