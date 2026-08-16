# Acceptance Criteria

## Architecture and configuration

- All source-column access occurs through compiled mappings.
- Renaming every synthetic source column and updating configuration alone preserves behaviour.
- Unknown configuration fields fail.
- Raw SQL, import paths, executable expressions, and unsupported callables fail.
- All operations resolve through immutable allow-list registries.
- JSON Schema is generated and example configurations validate.
- Canonical configuration receives a stable digest.

## Privacy

- Synthetic sentinel values do not appear in stdout, stderr, logs, exceptions, manifests, model metadata, or unrestricted reports.
- Candidate pairs are never printed.
- Original identifiers are absent from default intermediate/output schemas.
- Restricted exports contain exactly approved fields.
- CI generates all row-level inputs and uploads no row-level artifacts.
- No runtime network operation is implemented in the core package.
- Unsafe model deserialization is rejected.

## Functional behaviour

- Every target CLI command has an integration test when implemented.
- Stage artifacts are immutable and carry matching digests.
- Candidate generation enforces pair budgets.
- One-to-one assignment produces zero capacity violations.
- Every required query-side record receives a supported outcome.
- No command produces a consolidated entity or silently merges records.

## Model governance

- Supervised models refuse ineligible labels.
- Unknown pairs are not implicit nonmatches.
- Unverified crosswalks cannot enter train, calibration, decision, or test truth.
- Candidate ranking cannot emit relationship status.
- Uncalibrated model scores cannot satisfy default model-based confirmation.
- Test data do not select models, hyperparameters, blocking rules, calibration, or thresholds.
- Every model artifact retains versions, seed, feature schema, config digest, label provenance, and partition IDs.

## Validation

- Entity/household connected groups do not cross partitions.
- Candidate recall is measured before scoring.
- Pair, ranking, calibration, assignment, missingness, and candidate-size metrics are produced.
- CI compares synthetic metrics against a versioned regression baseline.
- Every synthetic report includes the standard validation warning.

## Engineering quality

```bash
ruff format --check .
ruff check .
mypy src tests
pytest
python scripts/verify_repository.py
python -m build
```

A check may be marked unavailable only with an explicit, accurate limitation report.
