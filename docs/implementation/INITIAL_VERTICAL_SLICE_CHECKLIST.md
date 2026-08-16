# Initial Synthetic Vertical Slice Checklist

## Repository foundation

- [ ] Python 3.12 package uses `src/mapel_linkage`.
- [ ] `pyproject.toml` builds wheel and source distribution.
- [ ] CLI entry point is `mapel-linkage`.
- [ ] private, data, and artifact paths are ignored.
- [ ] package publication remains blocked.
- [ ] Ruff, mypy, pytest, pre-commit, and repository verification pass.

## Configuration boundary

- [ ] YAML and JSON load through strict Pydantic models.
- [ ] unknown keys fail.
- [ ] raw SQL fails.
- [ ] Python/module/callable expressions fail.
- [ ] all configurable operations use allow-list registries.
- [ ] dataset-specific columns occur only in project configuration and IO mapping.
- [ ] paths resolve inside approved roots.
- [ ] output fields default to deny.

## Synthetic input

- [ ] deterministic seed is recorded.
- [ ] generator produces source-specific corruption.
- [ ] truth is held separately from linkage inputs.
- [ ] no truth field is exposed to linkage models.
- [ ] cases include missingness, duplicates, no-match records, competing candidates, and assignment conflicts.

## Linkage pipeline

- [ ] preprocessing operates on canonical variables.
- [ ] deterministic anchors are evidence-only by default.
- [ ] candidate generation has a hard pair budget.
- [ ] candidate rule provenance is recorded.
- [ ] DuckDB and supported Splink blocking paths have parity tests.
- [ ] Splink Fellegi–Sunter baseline runs.
- [ ] XGBoost pair classifier runs on comparison features.
- [ ] candidate ranker emits top-K only.
- [ ] probabilities are calibrated on disjoint data.
- [ ] global one-to-one assignment includes no-match.
- [ ] decision policy emits exactly one supported status.
- [ ] no model merges records.

## Validation

- [ ] entity/household groups do not cross partitions.
- [ ] hard negatives use eligible labels only.
- [ ] candidate recall@K is reported.
- [ ] sensitivity and PPV are reported.
- [ ] false-link and missed-link rates are reported.
- [ ] precision–recall curve is generated.
- [ ] Brier score and reliability diagnostics are generated.
- [ ] ranking and assignment accuracy are reported.
- [ ] missingness-pattern metrics are reported.
- [ ] candidate-set-size metrics are reported.
- [ ] synthetic corruption regression tests pass.

## Privacy

- [ ] sentinel participant-like values never occur in logs.
- [ ] record identifiers never occur in logs.
- [ ] candidate pairs never occur in logs.
- [ ] validation errors hide input values.
- [ ] backend errors are sanitized.
- [ ] unrestricted manifests contain no row values.
- [ ] CI generates synthetic inputs at runtime.
- [ ] CI uploads no row-level artifacts.

## Acceptance statement

Passing this checklist establishes a working synthetic software slice only. It is not evidence that linkage models are validated for real data.
