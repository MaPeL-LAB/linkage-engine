# Initial Synthetic Vertical Slice Checklist

The checked M1 items below are implemented in the `0.1.0.dev1` development candidate. The complete vertical slice still requires M2.

## Repository foundation

- [x] Python 3.12 package uses `src/mapel_linkage`.
- [ ] Current M1 branch builds wheel and source distribution in Python 3.12 CI.
- [x] CLI entry point is `mapel-linkage`.
- [x] private, data, and artifact paths are ignored.
- [x] package publication remains blocked.
- [ ] Ruff, mypy, pytest, pre-commit, repository verification, and build all pass on the current remote branch.

## Configuration boundary

- [x] YAML and JSON load through strict Pydantic models.
- [x] unknown and duplicate keys fail.
- [x] raw SQL fields fail.
- [x] Python, module, callable, and import-path fields fail.
- [x] all configurable operations use immutable allow-list registries.
- [x] dataset-specific columns occur only in project configuration and IO mapping contracts.
- [x] configured paths resolve inside project and host-approved roots.
- [x] remote URIs, UNC paths, project-root widening, and out-of-root paths fail.
- [x] output fields default to deny.
- [x] restricted variable values require separate permission.
- [x] parser complexity, aliases, merge keys, scalar types, and non-finite numbers are bounded or rejected.
- [x] validation errors hide values and arbitrary mapping keys.
- [x] the committed JSON Schema matches the Pydantic model.

## Synthetic input

- [x] deterministic seed is recorded.
- [x] generator produces source-specific corruption.
- [x] truth is held separately from linkage inputs.
- [x] no truth field is exposed to linkage model records.
- [x] cases include missingness, duplicates, no-match records, competing candidates, and assignment conflicts.
- [x] row-bearing representations hide values.
- [x] local synthetic writes translate filesystem failures without exposing paths.

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

- [ ] entity and household groups do not cross partitions.
- [ ] hard negatives use eligible labels only.
- [ ] candidate recall@K is reported.
- [ ] sensitivity and PPV are reported.
- [ ] false-link and missed-link rates are reported.
- [ ] precision–recall curve is generated.
- [ ] Brier score and reliability diagnostics are generated.
- [ ] ranking and assignment accuracy are reported.
- [ ] missingness-pattern metrics are reported.
- [ ] candidate-set-size metrics are reported.
- [ ] synthetic corruption regression metrics pass.

## Privacy

- [x] sentinel participant-like values never occur in configuration or CLI errors.
- [x] arbitrary identifier-like mapping keys are removed from displayed validation locations.
- [x] typed logging rejects unapproved record, identifier, and candidate-pair fields.
- [x] safe log construction hides rejected keys and values.
- [x] unrestricted manifests contain no row values, original identifiers, paths, or configuration payload.
- [ ] backend errors are sanitized after the DuckDB/model boundary exists.
- [x] CI tests generate synthetic inputs at runtime rather than reading committed row data.
- [x] CI uploads no row-level artifacts.

## Acceptance statement

Passing M1 establishes the safe configuration and synthetic-test foundation only. Passing the future full checklist establishes a working synthetic software slice only. Neither is evidence that linkage models are validated for real data.
