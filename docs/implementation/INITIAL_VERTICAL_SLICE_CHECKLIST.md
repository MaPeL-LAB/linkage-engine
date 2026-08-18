# Initial Synthetic Vertical Slice Checklist

The checked items below are implemented in the `0.2.0.dev0` complete-M2 candidate. Items marked **acceptance pending** require the exact review head and post-merge Python 3.12 workflows to pass before M2 is declared accepted.

## Repository foundation

- [x] Python 3.12 package uses `src/mapel_linkage`.
- [x] CLI entry point is `mapel-linkage`.
- [x] `private/`, `data/`, and `artifacts/` are Git-ignored and excluded from distributions.
- [x] package publication remains blocked by `Private :: Do Not Upload`.
- [x] wheel and source-distribution inspection is part of CI.
- [ ] **Acceptance pending:** Ruff, strict mypy, pre-commit, all tests, repository verification, and builds pass on the exact review and merged heads.

## Configuration boundary

- [x] YAML and JSON load through strict immutable Pydantic models.
- [x] unknown and duplicate keys fail.
- [x] raw SQL fields and arbitrary executable configuration fail.
- [x] Python/module/callable/import-path fields fail.
- [x] configurable operations resolve only through immutable allow-list registries.
- [x] dataset-specific columns occur only in project configuration and IO mapping contracts.
- [x] configured paths resolve inside project and host-approved roots.
- [x] remote URIs, UNC paths, project-root widening, and out-of-root paths fail.
- [x] output fields default to deny; restricted variable values require separate permission.
- [x] parser complexity, aliases, merge keys, scalar types, and non-finite numbers are bounded or rejected.
- [x] validation errors hide values and arbitrary mapping keys.
- [x] the generated JSON Schema is tested for parity with the Pydantic model.

## Synthetic input and protected truth

- [x] deterministic generator version and seed are recorded.
- [x] source-specific corruption, missingness, duplicates, no-match records, competitors, and assignment conflicts are generated.
- [x] truth is held separately from linkage inputs.
- [x] no truth field is exposed to preprocessing, retrieval, comparison, or model inputs.
- [x] row-bearing representations hide values.
- [x] local synthetic writes translate filesystem failures without exposing paths.
- [x] entity–household connected components are assigned to protected training, validation, calibration, decision, and test partitions.
- [x] pair, entity, and household overlap across protected label partitions fails.

## Linkage pipeline

- [x] preprocessing operates on canonical variables.
- [x] deterministic anchors are evidence-only and ineligible as training truth by default.
- [x] candidate generation has a hard pair budget and records retrieval-rule provenance.
- [x] comparison features contain configured levels, numeric evidence, and explicit missingness indicators without copying source values.
- [x] package-owned Fellegi–Sunter scoring runs over comparison levels.
- [x] canonical configuration compiles to a Splink 4 settings plan; runtime blocking parity is an acceptance test for the supported subset.
- [x] the XGBoost pair classifier trains only on eligible verified training labels.
- [x] validation-only champion selection excludes calibration, decision, and test partitions.
- [x] sigmoid and isotonic calibration fit only on a protected independent calibration partition.
- [x] the XGBoost candidate ranker emits score/rank/top-K evidence only.
- [x] global one-to-one assignment includes a private explicit no-match edge.
- [x] the decision policy emits exactly one of `confirmed`, `review_required`, `unresolved`, or `no_match`.
- [x] incomplete/truncated search and invalid calibration force `unresolved`, not `no_match`.
- [x] no stage exposes merge or master-record authority.
- [x] uncertain cases produce a restricted allow-listed review queue.

## Validation

- [x] hard negatives use eligible verified nonmatches only; unknown pairs remain unknown.
- [x] candidate recall@K, zero-candidate rate, candidate-set distribution, Cartesian reduction, and rule contribution are reported.
- [x] sensitivity, PPV, false-link rate, missed-link rate, average precision, ROC AUC, Brier score, and precision–recall points are reported.
- [x] reliability bins, calibration slope/intercept, expected calibration error, and maximum calibration error are reported.
- [x] ranking recall@K, top-1 rate, mean reciprocal rank, and true-match rank are reported.
- [x] assignment accuracy, no-match accuracy, change from independent top-1, and capacity violations are reported.
- [x] relationship-status counts and review burden are reported.
- [x] missingness-pattern and candidate-set-size stratified pair performance is reported.
- [x] configured synthetic decision-threshold evidence is isolated to the protected decision partition.
- [x] a versioned conservative synthetic regression guard detects catastrophic mechanical regressions.
- [ ] **Acceptance pending:** the complete Python 3.12 synthetic end-to-end run passes deterministically on the exact review and merged heads.

## Privacy and artifact integrity

- [x] synthetic sentinel values, original identifiers, candidate pairs, paths, and review values are absent from unrestricted logs, errors, representations, and manifests.
- [x] typed logging rejects unapproved row, identifier, and candidate-pair fields.
- [x] unrestricted manifests contain only aggregate metadata, versions, counts, and digests.
- [x] XGBoost pair, XGBoost ranking, and calibrator artifacts use native/package JSON rather than pickle or joblib.
- [x] model, ranker, and calibrator artifact tampering is detected before use.
- [x] restricted relationship and review exports obey the configured field allow-list.
- [x] CI generates synthetic row-level inputs at runtime and uploads no row-level data artifacts.
- [ ] **Acceptance pending:** built wheel and source distribution contain no restricted directories, row files, local configuration, databases, or model artifacts.

## Acceptance statement

Passing this checklist establishes a working two-source synthetic software slice only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.
