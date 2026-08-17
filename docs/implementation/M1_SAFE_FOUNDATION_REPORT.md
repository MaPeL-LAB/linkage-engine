# M1 Safe Foundation Implementation Report

- **Milestone:** M1 — safe configuration foundation
- **Package version:** `0.1.0.dev1`
- **Status:** Implementation candidate; local functional verification complete; remote Python 3.12 CI required before merge
- **Data used:** Generated synthetic data only

## Scope delivered

M1 establishes the trust boundary before any linkage model receives row-level data.

### Configuration

- Frozen Pydantic models with immutable mapping fields, `extra="forbid"`, validated defaults, and hidden input values.
- YAML and JSON loaders with UTF-8, file-size, duplicate-key, merge-key, alias, structural-depth, node-count, finite-number, top-level mapping, and JSON-compatible scalar controls.
- Cross-field validation for dataset counts and roles, identifiers, source mappings, transformation and comparator compatibility, comparison-level semantics, blocking predicates, label coverage and eligibility, enabled calibration sources, assignment constraints, non-disableable safeguards, decision thresholds, non-empty split fractions, and output permissions.
- Recursive typed predicates for `exact`, `prefix_equal`, `date_window`, `all`, and `any`.
- Typed transformation, comparison, model, calibration, assignment, and decision configuration.
- Committed JSON Schema generated directly from the Pydantic model.

### Compilation and registries

- Canonical SHA-256 configuration digest.
- Immutable operation registries and registry digest.
- No import-path, reflection, plugin, SQL, `eval`, or `exec` resolution from configuration.
- Immutable `ExecutionPlan` with local paths excluded from its representation.

### Privacy and governance

- Configured roots must fit inside a separately controlled host envelope.
- Default host input roots are `data/` and `private/`; default output roots are `private/` and `artifacts/`.
- Remote URI schemes, UNC paths, malformed paths, home expansion, project-root widening, and out-of-root paths are rejected.
- Pydantic errors are translated into stable safe codes. Submitted values and arbitrary mapping keys are excluded from displayed locations.
- Logging accepts only typed structural and aggregate fields.
- Run manifests contain versions, digests, seed, platform, status, and aggregate counts; they contain no row values, IDs, paths, or configuration payload.
- Output fields are deny-by-default. Variable values require separate `restricted_output` permission.

### Synthetic generation

- Deterministic generation from a recorded seed.
- Generic values with no study assumptions.
- Separate source A, source B, truth, and provenance structures.
- Source-specific typographical and date corruption.
- Missingness, duplicate records, source-only entities, competing candidates, and assignment conflicts.
- Row-bearing classes hide values from `repr()`.
- Optional local JSONL writer keeps truth in a distinct file, uses atomic file replacement, emits no logs, and translates filesystem errors without paths.

### CLI

`validate-config` now performs real validation and compilation. `emit-config-schema` writes the normative JSON Schema. Other linkage commands remain explicit pre-alpha placeholders.

## Files added

```text
schemas/linkage-config.schema.json
schemas/README.md
scripts/generate_config_schema.py
scripts/generate_repository_manifest.py
src/mapel_linkage/configuration/
src/mapel_linkage/governance/
src/mapel_linkage/synthetic/
tests/configuration/
tests/governance/
tests/synthetic/
```

## Verification completed locally

```text
python -m compileall src tests scripts                     passed
pytest                                                      74 passed
python scripts/verify_repository.py                         passed
mapel-linkage status                                        passed through module entry point
mapel-linkage validate-config on synthetic example          passed
committed JSON Schema equals generated Pydantic schema      passed
repository SHA-256 manifest generation and verification       passed
```

The local execution environment provides Python 3.13 rather than the repository's supported Python 3.12 and cannot download Ruff, mypy, Hatchling, or Build because external package access is unavailable. The implementation must therefore pass the repository's Python 3.12 GitHub Actions job before merge. That job is responsible for Ruff formatting/linting, strict mypy, pytest, wheel/source build, and restricted-distribution inspection.

## Acceptance mapping

| M1 gate | Evidence |
|---|---|
| unknown and duplicate keys fail | configuration injection and duplicate-key tests |
| raw SQL/callable/module keys fail | parameterized unsafe-field tests |
| parser complexity is bounded | alias, merge, depth, node, non-finite number, and scalar tests |
| operations use allow-list registries | immutable registry and unknown-key tests |
| paths remain inside approved roots | remote, out-of-root, host-envelope, and root-widening tests |
| output defaults to deny | empty allow-list test |
| unverified crosswalk is not truth | supervised-model eligibility and non-disableable policy tests |
| validation values remain hidden | invalid numeric, arbitrary map key, CLI sentinel tests |
| logs reject row-like fields | safe logging and safe builder tests |
| manifests exclude values and paths | manifest tests |
| synthetic generation is deterministic | repeatability test |
| truth is separate | model-input/truth separation test |
| required edge cases exist | missingness, duplicate, no-match, competitor, and corruption tests |

## Limitations

- No dataset is opened or inspected during M1 validation.
- Paths are checked structurally but input existence and file schema validation are M2 IO concerns.
- DuckDB security settings are not implemented until the DuckDB connection layer is introduced.
- No candidate pair, comparison feature, model, calibrator, assignment, relationship decision, or adjudication event is produced.
- Complete bitwise reproducibility is not claimed outside a recorded execution envelope.
- Synthetic passing results do not establish real-world linkage accuracy.

## Next bounded milestone

M2 should begin with the smallest two-source `link_only` vertical slice: canonical IO mapping, safe DuckDB setup, whitelisted normalisation, deterministic anchors, bounded candidate generation, comparison features, and the Splink Fellegi–Sunter baseline. Supervised models, calibration, assignment, and final decisions should be added only after candidate-retrieval parity and privacy tests pass.
