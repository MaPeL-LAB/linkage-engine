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

The repository name is **`linkage-engine`**. `MaPeL-LAB` identifies the developer and GitHub organisation; it is not part of the repository name.

## Status

This repository is a documentation-first, installable pre-alpha scaffold. It establishes the research basis, architectural boundaries, governance controls, configuration contract, verification expectations, and staged implementation plan. It does **not** yet implement or validate a production linkage model.

## Intended use

The package is designed to support, without study-specific assumptions:

- study-to-population-registry linkage;
- clinic-to-HDSS linkage;
- study-to-study linkage;
- registry-to-clinical-system linkage;
- multi-source entity resolution;
- within-dataset deduplication.

The implementation must never hard-code source dataset column names. Project configuration maps source columns to canonical variables and defines linkage mode, assignment constraints, transformations, blocking, comparisons, models, calibration, decision rules, validation, and restricted outputs.

## Non-negotiable privacy boundary

Only synthetic record-level data may appear in this repository, its documentation, examples, tests, notebooks, issues, pull requests, or continuous integration.

Real participant or operational data, identifiers, project configurations, adjudication records, secrets, model artefacts, candidate pairs, and linkage outputs must remain local under ignored directories. De-identified, hashed, tokenised, masked, sampled, or perturbed real records are **not** considered synthetic for repository purposes.

The package must never print or log record values, source identifiers, secrets, candidate pairs, training examples, or adjudication values. An existing crosswalk is not training truth unless independently verified under the label-provenance policy.

See:

- [`docs/governance/PRIVACY_THREAT_MODEL.md`](docs/governance/PRIVACY_THREAT_MODEL.md)
- [`docs/SYNTHETIC_DATA_POLICY.md`](docs/SYNTHETIC_DATA_POLICY.md)
- [`docs/governance/LABEL_PROVENANCE_POLICY.md`](docs/governance/LABEL_PROVENANCE_POLICY.md)

## Architectural position

> **Configuration is data, not executable code.**

YAML or JSON is validated with strict Pydantic models and compiled into an immutable execution plan. Configuration may select only package-owned allow-list registry entries. Raw SQL, arbitrary imports, dotted callable paths, `eval()`, `exec()`, and arbitrary code supplied through configuration are prohibited.

The planned pipeline separates configuration, local IO, normalisation, deterministic evidence, candidate retrieval, comparison features, pair scoring, ranking, calibration, model selection, assignment, decision policy, adjudication, and validation.

A ranking model may retrieve and order candidates, but it cannot decide identity. No model may silently merge records or create a master entity table.

## Relationship outcomes

- `confirmed`
- `review_required`
- `unresolved`
- `no_match`

`no_match` and `unresolved` are intentionally distinct.

## Planned command line

```text
mapel-linkage validate-config --config CONFIG
mapel-linkage generate-candidates --config CONFIG
mapel-linkage train --config CONFIG
mapel-linkage predict --config CONFIG
mapel-linkage assign --config CONFIG
mapel-linkage evaluate --config CONFIG
mapel-linkage run --config CONFIG
```

These are target interfaces. The current scaffold exposes package status and an explicit pre-alpha response for unimplemented commands.

## Documentation

The complete index is [`docs/README.md`](docs/README.md). Research claims use keys from [`docs/references/references.bib`](docs/references/references.bib).

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy src tests
pytest
python scripts/verify_repository.py
python -m build
```

Install the planned scientific core with `python -m pip install -e ".[core]"` after dependency compatibility is reviewed.

## Validation warning

> **Synthetic testing establishes software behaviour only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.**

## Publication and licence

The distribution is marked `Private :: Do Not Upload`. Publishing, public release, repository visibility changes, and licence selection require explicit MaPeL-LAB approval. See [`docs/governance/LICENSING_DECISION.md`](docs/governance/LICENSING_DECISION.md).
