# Contributing

Read `AGENTS.md`, the architecture documents, privacy threat model, label-provenance policy, and validation plan before contributing.

Use synthetic record-level data only. Do not paste real identifiers, candidate pairs, configurations, outputs, or secrets into GitHub. Add an ADR for changes to public contracts, trust boundaries, model authority, or irreversible dependency choices.

Run:

```bash
ruff format --check .
ruff check .
mypy src tests
pytest
python scripts/verify_repository.py
python -m build
```

Research claims should cite keys from `docs/references/references.bib`, such as `[@fellegi1969theory]`.
