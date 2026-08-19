# Linkage Engine

`mapel-linkage-engine` is a pre-alpha Python package for configurable probabilistic record
linkage, entity resolution, and within-dataset deduplication.

| Item | Canonical value |
|---|---|
| Developer / GitHub organisation | `MaPeL-LAB` |
| Repository | `linkage-engine` |
| Python distribution | `mapel-linkage-engine` |
| Import package | `mapel_linkage` |
| Command-line interface | `mapel-linkage` |
| Initial Python runtime | Python 3.12 |
| Current package version | `0.2.0.dev1` |

The repository name is **`linkage-engine`**. `MaPeL-LAB` identifies the developer and
GitHub organisation; it is not part of the repository name.

## Current platform status

The repository now distinguishes:

```text
specified
component implemented
workflow integrated
runtime verified
operationally validated
```

The normative machine-generated status is
[`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md).

### Complete integrated workflow

M0, M1, and the complete M2 generated-synthetic vertical slice are integrated:

```text
strict configuration
→ local DuckDB preparation
→ deterministic-anchor evidence
→ bounded candidate retrieval
→ comparison features
→ Fellegi-Sunter and XGBoost pair scoring
→ validation-only champion selection
→ sigmoid, isotonic, or Beta calibration
→ XGBoost candidate ranking
→ one-to-one assignment with explicit no-match
→ confirmed / review_required / unresolved / no_match
→ restricted review export and aggregate evaluation
```

This is a two-source `link_only`, `one_to_one` software-validation workflow.

### Implemented components awaiting general orchestration

Substantive components also exist for:

- adjudication import, disagreement resolution, label-promotion eligibility, and
  active-learning review ordering;
- many-to-one, one-to-many, and unconstrained assignment;
- single-source deduplication and combined-mode primitives;
- LightGBM pair classification and candidate ranking;
- a stacking meta-learner;
- a feature-based PyTorch pair matcher;
- source-aware multi-source entity resolution;
- correlation clustering, constrained agglomerative clustering, and cannot-link
  enforcement;
- BCubed, purity, pairwise, and constraint-violation cluster metrics.

These components are not yet all reachable through a complete configuration-driven CLI
workflow. Code presence is not reported as platform integration.

### Current integration track

I1 will add:

- genuine artifact-to-artifact stage boundaries;
- model-portfolio configuration;
- train, approve, and new-data inference separation;
- immutable approved pipeline-recipe artifacts;
- shadow challengers with no decision authority;
- complete M3 through M7 orchestration and synthetic end-to-end tests.

I2 will introduce the advisory-only Linkage Strategy Advisor, using privacy-safe task
profiles and benchmark evidence to recommend a small pipeline shortlist with uncertainty,
coverage checks, and abstention.

See
[`docs/architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md`](docs/architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md).

## Intended use

The package is designed to support, without study-specific assumptions:

- study-to-population-registry linkage;
- clinic-to-HDSS linkage;
- study-to-study linkage;
- registry-to-clinical-system linkage;
- multi-source entity resolution;
- within-dataset deduplication.

Source column names are accepted only through validated dataset and variable mappings. They
are not embedded in model, comparison, assignment, calibration, decision, or orchestration
logic.

## Non-negotiable privacy boundary

Only synthetic record-level data may appear in this repository, its documentation, examples,
tests, notebooks, issues, pull requests, or continuous integration.

Real participant or operational data, identifiers, completed project configurations,
adjudication records, secrets, model artefacts, candidate pairs, and linkage outputs must
remain local under ignored directories. De-identified, hashed, tokenised, masked, sampled,
or perturbed real records are **not** considered synthetic for repository purposes.

The package must never print or log record values, source identifiers, secrets, candidate
pairs, training examples, or adjudication values. An existing crosswalk is not training truth
unless independently verified under the label-provenance policy.

Repository visibility is not a data-security control. The synthetic-only boundary applies
whether GitHub visibility is public or private.

See:

- [`docs/governance/PRIVACY_THREAT_MODEL.md`](docs/governance/PRIVACY_THREAT_MODEL.md)
- [`docs/SYNTHETIC_DATA_POLICY.md`](docs/SYNTHETIC_DATA_POLICY.md)
- [`docs/governance/LABEL_PROVENANCE_POLICY.md`](docs/governance/LABEL_PROVENANCE_POLICY.md)

## Configuration is data, not executable code

The M1 trust boundary includes:

- strict immutable Pydantic models with unknown fields forbidden;
- bounded YAML and JSON loading;
- value-safe validation translation;
- a package-owned typed blocking and comparison DSL;
- immutable allow-list registries;
- canonical configuration and registry digests;
- project and host path envelopes;
- deny-by-default restricted outputs;
- aggregate-only logging and manifests;
- deterministic generated-synthetic source and truth separation;
- a generated machine-readable JSON Schema.

No configuration may provide raw SQL, a shell command, a module path, a Python callable,
`eval()`, `exec()`, or arbitrary executable content.

## Authority separation

```text
candidate retrieval ≠ identity decision
pair-model score ≠ calibrated probability
candidate rank ≠ confirmed relationship
assignment ≠ relationship status
cluster membership ≠ silent master-record creation
recommendation ≠ model promotion
```

Candidate retrieval does not decide identity. Pair and ranking models remain evidence-only.
Assignment performs constrained selection but does not classify relationships. Only the
explicit decision policy emits `confirmed`, `review_required`, `unresolved`, or `no_match`.
No component has silent merge authority.

## Plural model configuration and stage provenance

Project configuration can declare multiple boosted, ranking, neural, and stacking candidates while retaining the existing singular fields for compatibility. Immutable stage-artifact and out-of-fold manifests provide digest-linked provenance without exposing rows, identifiers, candidate pairs, or local paths. See [`docs/implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md`](docs/implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md).

## Stage-1 Linkage Strategy Advisor

The advisory-only Stage-1 workflow can build a privacy-safe preflight task profile, apply hard
lifecycle and runtime eligibility rules, retain the mandatory Fellegi-Sunter baseline, construct
a structural Pareto shortlist, explain every applied rule, and abstain from empirical ranking
while the benchmark registry is empty.

```text
mapel-linkage profile-job --config CONFIG --project-root ROOT
mapel-linkage recommend-pipeline --config CONFIG --project-root ROOT
```

A recommendation is not a `PipelineRecipeArtifact`, cannot approve a model, cannot use the locked
test partition, and has no identity, assignment, threshold, or merge authority. See
[`docs/architecture/ADR-0005-LINKAGE-STRATEGY-ADVISOR.md`](
docs/architecture/ADR-0005-LINKAGE-STRATEGY-ADVISOR.md).

## Command line

Environment, configuration, and capability commands:

```text
mapel-linkage status
mapel-linkage status --details
mapel-linkage status --json
mapel-linkage doctor --project-root ROOT
mapel-linkage init-local-project --directory ROOT
mapel-linkage validate-config --config CONFIG --project-root ROOT
mapel-linkage emit-config-schema --output OUTPUT
```

Generated-synthetic workflow commands:

```text
mapel-linkage generate-candidates --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage train --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage predict --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage assign --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage evaluate --config CONFIG --project-root ROOT --synthetic-demo
mapel-linkage run --config CONFIG --project-root ROOT --synthetic-demo
```

The current stage commands execute the required complete synthetic workflow and print the
requested stage summary. Genuine reusable stage artifacts are part of I1. The repository
build refuses row-level execution without `--synthetic-demo`.

## Model portfolio

| Role | Current implementation | Integrated workflow |
|---|---|---|
| Statistical baseline | package-owned Fellegi-Sunter; Splink settings/parity adapter | yes, package-owned path |
| Pair classifier | XGBoost | yes |
| Pair-classifier challenger | LightGBM | component only |
| Candidate ranker | XGBoost | yes |
| Ranking challenger | LightGBM | component only |
| Ensemble | stacking meta-learner | component only |
| Neural challenger | feature-based PyTorch MLP | component only |
| Calibration | sigmoid, isotonic, Beta | yes |
| Assignment | one-to-one OR-Tools plus SciPy reference | yes |
| Extended assignment | many-to-one, one-to-many, unconstrained | component only |
| Multi-source | source-aware graph resolver and constrained clustering | component only |

The default core CI remains fast. Dedicated all-model CI installs pinned LightGBM and CPU
PyTorch, runs the complete test suite, and fails when any test is skipped.

## Synthetic generator

`mapel_linkage.synthetic.generate_synthetic_bundle()` creates deterministic generic source
tables with:

- source-specific corruption;
- missing values;
- duplicates and assignment conflicts;
- source-only no-match records;
- competing candidates;
- truth held in a separate test-only structure.

Generated rows and truth records use value-hiding representations and are never stored in
the repository.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/ci-py312.txt -e ".[core,dev]"
python scripts/generate_config_schema.py
python scripts/generate_capability_matrix.py
python scripts/generate_repository_manifest.py
python scripts/verify_repository.py
ruff format --check .
ruff check .
mypy src tests
pre-commit run --all-files
pytest
python -m build
python scripts/verify_repository.py --distribution dist
```

For optional local model verification:

```bash
python -m pip install -c constraints/ci-py312.txt -e ".[core,dev,lightgbm,neural]"
pytest
```

## Documentation

The documentation index is [`docs/README.md`](docs/README.md). Research claims use keys from
[`docs/references/references.bib`](docs/references/references.bib).

## Validation warning

> **Synthetic testing establishes software behaviour only. It does not validate linkage
> accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates,
> missed-link rates, or operational fitness on real populations or systems.**

## Publication and licence

The distribution is marked `Private :: Do Not Upload`. Publishing a package, selecting a
licence, changing repository visibility, or creating a release requires explicit MaPeL-LAB
approval. See
[`docs/governance/LICENSING_DECISION.md`](docs/governance/LICENSING_DECISION.md).
