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
| Current package version | `0.2.0.dev3` |

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
→ native Splink 4.0.16 Fellegi-Sunter and XGBoost pair scoring
→ validation-only champion selection
→ sigmoid, isotonic, or Beta calibration
→ XGBoost candidate ranking
→ one-to-one assignment with explicit no-match
→ confirmed / review_required / unresolved / no_match
→ restricted review export and aggregate evaluation
```

This is a two-source `link_only`, `one_to_one` software-validation workflow.

### Configuration-driven model portfolio

I1B adds a second bounded path inside the same generated-synthetic, two-source
`link_only`, `one_to_one` authority boundary:

- compile the plural model configuration without implementation fallbacks;
- fit and reload the mandatory native Splink baseline;
- train configured XGBoost, LightGBM, PyTorch, stacking, and ranking candidates;
- create source-side entity/household-connected out-of-fold stacking evidence;
- select only on validation and fit calibration only on calibration;
- evaluate the frozen champion on locked test without selection or calibration access;
- persist and strictly reload the executable champion, calibrator, source-query ranker,
  and recipe-v1 binding;
- replay two disjoint synthetic decision-evidence subsets through the reloaded recipe.

Target-query rankers are trained and reported but are not silently reinterpreted by the
source-to-target assignment contract. All model and ranker outputs remain evidence or ordering
only; relationship status comes only from the decision policy and merge authority remains none.

Other M3–M7 workflows have the component/workflow states recorded in the generated capability
matrix. The complete CLI does not imply one general orchestrator for every capability.

### Configuration-driven linkage modes

I1C adds one separately allow-listed generated-synthetic command for exactly these combinations:

- `link_only` with `many_to_one`;
- `link_only` with `one_to_many`;
- `link_only` with `unconstrained`;
- `dedupe_only` with `unconstrained`; and
- `link_and_dedupe` with `one_to_one`.

Link-only decisions are made only from the protected decision partition. Same-source
candidate generation removes self-pairs and canonicalises symmetric pairs before feature
construction. `link_and_dedupe` fits and calibrates one strictly bound model over cross-source
and both same-source surfaces rather than transporting cross-source calibration. Dedupe-only
and link-and-dedupe outputs are aggregate assignment/cluster evidence: they emit no
relationship statuses and have no decision or merge authority. This I1C route is
synthetic-only, has operational validity `not_established`, and does not authorise arbitrary
mode combinations, multi-source dispatch, or real-data use.

The I2 advisory stack is integrated with coverage checks, uncertainty, fallback, abstention,
and explicit human approval boundaries. It cannot approve models, thresholds, assignments,
relationship statuses, or merges.

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

## Canonical synthetic lifecycle demonstration

The import-safe [`examples/e2e_linkage_lifecycle.py`](examples/e2e_linkage_lifecycle.py)
connects synthetic generation, preflight profiling, the advisory-only Stage-3 meta-ranker
(including fail-safe similarity fallback when the complete advisor grid is absent),
protected model-portfolio training and calibration, review ordering, human consensus and
partition-disjoint label promotion, immutable recipe IO, synthetic-only new-data inference,
and aggregate multi-source evaluation. It never establishes operational validity.

Run the focused lifecycle and tests with the canonical Python 3.12 environment:

```bash
scripts/run_e2e_lifecycle.sh
```

For a long verification run outside Codex, add `--full`. Inspect the exact commands without
writing files with `--dry-run`; use `--python PATH` or `--output-dir DIR` to override the safe
defaults.

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

## Active synthetic benchmark planning

Stage 4 computes aggregate coverage density over package-owned synthetic corruption axes and can
produce a bounded, snapshot-bound experiment plan when similarity evidence is out of distribution,
conformal intervals are wide, the meta-ranker abstains, or catalogue coverage is incomplete.

```text
mapel-linkage profile-job --config CONFIG --project-root ROOT > target-profile.json
mapel-linkage plan-benchmarks --registry-dir DIR --target-profile target-profile.json
```

The CLI plans only. Execution requires a separate plan-bound human approval contract, protects
prospectively held-out mechanisms, rejects stale registries and duplicate run IDs, and uses only the
package-owned deterministic synthetic generator. Plans and refitted meta-models remain
`advisory_only`; automatic model promotion and all decision, assignment, and merge authority remain
prohibited. Synthetic benchmark evidence does not establish operational validity.

### Advisor-scale synthetic corpus

The versioned `advisor_v2` design expands the stable seed-v1 catalogue from 10 families/19
instances with a separate 64-family/280-instance experimental programme. Its prospective family
roles are 40 meta-training, 8 conformal, 8 locked-evaluation, and 8 true-mechanism OOD families.
The historical seed-v1 transliteration family remains digest-stable but is excluded from true OOD
readiness because it is a typo/transposition proxy.

Quick aggregate planning is safe to run directly:

```text
mapel-linkage plan-advisor-corpus --shards 32 --replicates 5
scripts/run_advisor_corpus.sh --dry-run --shards 32 --replicates 5
```

The first execution-protocol-v1 run completed all 9,800 retained records on 2026-08-21. Its audit
exposed 688 Fellegi-Sunter score-materialisation failures alongside 3,512 successes and 5,600
expected ineligible records. That registry remains immutable diagnostic evidence; its former
family-level readiness result is superseded because excluding failed replicates would create
survivor bias.

Execution protocol v2 uses numerically stable Fellegi-Sunter scoring and requires all 1,400
scenario-replicate cells to contain successful evidence from each of the three required adapters.
The corrected run completed all 9,800 records with 4,200 required successes and 5,600 expected
ineligible records. Its aggregate readiness digest is
`4c91c1099c15f226ddded933a3fb5462e23f5ecf8c44914eac87944882d84e76`.

The first prospective advisor qualification returned `not_qualified`: both advisors had zero
locked-family regret but could not improve on the already-perfect fixed XGBoost-classifier
baseline; split-conformal coverage was 79.17 percent against the fixed 80 percent gate; and the
current task meta-features detected none of the eight true-mechanism OOD families. The exact
aggregate result and its consequences are recorded in
[`docs/implementation/I2_ADVISOR_EMPIRICAL_QUALIFICATION.md`](docs/implementation/I2_ADVISOR_EMPIRICAL_QUALIFICATION.md).
No threshold was changed after locked evaluation.
An exact complete registry is not sufficient to activate learned ranking: the Stage-3 advisor also
requires a canonical `qualified` artifact bound to that registry snapshot and policy. The current
artifact therefore forces similarity fallback.

The corpus driver remains the resumable outside-Codex route for rebuilding the heavy registry. A
human must choose a non-identifying approval reference and explicitly authorise the exact
digest-bound plan:

```bash
CORPUS_APPROVAL_REF="replace-with-approved-non-identifying-reference"
scripts/run_advisor_corpus.sh \
  --full \
  --approve-execution \
  --approval-reference "${CORPUS_APPROVAL_REF}"
```

The driver verifies the focused implementation first, then executes every deterministic shard
into the ignored project-relative execution-v2 registry; it never overwrites the diagnostic v1
registry. Reruns resume exact retained evidence; tamper, collision, path escape, symlink
traversal, or dependency/environment drift fails closed. Only the
Fellegi-Sunter reference, XGBoost classifier, and XGBoost ranker currently have truth-safe
success-capable benchmark adapters. Other portfolio entries retain stable ineligible evidence and
never receive placeholder metrics. Repository CI does not execute the heavy corpus, and its
scientific or operational validity remains `not_established`.

The bounded qualification itself is quick and can be run directly after explicit locked-family
approval:

```text
mapel-linkage qualify-advisor --project-root . \
  --registry-dir private/benchmark_registry/advisor_v2_execution_v2 \
  --output artifacts/advisor_qualification/advisor_v2_qualification.json \
  --approve-locked-evaluation --approval-reference NON_IDENTIFYING_REFERENCE
```

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
mapel-linkage run-model-portfolio --config configs/examples/synthetic_all_models.yaml \
  --project-root ROOT --synthetic-demo --entity-count 120 --k-folds 3
mapel-linkage run-linkage-mode --config CONFIG --project-root ROOT \
  --synthetic-demo --entity-count 120
```

The current stage commands execute the required complete synthetic workflow and print the
requested stage summary. `run-model-portfolio` executes the configured protected tournament,
strict artifact reload, locked-test evaluation, and disjoint recipe-bound replay and prints
aggregate metadata only. The repository build refuses row-level execution without
`--synthetic-demo`. For a robust long-run preflight and verification wrapper, run
`scripts/run_all_model_portfolio.sh --dry-run` first for I1B or
`scripts/run_i1c_linkage_modes.sh --dry-run` first for I1C.

## Model portfolio

| Role | Current implementation | Integrated workflow |
|---|---|---|
| Statistical baseline | native Splink 4.0.16 fit, canonical JSON reload, exact candidate parity, and scoring; package-owned reference matcher retained as deterministic oracle | yes |
| Pair classifier | XGBoost | yes |
| Pair-classifier challenger | LightGBM | yes; all-models CI |
| Candidate ranker | XGBoost | yes |
| Ranking challenger | LightGBM | yes; source-query replay, target-query reporting only |
| Ensemble | stacking meta-learner | yes |
| Neural challenger | feature-based PyTorch MLP | yes; all-models CI |
| Calibration | sigmoid, isotonic, Beta | yes |
| Assignment | one-to-one OR-Tools plus SciPy reference | yes |
| Extended assignment | many-to-one, one-to-many, unconstrained | exact generated-synthetic I1C combinations only |
| Deduplication | same-source canonical candidates and aggregate cluster evidence | `dedupe_only` + `unconstrained`, and two-source `link_and_dedupe` + `one_to_one`, synthetic only |
| Multi-source | source-aware graph resolver and constrained clustering | resolver workflow integrated; individual clustering/metric components remain separately classified |

The integrated synthetic lifecycle fits the native Splink model, serializes and strictly
reloads its value-hidden canonical artifact, and scores only the exact bounded pair set
authorized by package-owned candidate retrieval. Native scores are recomputed from typed
prepared-data replay when Splink wins; integrity-only score evidence cannot authorize generic
inference. Fellegi-Sunter outputs remain uncalibrated, evidence-only model posteriors; they
cannot emit relationship decisions or establish operational validity.

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
