# Changelog

## [Unreleased]

### Added — plural model configuration and stage provenance

- Backward-compatible plural boosted, ranking, neural, and stacking model declarations with a versioned bounded portfolio selection.
- Validation of model IDs, stacking base-model availability, enabled portfolio members, challenger limits, and supervised-label eligibility.
- Immutable stage-artifact references and ordered lineage ledgers with restricted row-level enforcement.
- Aggregate-only out-of-fold prediction manifests that prohibit test, calibration, and decision partition use.
- Synthetic-only configuration, lineage, privacy, and provenance tests.


### Added — capability audit and integration boundary

- Package-owned capability registry separating component implementation, workflow
  integration, runtime verification, and operational validation.
- Generated `docs/CAPABILITY_MATRIX.md` with CI parity tests.
- `mapel-linkage status --details` and `mapel-linkage status --json`.
- ADR-0004 defining train/approve/infer separation, immutable pipeline recipes,
  shadow-challenger authority, M3–M7 integration, and the advisory-only Linkage Strategy
  Advisor.
- Immutable `PipelineRecipeArtifact` approval contract separating development, shadow, and
  inference execution authority.
- Dedicated all-model CI installing LightGBM and CPU PyTorch, executing the complete suite,
  and failing when any test is skipped.
- Exact Python 3.12 CI pins for LightGBM and PyTorch.
- Corrected roadmap, milestones, limitations, documentation index, README, and initial
  vertical-slice acceptance status.

### Reconciled — implemented components beyond M2

- M3 adjudication review import, disagreement handling, promotion eligibility, verified
  label-batch construction, and active-learning review ordering.
- M4 many-to-one, one-to-many, unconstrained assignment, single-source deduplication, and
  combined-mode primitives.
- M5 LightGBM pair-classifier and ranker challengers, Beta calibration, and stacking
  meta-learner.
- M6 feature-based PyTorch pair matcher with controlled artifact handling.
- M7 source-aware multi-source graph resolution, correlation clustering, constrained
  agglomerative clustering, connected-components baseline, cannot-link enforcement,
  crosswalk export, BCubed metrics, purity, pairwise metrics, and violation diagnostics.

These are now reported as component implementations unless and until a complete configured
workflow reaches them.

### Added — complete M2 synthetic vertical slice

- Validation-only Fellegi-Sunter versus XGBoost champion selection.
- Independent monotone sigmoid, isotonic, and Beta probability calibration with native JSON
  manifests and integrity checks.
- XGBoost candidate-ranking model with ranking-only authority, top-K outputs, and native
  JSON artifact verification.
- Deterministic one-to-one assignment with OR-Tools minimum-cost flow, a SciPy reference
  solver, and a private explicit no-match edge.
- Explicit `confirmed`, `review_required`, `unresolved`, and `no_match`
  relationship-decision policy.
- Restricted allow-listed review queue with aggregate-only unrestricted manifest.
- Candidate, pair, calibration, ranking, assignment, decision, threshold, and stratified
  synthetic evaluation.
- Entity-household connected-component splitting across training, validation, calibration,
  decision, and test partitions.
- Functional synthetic workflow CLI commands, deterministic orchestration, environment
  doctor, local workspace initialiser, and bootstrap scripts.
- Complete synthetic end-to-end test, conservative versioned mechanical regression guard,
  local deployment guide, operational validation runbook, and handoff checklist.

### Added — M2E verified-label XGBoost challenger

- Verified synthetic, adjudicated, and gold-standard label-source contracts.
- Pair, entity, and household partition-disjointness guards.
- Immutable numeric training/scoring matrices over M2C comparison features.
- Deterministic bounded hard-negative selection using verified nonmatches only.
- Native JSON XGBoost pair-classifier artifacts with aggregate manifests and digests.
- Evidence-only uncalibrated score tables and diagnostic validation reports.
- Expanded statistical package-version provenance in run manifests.
- Synthetic-only label, training, artifact, metric, determinism, and privacy tests.

### Added — M2D Fellegi-Sunter evidence baseline

- Deterministic bounded cross-source sampling for `u` estimation.
- Smoothed comparison-level `u` probabilities and aggregate-vector EM for `m`.
- Immutable aggregate model artifacts and canonical parameter digests.
- Local DuckDB Fellegi-Sunter evidence scores with explicit uncalibrated status.
- Package-owned Splink 4 settings compilation over canonical internal columns.
- Synthetic-only statistical, adapter, budget, and privacy tests.

### Added — M2C comparison features and anchor evidence

- Package-owned DuckDB comparison-feature construction over bounded candidate pairs.
- Exact, categorical, Jaro-Winkler, normalised Levenshtein/Damerau, q-gram, date, and
  numeric metrics.
- Configured comparison-level indices plus left, right, both, and any-missing indicators.
- Deterministic exact, prefix, date-window, conjunction, and disjunction anchor evidence.
- Per-anchor left/right uniqueness counts and evidence-only/training-ineligible flags.
- Pre-materialisation anchor-pair budgets and value-safe M2C errors.
- Synthetic-only feature, anchor, missingness, authority-boundary, and privacy tests.

### Added — M2B configured ingestion and preprocessing

- Local configuration-driven readers for Parquet, CSV, TSV, and newline-delimited JSON.
- Canonical variable mapping independent of source column names.
- Deterministic SHA-256 surrogate record keys with duplicate-ID rejection.
- Allow-listed Unicode, text, date, numeric, and Boolean normalisation.
- Explicit per-variable missingness indicators in canonical DuckDB tables.
- Opaque prepared-dataset and prepared-catalog structural contracts.
- Value-safe ingestion, normalisation, and data-preparation errors.
- Synthetic-only preprocessing, configuration-renaming, determinism, and privacy tests.

### Added — M2A local data plane and candidate generation

- Opaque row-bearing `TableRef` contract.
- Parameterised local DuckDB table creation.
- Typed exact, prefix, conjunction, and disjunction blocking predicates.
- Bounded, deduplicated, multi-rule candidate retrieval with aggregate diagnostics.

### Added — M1 safe configuration foundation

- Strict immutable Pydantic configuration schema and cross-field validation.
- Safe bounded YAML and JSON loaders with value-safe error translation.
- Typed blocking/comparison DSL and immutable operation registries.
- Immutable execution-plan compiler with configuration and registry digests.
- Host-enveloped local path policy and deny-by-default output controls.
- Typed aggregate-only logging and privacy-safe run manifests.
- Deterministic synthetic source/truth generator with corruption and edge cases.
- Generated `schemas/linkage-config.schema.json` and schema-parity tests.
- Functional `validate-config` and `emit-config-schema` CLI commands.
- Configuration, governance, synthetic, CLI, and repository-boundary test coverage.

### Added — M0 baseline

- Documentation-first repository scaffold.
- Canonical naming and synthetic-only privacy boundary.
- Research synthesis, method/software landscapes, ADRs, governance policies, roadmap, and
  BibTeX bibliography.
- Minimal installable package and safe pre-alpha CLI shell.
- Synthetic-only CI and repository verification checks.
- Repository manifest generation and integrity verification.

### Not yet workflow integrated or operationally validated

- Genuine artifact-to-artifact CLI stage boundaries.
- Model-portfolio configuration and protected ensemble orchestration.
- New-data inference from an immutable approved `PipelineRecipeArtifact` without retraining
  or model reselection.
- Complete configured M3 adjudication and authorised retraining lifecycle.
- Complete `dedupe_only`, `link_and_dedupe`, extended-assignment, and N-source workflows.
- Shadow challengers, drift monitoring, and the Linkage Strategy Advisor.
- Full native Splink model lifecycle and term-frequency-adjusted reference scoring.
- Operational performance, calibration, fairness, and real-data validation.
- Licence selection, release approval, and package publication.
