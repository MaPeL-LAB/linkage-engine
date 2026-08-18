# Roadmap

## Completed foundations

- **M0:** research, repository architecture, ADRs, governance, bibliography, package shell, and publication guard;
- **M1:** strict configuration compilation, path and logging controls, manifests, generated schema, and deterministic synthetic generation;
- **M2A:** local DuckDB data plane and bounded candidate retrieval;
- **M2B:** configuration-driven local ingestion and canonical preprocessing;
- **M2C:** comparison-feature construction and deterministic-anchor evidence;
- **M2D:** package-owned Fellegi–Sunter evidence baseline and Splink settings compiler;
- **M2E:** verified-label provenance, protected partitions, hard-negative selection, and XGBoost pair classifier.

## Current acceptance target: complete M2 synthetic MVP

The complete two-source `link_only`, `one_to_one` synthetic slice adds:

- validation-only Fellegi–Sunter versus XGBoost champion selection;
- calibration on a separate protected calibration partition;
- sigmoid and isotonic calibrators with native JSON manifests and integrity checks;
- XGBoost candidate ranking with ranking-only authority;
- OR-Tools one-to-one assignment with a private explicit no-match edge;
- an explicit decision layer producing `confirmed`, `review_required`, `unresolved`, or `no_match`;
- a restricted local review queue;
- candidate, pair, calibration, ranking, assignment, decision, and stratified evaluation;
- functional synthetic workflow CLI commands and deterministic orchestration;
- local bootstrap, environment-doctor, project-initialisation, and operational-validation handoff documentation.

M2 is complete only after the exact review head and the resulting `main` commit pass the Python 3.12 repository, typing, test, build, distribution, privacy, and synthetic end-to-end gates.

## Next: M3 adjudication and label lifecycle

- append-only adjudication decision import;
- superseding corrections and protocol/version provenance;
- double-review and disagreement handling where configured;
- explicit label promotion eligibility;
- active-learning or uncertainty-based review ordering;
- locked-test protection during review and retraining.

## Later milestones

- **M4:** `dedupe_only`, `link_and_dedupe`, many-to-one, one-to-many, and unconstrained modes;
- **M5:** LightGBM challenger, broader champion–challenger policy, model cards, and optional stacking;
- **M6:** optional feature-based PyTorch matcher under a deterministic and privacy-tested envelope;
- **M7:** source-aware multi-source entity resolution, cannot-link constraints, and contradiction-safe graph decisions;
- **M8:** compatibility matrix, performance benchmarks, security review, API hardening, release checklist, licence decision, and separately approved publication.

Privacy-preserving record linkage remains a separate research stream requiring its own threat model. No roadmap milestone implies operational validation or package publication without explicit approval.
