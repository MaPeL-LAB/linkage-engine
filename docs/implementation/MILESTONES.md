# Staged Milestones

The status column distinguishes bounded source-code components from complete configured
workflows. See [`../CAPABILITY_MATRIX.md`](../CAPABILITY_MATRIX.md) for the machine-generated
component-level view.

| Milestone | Status | Scope | Remaining exit gate |
|---|---|---|---|
| M0 — research and repository baseline | Complete | documentation, ADRs, governance, bibliography, package shell, privacy controls, dependency plan | maintained through repository verification |
| M1 — safe configuration foundation | Merged and integrated | Pydantic schema, safe loader, compiler, registries, path policy, safe errors/logging, manifests, synthetic generator | maintained through malicious-config, privacy, schema, Python 3.12, and distribution checks |
| M2 — complete two-source synthetic slice | Merged and integrated | preparation, anchors, candidates, comparisons, Fellegi-Sunter, XGBoost classifier/ranker, champion selection, sigmoid/isotonic/Beta calibration, one-to-one no-match assignment, four-status decisions, review export, evaluation, CLI orchestration | operational validation remains separate and local |
| M3 — adjudication and label lifecycle | Components implemented; workflow integration pending | review import, disagreement handling, promotion eligibility, verified label batches, active-learning ordering | explicit CLI/artifact lifecycle, append-only persistence, protocol roles, retraining gate, and end-to-end tests |
| M4 — extended linkage modes | Components implemented; workflow integration pending | deduplication, many-to-one, one-to-many, unconstrained assignment; combined mode primitives | complete configured `dedupe_only` and `link_and_dedupe` runners plus mode-specific review/evaluation |
| M5 — broader champion-challenger and ensemble | Components implemented; workflow integration pending | LightGBM classifier/ranker, Beta calibration, stacking meta-learner | portfolio configuration, out-of-fold stacking evidence, model cards, selection and promotion policy |
| M6 — optional neural matcher | Component implemented; workflow integration pending | feature-based PyTorch matcher and controlled artifacts | full bounded configuration, all-model CI, calibration, portfolio integration, reproducibility reporting |
| M7 — multi-source resolution | Components implemented; workflow integration pending | evidence graph, correlation and agglomerative solvers, cannot-link constraints, crosswalk export, BCubed and cluster diagnostics | N-source configuration, approved pairwise-evidence ingestion, cluster review/decisions, end-to-end tests |
| I1 — audit and integration | In progress | capability registry, all-model CI, stage boundaries, train/approve/infer separation, immutable pipeline recipes, M3–M7 orchestration | each component is truthfully classified and each claimed workflow has a synthetic end-to-end test |
| I2 — Linkage Strategy Advisor | Planned | privacy-safe task profiles, benchmark registry, eligibility, Pareto shortlist, coverage, uncertainty, abstention | advisory-only recommendations evaluated on held-out scenario families; no identity or threshold authority |
| M8 — release hardening | Not started | compatibility matrix, scale benchmarks, security review, API stability, migration policy, release checklist | publication, licence, visibility, and operational approval remain separately authorised |

Passing any synthetic exit gate establishes software behaviour only. It does not establish
linkage validity, fairness, calibration, sensitivity, positive predictive value, false-link
rates, missed-link rates, or operational fitness on a real population or system.
