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
| M4 — extended linkage modes | Bounded I1C synthetic workflow integrated | exactly `link_only` + many-to-one/one-to-many/unconstrained, `dedupe_only` + unconstrained, and `link_and_dedupe` + one-to-one | operational validation, mode-specific human review, arbitrary combinations, and real-data dispatch remain outside the repository route |
| M5 — broader champion-challenger and ensemble | Components implemented; workflow integration pending | LightGBM classifier/ranker, Beta calibration, stacking meta-learner, plural portfolio configuration, immutable stage provenance | general portfolio runner, model cards, selection, calibration, approval, and inference orchestration |
| M6 — optional neural matcher | Component implemented; workflow integration pending | feature-based PyTorch matcher and controlled artifacts | full bounded configuration, calibration, portfolio integration, reproducibility reporting |
| M7 — multi-source resolution | Components implemented; workflow integration pending | evidence graph, correlation and agglomerative solvers, cannot-link constraints, crosswalk export, BCubed and cluster diagnostics | N-source configuration, approved pairwise-evidence ingestion, cluster review/decisions, end-to-end tests |
| I1 — bounded orchestration | I1A–I1C integrated within declared boundaries | artifact stages, protected portfolio training, strict recipe replay, and five allow-listed synthetic mode combinations | no general M3–M7, multi-source, real-data, or operational orchestration is claimed |
| I2A — Stage-1 Strategy Advisor | Implemented and CLI integrated | staged aggregate profiles, evidence hierarchy, hard eligibility, structural Pareto shortlist, transparent explanations, abstention | maintain zero authority violations and no empirical performance claims without evidence |
| B1 — synthetic benchmark library | Corrected v2 evidence and v3 corpus complete; v3.1 governance remediation pending audit | stable seed-v1, immutable v2 evidence, completed 84-family/336-instance v3 source, observable mechanism profiles, causal supervised label budgets, whole-family shards, frozen-provenance validation, and a role-specific governance-only amendment | audit the distinct v3.1 governance bridge; qualification, promotion, operational validity, and a redundant second corpus remain outside this gate |
| I2B — nearest-family advisor | Workflow integrated; first empirical qualification failed closed | similarity retrieval, coverage, OOD detection, uncertainty, abstention, and family-held-out evaluation | a new prospective round must outperform the best fixed recipe and detect held-out mechanisms |
| I2C — meta-learning ranker | Workflow integrated; first empirical qualification failed closed | family-disjoint training, conformal intervals, learned shortlist ordering, locked evaluation, and similarity fallback | a new prospective round must improve on I2B and pass locked interval coverage; fallback remains required |
| I2D — active benchmark planning | Workflow integrated; empirical qualification pending | prospective gap planning plus separately approved digest-bound shard execution | demonstrates reduction in uncertainty or regret while preserving fixed evaluation and OOD families |
| M8 — release hardening | Not started | compatibility matrix, scale benchmarks, security review, API stability, migration policy, release checklist | publication, licence, visibility, and operational approval remain separately authorised |

Passing any synthetic exit gate establishes software behaviour only. It does not establish
linkage validity, fairness, calibration, sensitivity, positive predictive value, false-link
rates, missed-link rates, or operational fitness on a real population or system.
