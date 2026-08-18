# Staged Milestones

| Milestone | Status | Scope | Exit gate |
|---|---|---|---|
| M0 — research and repository baseline | Complete | documentation, ADRs, governance, bibliography, package shell, privacy controls, dependency plan | repository verification and scaffold tests pass |
| M1 — safe configuration foundation | Merged | Pydantic schema, safe loader, compiler, registries, path policy, safe errors/logging, manifests, synthetic generator | malicious/invalid configuration, privacy sentinel, Python 3.12, and distribution checks pass |
| M2 — complete two-source synthetic slice | Implementation candidate | canonical preparation, anchors, bounded candidates, comparison features, Fellegi–Sunter, XGBoost pair classifier, validation-only champion selection, independent calibration, learned ranking, one-to-one no-match assignment, four-status decisions, restricted review export, aggregate evaluation, CLI/orchestration | deterministic full synthetic `run`; zero authority violations; exact review-head and post-merge Python 3.12 gates pass |
| M3 — adjudication and label lifecycle | Not started | append-only review import, superseding events, protocol provenance, label promotion, active-learning ordering | provenance, test-protection, and leakage tests pass |
| M4 — extended linkage modes | Not started | dedupe-only, link-and-dedupe, many-to-one, one-to-many, unconstrained | mode-specific pair and assignment invariants pass |
| M5 — broader champion–challenger and ensemble | Not started | LightGBM challenger, model cards, predefined selection policy, optional stacking | locked test remains unused for selection and calibration |
| M6 — optional neural matcher | Not started | feature-based PyTorch matcher, calibration, reproducibility envelope | optional dependency, artifact, determinism, and privacy tests pass |
| M7 — multi-source resolution | Not started | source-aware graph/partition logic, cannot-link constraints, global consistency, conflict review | contradictory clusters cannot be silently emitted |
| M8 — release hardening | Not started | compatibility matrix, scale benchmarks, security review, API stability, migration policy, release checklist | publication and licence remain separately approved |

The M2 implementation is for generated synthetic data only. Passing its acceptance gate establishes software behaviour, not linkage validity on real populations or systems.
