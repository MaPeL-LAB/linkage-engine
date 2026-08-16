# Staged Milestones

| Milestone | Scope | Exit gate |
|---|---|---|
| M0 — research and repository baseline | documentation, ADRs, package shell, privacy controls, dependency plan | repository verification and scaffold tests pass |
| M1 — safe configuration foundation | Pydantic schema, loader, compiler, registries, path policy, safe errors/logging, manifests, synthetic generator | malicious/invalid config tests and privacy sentinels pass |
| M2 — complete two-source synthetic slice | normalization, anchors, blocking, comparisons, Splink, XGBoost classifier/ranker, calibration, one-to-one no-match assignment, decisions, evaluation | deterministic synthetic `run` and all acceptance gates pass |
| M3 — adjudication and label lifecycle | queue export/import, append-only events, hard negatives, grouped split, active-learning ordering | provenance and leakage tests pass |
| M4 — extended linkage modes | dedupe-only, link-and-dedupe, many-to-one, one-to-many, unconstrained | mode-specific invariants pass |
| M5 — champion–challenger and ensemble | LightGBM challenger, model selection policy, optional stacking | locked test remains unused for selection |
| M6 — optional neural matcher | feature-based PyTorch matcher, calibration, reproducibility envelope | optional dependency and privacy tests pass |
| M7 — multi-source resolution | source-aware graph/partition logic, cannot-link constraints, conflict review | contradictory clusters cannot be silently emitted |
| M8 — release hardening | compatibility matrix, performance benchmarks, model cards, security review, release checklist | publication remains separately approved |

M0 is the scope of the initial repository bundle. M1 and later are implementation work, not implied by the presence of documentation.
