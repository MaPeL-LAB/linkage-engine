# Staged Milestones

| Milestone | Status | Scope | Exit gate |
|---|---|---|---|
| M0 — research and repository baseline | Complete | documentation, ADRs, package shell, privacy controls, dependency plan | repository verification and scaffold tests pass |
| M1 — safe configuration foundation | Implementation candidate | Pydantic schema, loader, compiler, registries, path policy, safe errors/logging, manifests, synthetic generator | malicious/invalid config tests, parser-complexity guards, privacy sentinels, Python 3.12 CI, and package-build checks pass |
| M2 — complete two-source synthetic slice | Not started | normalization, anchors, blocking, comparisons, Splink, XGBoost classifier/ranker, calibration, one-to-one no-match assignment, decisions, evaluation | deterministic synthetic `run` and all acceptance gates pass |
| M3 — adjudication and label lifecycle | Not started | queue export/import, append-only events, hard negatives, grouped split, active-learning ordering | provenance and leakage tests pass |
| M4 — extended linkage modes | Not started | dedupe-only, link-and-dedupe, many-to-one, one-to-many, unconstrained | mode-specific invariants pass |
| M5 — champion–challenger and ensemble | Not started | LightGBM challenger, model selection policy, optional stacking | locked test remains unused for selection |
| M6 — optional neural matcher | Not started | feature-based PyTorch matcher, calibration, reproducibility envelope | optional dependency and privacy tests pass |
| M7 — multi-source resolution | Not started | source-aware graph/partition logic, cannot-link constraints, conflict review | contradictory clusters cannot be silently emitted |
| M8 — release hardening | Not started | compatibility matrix, performance benchmarks, model cards, security review, release checklist | publication remains separately approved |

M1 is not complete for the remote repository until the implementation branch passes the full Python 3.12 CI job and is explicitly approved for merge.
