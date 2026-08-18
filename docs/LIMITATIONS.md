# Current Limitations

## Evidence and validation boundary

- The complete M2 workflow is validated only as generated-synthetic software behaviour.
- No real-data, population, subgroup, fairness, calibration, threshold, or operational validation has occurred.
- Synthetic metric thresholds in examples and regression guards are mechanical test settings, not operational recommendations.
- Operational champion selection, calibration, decision thresholds, and final test evaluation require locally approved verified truth.

## Current functional scope

- The complete vertical slice supports two-source `link_only` with one-to-one assignment.
- `dedupe_only`, `link_and_dedupe`, many-to-one, one-to-many, unconstrained, and multi-source resolution remain later milestones.
- Candidate generation supports the package-owned exact, prefix, conjunction, and disjunction subset; date-window candidate retrieval remains deferred even though date-window anchor evidence is supported.
- The package-owned Fellegi–Sunter reference estimator is the tested scoring path. A Splink 4 settings compiler exists; broader runtime parity and native Splink model lifecycle remain bounded adapter work.
- The ranker uses comparison features and verified binary relevance. Graded adjudication relevance and more advanced ranking objectives remain later work.
- The review queue is export-only in M2. Append-only adjudication import, superseding events, double review, and label promotion belong to M3.
- The engine emits relationship evidence and statuses; it never constructs a consolidated master record.

## Engineering and deployment limitations

- Python 3.12 is the only supported runtime in the initial compatibility contract.
- The initial CI and synthetic acceptance environment is Linux; macOS and Windows scripts are provided but platform smoke tests remain release-hardening work.
- Determinism means repeatability inside a recorded software/hardware envelope, not universal cross-platform bitwise identity.
- Performance and memory behaviour at operational scale have not been benchmarked.
- Host-level filesystem, process, and network controls remain the responsibility of the authorised local environment.
- No licence has been selected, and package publication remains blocked.
- Privacy-preserving record linkage and optional neural models are separate, later threat-modelled workstreams.

These limitations are intentional and must remain visible in reports, model manifests, user documentation, and manuscript claims.
