# I1B Configuration-Driven Model Portfolio

## Status

Integrated and all-models-CI verified for generated-synthetic, two-source `link_only`,
`one_to_one` software behaviour. Operational validity, release readiness, real-data approval,
decision thresholds, and merge authority are not established.

## Bounded workflow

`SyntheticPortfolioWorkflowRunner` and `mapel-linkage run-model-portfolio`:

1. compile project configuration schema `0.1` and retain pipeline recipe IO schema v1;
2. generate only package-attested synthetic data with seed `20260816`;
3. prepare five entity/household-disjoint training, validation, calibration, decision, and
   locked-test matrices;
4. fit, serialize, strictly reload, and score the mandatory native Splink baseline against
   the exact package-owned candidate set;
5. train configured XGBoost, LightGBM, PyTorch, stacking, XGBoost-ranker, and
   LightGBM-ranker candidates when enabled;
6. build stacking inputs from source-side entity/household-connected OOF groups and reject
   missing or inconsistent group provenance;
7. select pair and executable source-query ranking artifacts on validation only;
8. fit the selected calibrator on calibration only;
9. evaluate the frozen champion and calibrator on locked test without permitting selection
   or calibration access;
10. persist and strictly reload the champion bundle, calibrator, ranker, and recipe-v1
    binding under `PathPolicy`; and
11. prove two source-disjoint, package-attested decision-partition replays through the
    reloaded recipe.

## Integrity and authority boundaries

- Retrieval indexes candidates and has no relationship authority.
- Pair models emit evidence only; rankers emit ordering only.
- Native score evidence is an integrity bridge, not an inference capability. When Splink is
  champion, inference recomputes scores from typed prepared data, settings, and the persisted
  native artifact.
- Bare scores are accepted only in development mode; synthetic and operational inference
  require typed replay.
- A target-query ranker is reported but cannot be used by the source-to-target assignment
  contract.
- Assignment selects compatible edges and cannot classify relationships.
- Only the configured relationship policy emits status. No stage can merge records.
- CLI and artifact summaries contain aggregates and digests, not rows, pair references,
  source values, or local paths.

## Residuals

- This is not a general runner for every linkage mode or M3–M7 workflow.
- Optional LightGBM and PyTorch paths require the pinned all-models environment.
- Operational model cards, population/subgroup validation, calibration evidence, thresholds,
  monitoring, rollback, and approval remain local human-governed work.
- The repository does not contain or authorize real participant or operational data.
