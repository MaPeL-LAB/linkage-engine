# I2 Advisor Empirical Qualification

## Qualification boundary

The first prospective advisor-v2 qualification was executed on 2026-08-21 after the complete
execution-v2 registry passed its independent evidence audit. The registry contained exactly
9,800 retained synthetic benchmark runs: 4,200 successful runs from the three truth-safe core
adapters and 5,600 explicitly ineligible runs from unsupported adapters. The statistical unit was
the scenario family, not an instance or replicate.

The family roles were fixed before fitting or locked evaluation:

```text
meta-training       40 families
conformal            8 families
locked evaluation    8 families
true-mechanism OOD   8 families
```

The utility estimand was also shared and fixed before locked access:

```text
0.4 * recall@1 + 0.4 * positive predictive value + 0.2 * (1 - Brier score)
```

Stage 2 retrieved only meta-training families. Stage 3 fit only on meta-training families,
calibrated its split-conformal residual on the conformal families, and evaluated the locked
families only after fitting. OOD families entered neither model fitting nor interval calibration.
The learned predictions now determine the order of supported Stage-3 candidates; unsupported
candidates retain their Stage-2 order and the mandatory baseline remains present.

## Prespecified gates

The immutable policy required all of the following:

| Gate | Required |
|---|---:|
| Stage-2 mean-regret improvement over best fixed recipe | at least 0.005 |
| Stage-3 mean-regret improvement over Stage 2 | at least 0.010 |
| Stage-2 and Stage-3 top-2 oracle coverage | at least 0.875 |
| Stage-2 and Stage-3 leave-one-training-family-out selection stability | at least 0.800 |
| Locked split-conformal coverage | at least 0.800 |
| Mean conformal interval width | at most 0.500 |
| OOD-family detection rate | at least 0.750 |
| Locked-family false-abstention rate | at most 0.125 |
| Final learning-curve tail regret range | at most 0.020 |

Thresholds are package-owned and are not exposed as CLI tuning options. The five nested learning
curve sizes are 8, 16, 24, 32, and 40 training families.

## Result

The result is `not_qualified`. No threshold was changed after the protected outcomes were read.

| Measure | Result | Gate outcome |
|---|---:|---|
| Stage-2 mean regret | 0.000000 | descriptive pass |
| Stage-2 top-1 / top-2 oracle coverage | 1.000 / 1.000 | pass |
| Stage-2 selection stability | 1.000 | pass |
| Stage-2 improvement over best fixed recipe | 0.000000 | fail |
| Stage-3 mean regret | 0.000000 | descriptive pass |
| Stage-3 top-1 / top-2 oracle coverage | 1.000 / 1.000 | pass |
| Stage-3 selection stability | 1.000 | pass |
| Stage-3 improvement over Stage 2 | 0.000000 | fail |
| Locked conformal coverage | 0.791667 | fail |
| Mean conformal interval width | 0.024856 | pass |
| OOD detection | 0 / 8 families | fail |
| Locked false abstention | 0 / 8 families | pass |
| Learning-curve tail regret range | 0.000000 | pass |
| Hard authority-constraint violations | 0 | pass |

The fixed XGBoost classifier was the oracle for all eight locked families and therefore had zero
mean regret. Stage 2 and Stage 3 also selected it for every locked family, so neither method could
meet a positive improvement gate. Interval coverage was 19 of 24 locked family-by-recipe cells,
one cell below the fixed 80 percent threshold. The current task meta-feature representation did
not distinguish any of the eight true-mechanism OOD families from the training envelope.

The [canonical aggregate report](../evidence/advisor_v2_qualification_20260821.json) has artifact digest
`ffb6f2b5b29856e0e40fba0999803a931fd967d9027523abedd330f2c135a4cd` and report digest
`b796950198d57d0686ab54cee6a48987f81fec9b1f264a217ccfc3265921a0a1`. It contains no record
values, source identifiers, candidate pairs, local paths, approval reference, labels, or score
vectors.

## Consequence

Stage 2 and Stage 3 remain integrated software components but are not empirically qualified for
automatic evidence-backed promotion. Stage 3 must fall back to Stage 2, local confirmation remains
required, and the active planner may only propose additional synthetic experiments. Automatic
promotion, relationship decisions, assignment, merging, release approval, and operational use
remain prohibited.

Runtime learned ranking now requires a canonical `qualified` artifact bound to the exact registry
snapshot, design digest, and immutable policy digest. A complete registry by itself cannot activate
Stage 3. A missing, `not_qualified`, forged, stale, or differently bound artifact produces a
similarity fallback and no fitted Stage-3 recommendation.

A new scientific round must be prospective. It must first revise the task meta-features so the
held-out mechanisms are representable and expand the design so more than one core recipe is
competitive. New families and thresholds require a new versioned policy and new locked families;
the evaluated advisor-v2 locked outcomes cannot be reused for tuning.

## Reproduction

The quick, approved aggregate qualification command is:

```text
mapel-linkage qualify-advisor \
  --project-root . \
  --registry-dir private/benchmark_registry/advisor_v2_execution_v2 \
  --shards 32 \
  --replicates 5 \
  --output artifacts/advisor_qualification/advisor_v2_qualification.json \
  --approve-locked-evaluation \
  --approval-reference NON_IDENTIFYING_REFERENCE
```

The command is idempotent only for an exact canonical artifact. Missing cells, mixed provenance,
tampering, a conflicting output, path escape, or symbolic-link traversal fails closed. The output
is aggregate synthetic evidence and retains `operational_validity = not_established`.
