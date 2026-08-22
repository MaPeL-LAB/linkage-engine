# I2 Advisor-v3.1 Empirical Qualification

## Qualification boundary

The first advisor-v3.1 qualification was executed on 2026-08-22 only after a fresh governance-only
re-audit bound the frozen synthetic source registry to the current evaluator. Separate human
approval authorized access to the 12 locked and 12 OOD families. Advisor-v3.1 remains a disclosed
post-corpus evidence-contract amendment and is not represented as the original v3 preregistration.

Recipe utility was aggregated at the family level from exactly 20 cells per recipe for each of 48
meta-training, 12 conformal, and 12 locked families. OOD evaluation used the 12 observable
mechanism profiles and the preregistered distance geometry. OOD metric sidecars were parsed only
for digest integrity; their values were excluded from fitting, threshold selection, interval
calibration, and qualification.

The qualification bound remediation approval
`bdb47868dffc02d0db1d207ed14a116d16105533fe6fc36afe47f69b9458638d` and readiness
`fe2239fb40b1ce4091c453f674f43cd16404ad678fb5b5f5e67712b69463fe62`.

## Result

The aggregate result is `not_qualified`. No threshold was changed after protected outcomes were
read, and no automatic promotion occurred.

| Measure | Result | Gate outcome |
|---|---:|---|
| Stage-2 mean regret | 0.000000000000 | descriptive pass |
| Stage-2 improvement over best fixed recipe | 0.005468332702 | pass |
| Stage-2 top-1 / top-2 oracle coverage | 1.000 / 1.000 | pass |
| Stage-2 selection stability | 0.996527777778 | pass |
| Stage-3 mean regret | 0.005468332702 | descriptive result |
| Stage-3 improvement over Stage 2 | -0.005468332702 | fail |
| Stage-3 top-1 / top-2 oracle coverage | 0.583333333333 / 1.000 | pass for top-2 gate |
| Stage-3 selection stability | 1.000000000000 | pass |
| Locked conformal coverage | 1.000000000000 | pass |
| Mean conformal interval width | 0.283697101111 | pass |
| OOD detection | 12 / 12 families | pass |
| Locked false abstention | 0 / 12 families | pass |
| Learning-curve tail regret range | 0.000000000000 | pass |
| Hard authority-constraint violations | 0 | pass |

Stage 2 passed every fixed v3.1 gate. Stage 3 failed only the prespecified improvement gate: its
mean regret was worse than Stage 2 by `0.005468332702`, rather than at least `0.010` better. The
mandatory outcome is therefore similarity fallback; the learned meta-ranker is not empirically
qualified for promotion.

The [canonical aggregate artifact](../evidence/advisor_v31_qualification_20260822.json) has artifact
digest `b633eac62b463f871ec2c34dec4a8481a81346e7acf3a112f83151ad33342fac` and report digest
`5645e65dfdd24ae9f511417e8ead16b0d2cae538c3d13e8ac24c2f347702d4b9`. It contains no record
values, identifiers, candidate pairs, local paths, approval reference, labels, or score vectors.

## Consequence

The Stage-2 similarity advisor remains the required advisory fallback. Stage 3 cannot be promoted
from this evidence. The result grants no relationship-decision, assignment, merge, model-release,
or operational authority. Runtime mechanism-profile production and operational validity remain
`not_established`.

A later scientific round must be prospective and must use a new versioned policy and new locked
families before evaluating any revised Stage-3 method. These protected outcomes cannot be reused
to tune the current model or thresholds.
