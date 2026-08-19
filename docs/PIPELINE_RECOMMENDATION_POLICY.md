# Pipeline Recommendation Policy

## Governing rule

The Linkage Strategy Advisor recommends which complete pipeline strategies should be evaluated.
It does not select identities, assignment edges, thresholds, or an operational champion.

## Decision order

```text
1. hard eligibility constraints
2. candidate-retrieval viability
3. structural Pareto frontier
4. family-diverse bounded shortlist
5. explicit abstention and local confirmation requirements
```

A failed hard constraint cannot be offset by a weighted utility score.

## Stage-1 eligibility examples

| Condition | Result |
|---|---|
| No eligible verified labels for new supervised development | Supervised pair models and learned rankers are excluded |
| Approved supervised recipe used for compatible inference | Labels are not required merely to execute the approved artifact |
| LightGBM runtime unavailable | LightGBM candidates are excluded |
| PyTorch runtime unavailable | Neural candidates are excluded |
| Stacking lacks protected out-of-fold predictions | Stacking is excluded |
| Candidate retrieval has failed | A recipe depending on that plan is excluded |
| Locked test partition requested for recommendation | The advisor fails safely |

## Structural Pareto analysis

Before benchmark evidence exists, the advisor may compare declared structural attributes:

```text
verified-label requirement
optional-runtime burden
structural complexity
interaction capacity
interpretability
artifact portability
```

These are design attributes, not estimates of sensitivity, PPV, calibration, review burden, or
operational performance.

## Required Stage-1 output

A Stage-1 recommendation includes:

```text
task-profile digest
intent
eligibility and utility policy digests
mandatory baseline
eligible shortlist
structural Pareto members
disqualified candidates and reason codes
applied-rule explanations
empty evidence contributions
coverage_status = structural_only
abstained_from_empirical_ranking = true
empirical_performance_claims = none
required local confirmation steps
```

## Fixed authority

```text
recommendation_authority = advisory_only
decision_authority = none
assignment_authority = none
merge_authority = none
automatic_promotion = prohibited
operational_validity = not_established
```
