# Validation Plan

## Principle

Synthetic tests validate software behaviour. Operational linkage quality requires approved real-world validation under separate governance.

## Label eligibility

Only purpose-eligible verified truth supports supervised metrics. Unverified crosswalks, weak rules, and unknown pairs are not truth.

## Grouped split design

1. identify verified entities and households/groups;
2. build the entity–household graph;
3. compute connected components;
4. assign complete components deterministically to partitions;
5. verify zero entity/household overlap.

Preferred partitions where volume permits:

```text
training
validation
calibration
decision-threshold selection
final test
```

A giant component that prevents target proportions is a reported limitation, not permission to break the group boundary silently.

## Candidate retrieval metrics

- true-candidate retrieval rate;
- recall@1, @5, @10, and configured K;
- zero-candidate rate;
- mean, median, p95, p99 candidate-set size;
- candidate counts per rule;
- incremental recall per blocking rule;
- pair-budget and truncation diagnostics.

## Pair metrics

- sensitivity/recall;
- positive predictive value/precision;
- false-link rate;
- missed-link rate;
- average precision;
- precision–recall curve;
- threshold confusion matrices.

ROC AUC may be secondary but does not replace precision–recall analysis for highly imbalanced linkage.

## Ranking metrics

- recall@K;
- mean reciprocal rank;
- true-match rank distribution;
- fraction ranked first;
- NDCG where graded relevance is meaningful.

## Calibration metrics

- reliability diagram;
- Brier score;
- calibration slope/intercept;
- expected calibration error or documented alternative;
- observed event rate by probability bin;
- before/after comparison.

Brier score is a proper score but is not a pure calibration measure; interpret it alongside reliability and resolution [@sklearncalibration2026].

## Assignment metrics

- correct assignment rate;
- false assignment rate;
- missed assignment rate;
- no-match accuracy;
- constraint violation count;
- percentage changed from unconstrained top-1;
- conflicts routed to review.

Constraint violations must be zero for a successful constrained run.

## Stratified evaluation

At minimum:

- missingness pattern;
- observed-variable count;
- candidate-set size;
- dataset/source pair;
- blocking-rule provenance;
- anchor status;
- duplication context.

Rare strata are pooled under a documented rule; they are not silently omitted.

## Hard negatives

Use only eligible verified nonmatches. Prioritize same-block competitors, high-similarity nonmatches, contradictory-field examples, assignment competitors, and boundary cases. Unknown pairs remain unknown.

## Test integrity

The final test partition may not select models, features, hyperparameters, blocking rules, calibration method, assignment utility, or decision thresholds.

## Required report statement

> **Synthetic testing establishes software behaviour only. It does not validate linkage accuracy, calibration, fairness, sensitivity, positive predictive value, false-link rates, missed-link rates, or operational fitness on real populations or systems.**

## M2E supervised-model safeguards

Before supervised fitting, label snapshots must pass a partition-disjointness
check covering private pair digests, entity components, and household
components. Duplicate partitions, duplicate/conflicting pair labels, or any
cross-partition overlap are validation failures.

The XGBoost challenger is fitted only on the training partition. Hard-negative
selection retains all eligible verified matches and selects only eligible
verified nonmatches from the bounded candidate-feature table. Unknown pairs are
never recoded as nonmatches.

Model comparison uses the validation partition. M2E reports aggregate average
precision, ROC AUC, Brier score, sensitivity, positive predictive value,
false-link rate, and missed-link rate at a fixed diagnostic threshold. These
metrics are labelled according to their evidence scope. Synthetic metrics are
`synthetic_mechanical_evaluation`; they do not establish operational validity.

Calibration and decision-threshold selection remain separate subsequent stages.
The M2E model and validation report explicitly retain:

```text
calibration_status = not_calibrated
threshold_authority = diagnostic_only
decision_authority = evidence_only
real_data_validation_status = not_established
```
