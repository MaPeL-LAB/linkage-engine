# Operational Validation Runbook

## Scope

This runbook governs the future local validation of a project-specific Linkage Engine implementation. It does not authorise access to data or establish ethical, legal, or organisational approval.

## 1. Authorisation and custody

Record locally:

- approved purpose and scope;
- authorised environment;
- data custodian and analysis custodian;
- permitted users and roles;
- governing protocol, ethics wording, data-sharing terms, and retention policy;
- incident-reporting route;
- approved output recipients.

## 2. Source and configuration assessment

Within the restricted environment:

- inspect source schemas and data-quality patterns;
- map source columns to canonical variable IDs;
- document missingness, duplication, temporal coverage, and source-specific error mechanisms;
- define blocking and comparison rules without embedding project column names in package code;
- confirm candidate and runtime budgets;
- approve the restricted output allow-list.

Never copy the completed configuration or source schema into the repository or ChatGPT.

## 3. Truth and label provenance

Classify every potential label source as one of:

```text
synthetic_truth
verified_human_adjudication
verified_gold_standard
unverified_reference
unknown
```

An existing crosswalk is `unverified_reference` unless independent verification supports a stronger classification. Unknown and unverified pairs must not become implicit nonmatches.

## 4. Protected partitioning

Construct entity–household connected components before candidate-label creation. Allocate complete components to:

```text
training
validation
calibration
decision
final test
```

Verify zero pair, entity, and household overlap. A giant component or inadequate class coverage is a reported limitation, not permission to break the boundary silently.

## 5. Candidate retrieval validation

Before model fitting, report candidate recall at configured K values, zero-candidate rate, candidate-set distribution, blocking-rule contribution, truncation, and Cartesian reduction. Revise blocking using training/development evidence only; keep the locked test partition untouched.

## 6. Model fitting and champion selection

- fit Fellegi–Sunter and supervised challengers on their eligible data only;
- use the validation partition for model and hyperparameter comparison;
- record all candidate models, metrics, feature schemas, versions, and selection rationale;
- do not use the calibration, decision, or test partitions to select the champion.

## 7. Probability calibration

Fit the selected calibrator on the independent calibration partition. Evaluate reliability, Brier score, calibration intercept/slope, and calibration error. Reject calibration artifacts with invalid provenance, non-monotone behaviour, or failed integrity checks.

## 8. Decision policy and thresholds

Use the protected decision partition to assess candidate thresholds, probability margins, no-match utility, and review burden. Threshold approval must consider false links, missed links, subgroup/missingness performance, downstream consequences, and adjudication capacity. Thresholds are project-specific and versioned.

## 9. Locked final test

Evaluate once after model, calibrator, assignment utility, and decision policy are frozen. Report:

- candidate recall and zero-candidate rate;
- sensitivity and PPV;
- false-link and missed-link rates;
- precision–recall results;
- calibration diagnostics;
- ranking metrics;
- assignment and no-match accuracy;
- status and review counts;
- missingness, candidate-set-size, source-pair, and other approved strata;
- runtime and memory in the actual local environment.

## 10. Human adjudication

Use restricted review artifacts only. Decisions are append-only, versioned, and attributable to a local role or pseudonymous reviewer ID. Corrections supersede earlier events. Review decisions do not automatically become training truth.

## 11. Downstream sensitivity

Where linked data support research analyses, evaluate the likely influence of false and missed links on downstream estimates. Document assumptions, sensitivity analyses, and populations with elevated linkage uncertainty.

## 12. Approval and monitoring

Before operational use, obtain documented approval for the exact engine version, configuration digest, artifact set, thresholds, output contract, and intended use. Monitor source drift, candidate-set changes, missingness, calibration, review burden, and unexpected decision patterns. A material change requires revalidation.
