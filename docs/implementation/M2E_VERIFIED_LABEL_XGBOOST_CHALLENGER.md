# M2E Verified-Label XGBoost Challenger

Status: implementation candidate

## Purpose

M2E adds the first supervised pair-classifier challenger while preserving the
separation between verified evidence, model fitting, probability calibration,
assignment, and final relationship decisions.

The increment has four linked trust boundaries:

1. verified label provenance and protected partitions;
2. deterministic construction of numeric comparison-feature matrices;
3. bounded, reproducible hard-negative selection and XGBoost fitting; and
4. privacy-safe native model artifacts and aggregate validation diagnostics.

M2E consumes the canonical M2C comparison features. It does not ingest raw
source field values and does not embed study-specific column names in model
logic.

## Eligible label authority

Supervised fitting accepts only:

```text
synthetic_truth
verified_human_adjudication
verified_gold_standard
```

Every label batch records a verification protocol, source-artifact digest,
protected partition, deterministic label-authority digest, and private entity
and household component digests. An unverified crosswalk or reference remains
ineligible for training, validation, calibration, threshold selection, or
final testing.

The partition contract rejects:

- duplicate or contradictory labels for one pair;
- duplicate protected partition snapshots;
- the same pair in more than one partition;
- an entity component in more than one partition;
- a household component in more than one partition;
- malformed label-source or verification metadata.

Pair references and component digests are excluded from public object
representations and translated errors.

## Matrix construction

`DuckDBVerifiedMatrixBuilder` joins a verified label snapshot to the bounded
M2C feature table through private surrogate pair references. It returns an
immutable matrix containing only package-generated numeric comparison
features, explicit missingness indicators, private pair references, and
aggregate structural digests.

Unknown pairs do not become implicit nonmatches. A labelled matrix contains
only pairs with an eligible verified label. Training requires at least one
verified match and one verified nonmatch.

## Hard-negative selection

Training selection is deterministic and restricted to the training partition.
The procedure:

1. retains every eligible verified match, subject to the configured training
   budget;
2. ranks eligible verified nonmatches using package-owned comparison-feature
   hardness signals;
3. selects the configured hard-negative fraction;
4. fills remaining capacity using a seeded SHA-256 tie-break; and
5. records the selected pair digests and labels in a selection digest.

All selected pairs already passed candidate generation and comparison-feature
construction. The procedure does not relabel unknown pairs.

## XGBoost model

The initial challenger uses native XGBoost `DMatrix` and `xgb.train` interfaces
with:

```text
objective = binary:logistic
training method = histogram trees
fixed random seed
single-thread execution
bounded boosting rounds
bounded tree depth
fixed row and column subsampling
```

Feature names are the generated internal M2C feature columns. They are retained
in the native model so scoring must match the exact feature schema and order.

The package saves the model in XGBoost native JSON rather than pickle or
joblib. The artifact manifest records aggregate metadata and digests for the
model, parameters, feature schema, label authority, training selection, random
seed, class counts, XGBoost version, calibration status, and decision
authority. It contains no pair references, source values, identifiers, or
training rows.

## Score and evaluation authority

M2E scores are marked:

```text
probability_status = model_score_uncalibrated
calibration_status = not_calibrated
decision_authority = evidence_only
real_data_validation_status = not_established
```

The model may be evaluated on a nontraining verified partition using aggregate
average precision, ROC AUC, Brier score, sensitivity, positive predictive
value, false-link rate, and missed-link rate. The threshold in this report is
fixed for diagnostics and is explicitly marked `diagnostic_only`; it cannot be
used as an operational decision threshold.

Synthetic evaluation is labelled `synthetic_mechanical_evaluation`. It is not
presented as real-population validation.

## Configuration

The boosted-tree plan is bounded by validated configuration:

```text
n_estimators
max_depth
learning_rate
subsample
column_sample
maximum_training_pairs
hard_negative_fraction
n_jobs = 1
deterministic_mode = true
require_verified_labels = true
```

The training-pair budget may not exceed the runtime candidate-pair budget.
Project configuration cannot provide an XGBoost callable, Python import, raw
SQL, arbitrary objective, executable expression, or unbounded parameter map.

## Reproducibility metadata

Run manifests record the installed versions of DuckDB, Linkage Engine, NumPy,
Pydantic, PyYAML, scikit-learn, Splink, and XGBoost. Model artifacts separately
record the XGBoost version and deterministic training digests.

Repeatability is defined within the recorded Python 3.12 software and hardware
execution envelope; synthetic tests do not imply cross-platform or
cross-version bitwise identity.

## Privacy and governance boundary

M2E does not:

- use real records in tests or CI;
- print pair references, feature values, labels, or model scores in logs;
- retain source field values in the model artifact or unrestricted manifest;
- infer nonmatch labels from absent links;
- accept an unverified crosswalk as truth;
- independently calibrate model scores;
- select operational thresholds;
- rank candidates with a learned ranker;
- solve assignment;
- emit relationship statuses;
- merge records or create master entities.

## Acceptance tests

M2E tests cover:

- eligible and ineligible label-source types;
- duplicate and conflicting labels;
- pair/entity/household partition disjointness;
- value-hiding label and matrix representations;
- exact feature-schema retention and mismatch rejection;
- deterministic hard-negative selection and pair-budget enforcement;
- deterministic native XGBoost JSON artifacts;
- artifact digest validation and restricted output paths;
- evidence-only and uncalibrated score metadata;
- aggregate validation diagnostics on a nontraining partition;
- package-version provenance in run manifests;
- source-value and pair-reference privacy sentinels.

> Synthetic testing establishes software behaviour only. It does not validate
> linkage accuracy, calibration, fairness, sensitivity, positive predictive
> value, false-link rates, missed-link rates, or operational fitness on real
> populations or systems.
