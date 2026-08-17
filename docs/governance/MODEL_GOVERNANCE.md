# Model Governance

## Model roles

| Model type | Authority |
|---|---|
| deterministic anchor | evidence only by default |
| Fellegi–Sunter | pair evidence score/probability estimate |
| boosted classifier | pair score or calibrated probability |
| candidate ranker | candidate order and top-K membership only |
| neural matcher | challenger pair score only |
| ensemble | combined score/probability under explicit policy |
| assignment solver | capacity-constrained edge selection |
| decision policy | final relationship status |

## Required artifact manifest

Every model artifact records:

- model family and implementation;
- model version;
- engine and dependency versions;
- code commit;
- random seed and thread settings;
- configuration and feature-schema digests;
- label snapshot and eligibility policy;
- training/validation/calibration partition IDs;
- hyperparameter digest;
- calibration artifact linkage;
- intended role and prohibited authority;
- known limitations.

## Champion–challenger policy

Model selection uses validation data and predefined primary/secondary criteria. Test data may not select a model. A challenger does not replace the champion solely because one aggregate metric improves; calibration, stability, subgroup behaviour, review burden, and assignment impact are also considered.

## Serialization

Prefer non-executable native formats. Arbitrary pickle/joblib/PyTorch object loading from untrusted sources is prohibited. Artifact loading verifies format, manifest, digest, model role, and compatible engine version.

## Confirmation gate

A default model-based confirmation requires:

- eligible calibrated probability;
- valid calibration diagnostics;
- configured minimum margin;
- applicable assignment result;
- no blocking/data-quality failure;
- no mandatory-review reason;
- recorded decision rule.

## Model cards

Every operational candidate model requires a model card describing intended use, data provenance, features, validation design, thresholds, calibration, subgroup performance, limitations, and prohibited uses.

## M2E boosted challenger boundary

The initial XGBoost challenger is trained only from a verified training
partition over package-generated comparison features. It is persisted as native
JSON with a safe aggregate manifest and exact model, parameter, feature-schema,
label-authority, and selection digests. Pickle and joblib are not accepted as
canonical artifacts.

Its output is `model_score_uncalibrated`, `not_calibrated`, and
`evidence_only`. Aggregate validation on a nontraining partition may support
champion–challenger review, but cannot itself authorise an operational threshold
or relationship decision.
