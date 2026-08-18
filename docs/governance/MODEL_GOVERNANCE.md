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


## Complete M2 selection and calibration boundary

Fellegi–Sunter and XGBoost are candidate pair-evidence models. Their validation evidence must share the same protected validation label authority and partition manifest. Champion selection records every candidate and cannot access calibration, decision, or test evidence.

The selected champion is calibrated only on the protected calibration partition. Sigmoid and isotonic artifacts are monotone, native/package JSON, bound to the selected source model, and checked for payload and manifest tampering before use.

The learned ranker is separately trained from eligible verified training labels and has ranking-only authority. Assignment receives calibrated probabilities and performs constrained edge selection, but only the decision policy may emit relationship status. No artifact or adapter receives merge authority.

Operational promotion requires a locally approved model card, candidate-retrieval evidence, calibration evidence, threshold rationale, locked-test results, review burden, subgroup/missingness analysis, intended-use statement, and rollback plan. Synthetic CI cannot promote an operational model.
