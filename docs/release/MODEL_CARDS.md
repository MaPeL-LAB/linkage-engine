# Model Cards

These concise cards describe package model families and their authority boundaries. Synthetic
runtime verification is not evidence of population validity.

| Family | Input | Output | Protected fitting boundary | Authority | Principal limitations |
|---|---|---|---|---|---|
| Fellegi–Sunter reference/native Splink | package-owned comparison levels | uncalibrated match-weight evidence | training/EM evidence only; downstream calibration remains separate | evidence only | conditional-independence and estimation assumptions; no relationship decision |
| XGBoost pair classifier | derived comparison features | uncalibrated pair score | eligible verified training labels | evidence only | distribution shift, label quality, and calibration remain separate |
| LightGBM pair classifier | derived comparison features | uncalibrated pair score | eligible verified training labels | evidence only | optional runtime; no identity or threshold authority |
| XGBoost candidate ranker | grouped derived comparison features | rank and top-K ordering | verified grouped training evidence | ordering only | query-side semantics are fixed; rank is not a match decision |
| LightGBM candidate ranker | grouped derived comparison features | rank and top-K ordering | verified grouped training evidence | ordering only | optional runtime; target-query output is not silently reinterpreted |
| PyTorch tabular matcher | derived comparison features, never raw identifying text | uncalibrated pair score | deterministic CPU training on eligible verified labels | evidence only | optional runtime; model privacy and calibration require local controls |
| Stacking ensemble | protected out-of-fold base-model evidence | uncalibrated meta-score | family/entity-group-disjoint out-of-fold training | evidence only | base-model provenance and leakage controls must remain bound |
| Stage-2 similarity advisor | aggregate scenario-family profiles and evidence | advisory shortlist/ranking or abstention | no model fitting; meta-training evidence only | advisory only | runtime profile production and operational validity unestablished |
| Stage-3 meta-ranker | aggregate family features and recipe utilities | advisory predicted utility with conformal interval | meta-training fit; conformal families only for intervals | advisory only | v3.1 failed the regret-improvement gate; similarity fallback is mandatory |

## Shared prohibitions

- Uncalibrated model scores cannot emit relationship status.
- Calibration fits only on the protected calibration partition.
- Assignment selects compatible edges but does not classify relationships.
- Only the decision policy emits relationship status.
- Advisors cannot promote models or change decision, assignment, or merge state.
- No model has master-record overwrite or silent merge authority.
- Operational model artifacts remain local and restricted.
