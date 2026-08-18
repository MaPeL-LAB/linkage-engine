# Label provenance declaration

Complete locally for every proposed truth source.

- Source type: `synthetic_truth`, `verified_human_adjudication`, `verified_gold_standard`, or `unverified_reference`
- Verification protocol: `REPLACE_LOCALLY`
- Source artifact digest: `REPLACE_LOCALLY`
- Entity grouping authority: `REPLACE_LOCALLY`
- Household grouping authority: `REPLACE_LOCALLY`
- Eligible for training: `yes/no`
- Eligible for validation: `yes/no`
- Eligible for calibration: `yes/no`
- Eligible for decision-threshold selection: `yes/no`
- Eligible for final testing: `yes/no`
- Approval reference: `REPLACE_LOCALLY`

An `unverified_reference` must be marked `no` for every modelling and evaluation use.
