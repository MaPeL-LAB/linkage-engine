# Label Provenance Policy

## Principle

A pair label is not trusted merely because it exists. Every label has source, verification, protocol, snapshot, and purpose-specific eligibility.

## Permitted source types

```text
synthetic_truth
verified_human_adjudication
verified_gold_standard
unverified_reference
weak_rule_evidence
unknown
```

## Required metadata

```text
label_value
label_source_type
verification_status
verification_protocol
source_artifact_digest
created_at
supersedes_label_id
eligible_for_training
eligible_for_validation
eligible_for_calibration
eligible_for_decision_selection
eligible_for_testing
```

## Eligibility rules

- `synthetic_truth` is eligible only for synthetic software evaluation.
- `verified_human_adjudication` is eligible according to the approved review protocol and partition assignment.
- `verified_gold_standard` requires documented independent verification.
- `unverified_reference` may be inspected as evidence but is ineligible for all model/evaluation purposes.
- `weak_rule_evidence` cannot be silently converted into gold truth.
- `unknown` remains unknown.

## Prohibitions

- Unlinked pairs are not automatically nonmatches.
- Existing crosswalks are not truth without verification.
- Deterministic anchors are not gold labels by default.
- Test labels cannot migrate into training without a new versioned evaluation design.
- Corrected adjudication overwrites nothing; it creates a superseding event.

## Partition protection

Label snapshots are assigned to entity/household-disjoint partitions before pair-model development. Every model manifest records the exact eligible snapshot and partition IDs.

## M2E implementation contract

M2E implements supervised-use objects only for `synthetic_truth`,
`verified_human_adjudication`, and `verified_gold_standard`. Other source types
remain policy concepts and cannot be instantiated as an eligible
`VerifiedLabelBatch`.

The implementation rejects duplicate/conflicting pair labels and requires pair,
entity-component, and household-component disjointness across protected
partitions. The deterministic `label_authority_digest` and training-selection
digest are retained in model manifests; private pair and grouping references
are not retained in unrestricted metadata.
