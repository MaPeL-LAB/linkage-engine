# Output Governance

## Default deny

No source value or metadata field is exported unless explicitly allowed by both project configuration and package policy.

## Relationship output

The standard restricted relationship record may include:

```text
relationship_id
source_dataset_id
target_dataset_id
source_record_ref
target_record_ref
relationship_status
model_version
calibrated_probability
candidate_rank
probability_margin
decision_rule_id
assignment_method
run_id
configuration_digest
non_sensitive_provenance
```

Source record references should be run-local surrogates by default. Re-identification mappings remain local and restricted.

## Review output

Adjudication exports may contain explicitly approved fields needed for review. They remain under `private/`, are never logged, and never enter CI or source control.

## Aggregate reports

Aggregate reports may include counts, rates, metric curves, performance strata, timings, and version metadata. Small-cell suppression and local disclosure controls may be required in operational environments.

## No master record

The engine does not construct a consolidated person/entity record. Survivorship rules, source precedence, and master-data output require separate governance and architecture.
