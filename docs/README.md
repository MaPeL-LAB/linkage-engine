# Documentation Index

The repository currently contains the M0 research baseline and the M1 safe configuration implementation candidate.

This directory is the normative documentation set for Linkage Engine.

## Research

- [`research/RESEARCH_SYNTHESIS.md`](research/RESEARCH_SYNTHESIS.md) — evidence-based architectural synthesis.
- [`research/METHODS_LANDSCAPE.md`](research/METHODS_LANDSCAPE.md) — established, alternative, emerging, and deferred methods.
- [`research/SOFTWARE_LANDSCAPE.md`](research/SOFTWARE_LANDSCAPE.md) — software and framework comparison.
- [`research/RESEARCH_GAPS.md`](research/RESEARCH_GAPS.md) — questions not resolved by the current evidence base.
- [`research/SOURCE_PROVENANCE.md`](research/SOURCE_PROVENANCE.md) — provenance and citation-normalisation note.

## Architecture

- [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) — target package architecture and execution model.
- [`architecture/ADR-0001-CONFIGURATION-COMPILED-ENGINE.md`](architecture/ADR-0001-CONFIGURATION-COMPILED-ENGINE.md)
- [`architecture/ADR-0002-SEPARATE-RETRIEVAL-SCORING-ASSIGNMENT-DECISION.md`](architecture/ADR-0002-SEPARATE-RETRIEVAL-SCORING-ASSIGNMENT-DECISION.md)
- [`architecture/ADR-0003-SYNTHETIC-ONLY-REPOSITORY.md`](architecture/ADR-0003-SYNTHETIC-ONLY-REPOSITORY.md)

## Contracts

- [`CONFIGURATION_REFERENCE.md`](CONFIGURATION_REFERENCE.md)
- [`MODEL_INTERFACES.md`](MODEL_INTERFACES.md)
- [`VALIDATION_PLAN.md`](VALIDATION_PLAN.md)
- [`SYNTHETIC_DATA_POLICY.md`](SYNTHETIC_DATA_POLICY.md)
- [`ADJUDICATION_WORKFLOW.md`](ADJUDICATION_WORKFLOW.md)

## Governance

- [`governance/PRIVACY_THREAT_MODEL.md`](governance/PRIVACY_THREAT_MODEL.md)
- [`governance/LABEL_PROVENANCE_POLICY.md`](governance/LABEL_PROVENANCE_POLICY.md)
- [`governance/MODEL_GOVERNANCE.md`](governance/MODEL_GOVERNANCE.md)
- [`governance/OUTPUT_GOVERNANCE.md`](governance/OUTPUT_GOVERNANCE.md)
- [`governance/LICENSING_DECISION.md`](governance/LICENSING_DECISION.md)

## Delivery

- [`implementation/INITIAL_VERTICAL_SLICE_CHECKLIST.md`](implementation/INITIAL_VERTICAL_SLICE_CHECKLIST.md)
- [`implementation/MILESTONES.md`](implementation/MILESTONES.md)
- [`implementation/ACCEPTANCE_CRITERIA.md`](implementation/ACCEPTANCE_CRITERIA.md)
- [`ROADMAP.md`](ROADMAP.md)
- [`LIMITATIONS.md`](LIMITATIONS.md)

## References

- [`references/references.bib`](references/references.bib) — canonical BibTeX library.
- [`references/README.md`](references/README.md) — citation conventions.

## Document authority

When documents conflict, use this precedence unless an ADR explicitly supersedes it:

1. privacy, security, and label-provenance policies;
2. accepted ADRs;
3. configuration and model-interface contracts;
4. validation and output-governance requirements;
5. implementation checklists and roadmap;
6. research synthesis and comparative landscape documents.

Research documents explain the evidence and trade-offs. Accepted ADRs define the chosen architecture.

## M1 implementation

- [`implementation/M1_SAFE_FOUNDATION_REPORT.md`](implementation/M1_SAFE_FOUNDATION_REPORT.md) — delivered configuration, governance, schema, synthetic generation, tests, limitations, and next gate.
- [`../schemas/linkage-config.schema.json`](../schemas/linkage-config.schema.json) — normative generated configuration schema.

## Active implementation increments

- [`implementation/M2A_LOCAL_DATA_PLANE_AND_CANDIDATE_GENERATION.md`](implementation/M2A_LOCAL_DATA_PLANE_AND_CANDIDATE_GENERATION.md)
