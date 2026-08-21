# Documentation Index

Linkage Engine documentation is organized around architecture, governance, implementation,
validation, local operation, and research evidence.

## Start here

- [`../README.md`](../README.md) — project identity, privacy boundary, current platform
  status, CLI, and development gate.
- [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md) — generated component, workflow,
  runtime-verification, and operational-validation status.
- [`ROADMAP.md`](ROADMAP.md) — integrated foundations, implemented components, and remaining
  integration work.
- [`LIMITATIONS.md`](LIMITATIONS.md) — current statistical, orchestration, engineering, and
  deployment limitations.

## Architecture

- [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) — system components and
  authority boundaries.
- [`architecture/ADR-0001-CONFIGURATION-COMPILED-ENGINE.md`](architecture/ADR-0001-CONFIGURATION-COMPILED-ENGINE.md)
  — configuration is validated data, not executable code.
- [`architecture/ADR-0002-SEPARATE-RETRIEVAL-SCORING-ASSIGNMENT-DECISION.md`](architecture/ADR-0002-SEPARATE-RETRIEVAL-SCORING-ASSIGNMENT-DECISION.md)
  — retrieval, scoring, ranking, calibration, assignment, and decisions remain separate.
- [`architecture/ADR-0003-SYNTHETIC-ONLY-REPOSITORY.md`](architecture/ADR-0003-SYNTHETIC-ONLY-REPOSITORY.md)
  — repository, tests, examples, and CI contain synthetic data only.
- [`architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md`](architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md)
  — component versus workflow status, all-model CI, train/approve/infer separation, and the
  advisory strategy layer.

## Configuration and interfaces

- [`CONFIGURATION_REFERENCE.md`](CONFIGURATION_REFERENCE.md) — normative YAML/JSON schema,
  safe operation registry, linkage modes, model declarations, outputs, and validation.
- [`MODEL_INTERFACES.md`](MODEL_INTERFACES.md) — model, ranker, calibrator, assignment, and
  artifact contracts.
- [`schemas/linkage-config.schema.json`](../schemas/linkage-config.schema.json) — generated
  machine-readable schema.

## Governance and privacy

- [`governance/PRIVACY_THREAT_MODEL.md`](governance/PRIVACY_THREAT_MODEL.md)
- [`SYNTHETIC_DATA_POLICY.md`](SYNTHETIC_DATA_POLICY.md)
- [`governance/LABEL_PROVENANCE_POLICY.md`](governance/LABEL_PROVENANCE_POLICY.md)
- [`governance/MODEL_GOVERNANCE.md`](governance/MODEL_GOVERNANCE.md)
- [`governance/OUTPUT_GOVERNANCE.md`](governance/OUTPUT_GOVERNANCE.md)
- [`governance/LICENSING_DECISION.md`](governance/LICENSING_DECISION.md)
- [`ADJUDICATION_WORKFLOW.md`](ADJUDICATION_WORKFLOW.md)

## Validation and operation

- [`VALIDATION_PLAN.md`](VALIDATION_PLAN.md) — protected partitions, retrieval, pair,
  calibration, ranking, assignment, cluster, and stratified evaluation.
- [`OPERATIONAL_VALIDATION_RUNBOOK.md`](OPERATIONAL_VALIDATION_RUNBOOK.md) — local real-data
  validation and approval process.
- [`LOCAL_DEPLOYMENT_GUIDE.md`](LOCAL_DEPLOYMENT_GUIDE.md)
- [`LOCAL_HANDOFF_CHECKLIST.md`](LOCAL_HANDOFF_CHECKLIST.md)
- [`local_templates/`](local_templates/) — generic source mapping, label provenance, and
  output allow-list worksheets.

## Implementation records

- [`implementation/MILESTONES.md`](implementation/MILESTONES.md)
- [`implementation/INITIAL_VERTICAL_SLICE_CHECKLIST.md`](implementation/INITIAL_VERTICAL_SLICE_CHECKLIST.md)
- [`implementation/ACCEPTANCE_CRITERIA.md`](implementation/ACCEPTANCE_CRITERIA.md)
- [`implementation/M1_SAFE_FOUNDATION_REPORT.md`](implementation/M1_SAFE_FOUNDATION_REPORT.md)
- [`implementation/M2A_LOCAL_DATA_PLANE_AND_CANDIDATE_GENERATION.md`](implementation/M2A_LOCAL_DATA_PLANE_AND_CANDIDATE_GENERATION.md)
- [`implementation/M2B_CONFIGURED_INGESTION_AND_PREPROCESSING.md`](implementation/M2B_CONFIGURED_INGESTION_AND_PREPROCESSING.md)
- [`implementation/M2C_COMPARISON_FEATURES_AND_ANCHOR_EVIDENCE.md`](implementation/M2C_COMPARISON_FEATURES_AND_ANCHOR_EVIDENCE.md)
- [`implementation/M2D_FELLEGI_SUNTER_BASELINE.md`](implementation/M2D_FELLEGI_SUNTER_BASELINE.md)
- [`implementation/M2E_VERIFIED_LABEL_XGBOOST_CHALLENGER.md`](implementation/M2E_VERIFIED_LABEL_XGBOOST_CHALLENGER.md)
- [`implementation/M2_COMPLETE_SYNTHETIC_MVP.md`](implementation/M2_COMPLETE_SYNTHETIC_MVP.md)
- [`implementation/I1B_CONFIGURATION_DRIVEN_MODEL_PORTFOLIO.md`](implementation/I1B_CONFIGURATION_DRIVEN_MODEL_PORTFOLIO.md)
- [`implementation/I1C_CONFIGURATION_DRIVEN_LINKAGE_MODES.md`](implementation/I1C_CONFIGURATION_DRIVEN_LINKAGE_MODES.md)
- [`implementation/B1_ADVISOR_SCALE_CORPUS.md`](implementation/B1_ADVISOR_SCALE_CORPUS.md)
- [`implementation/I2_ADVISOR_EMPIRICAL_QUALIFICATION.md`](implementation/I2_ADVISOR_EMPIRICAL_QUALIFICATION.md)

M3 through M7 have a mix of integrated workflows and component-only capabilities. I1B adds
the bounded configuration-driven all-model synthetic portfolio; I1C adds only five exact
generated-synthetic linkage-mode combinations. Neither claims one general CLI for every
milestone or any operational validation. The generated capability matrix is normative when
older implementation reports describe historical status.

## Research evidence

- [`evidence/advisor_v2_qualification_20260821.json`](evidence/advisor_v2_qualification_20260821.json)
  — canonical aggregate-only result from the first prospective advisor-v2 qualification.
- [`research/RESEARCH_SYNTHESIS.md`](research/RESEARCH_SYNTHESIS.md)
- [`research/METHODS_LANDSCAPE.md`](research/METHODS_LANDSCAPE.md)
- [`research/SOFTWARE_LANDSCAPE.md`](research/SOFTWARE_LANDSCAPE.md)
- [`research/RESEARCH_GAPS.md`](research/RESEARCH_GAPS.md)
- [`research/SOURCE_PROVENANCE.md`](research/SOURCE_PROVENANCE.md)
- [`references/references.bib`](references/references.bib)

## Status rule

Documentation must distinguish:

```text
specified
component implemented
workflow integrated
runtime verified
operationally validated
```

A test that is collected and skipped is not described as passed. Passing synthetic checks
establishes software behaviour only and does not establish real-population linkage validity.
