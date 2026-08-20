# ADR-0004: Capability Status and Pipeline Integration Boundary

**Status:** Accepted  
**Date:** 2026-08-19

**I1C amendment:** 2026-08-20

## Context

Linkage Engine now contains substantial implementations from M0 through M7. The complete
configuration-driven workflow, however, remains the two-source generated-synthetic M2 slice.
Several later model, adjudication, assignment, deduplication, ensemble, neural, and
multi-source components exist below the orchestration boundary.

Describing every source-code component as an implemented platform workflow would conflate:

1. specification;
2. component implementation;
3. configuration and CLI integration;
4. execution under the real optional dependency;
5. operational validation on an approved population.

The same distinction applies to automated tests. A test that was collected and skipped
because an optional dependency was unavailable is not a passed runtime test.

## Decision

### 1. Maintain a package-owned capability registry

`mapel_linkage.capabilities` is the normative machine-readable registry. Each capability
records:

```text
component_status
workflow_status
runtime_verification
operational_validation
decision_authority
merge_authority
```

The generated [`../CAPABILITY_MATRIX.md`](../CAPABILITY_MATRIX.md) is checked against that
registry.

### 2. Preserve the current authority separation

```text
candidate retrieval
→ comparison evidence
→ pair-model scoring
→ model selection
→ calibration
→ candidate ranking
→ assignment
→ relationship decision
→ adjudication and evaluation
```

Candidate retrieval, pair models, rankers, assignment solvers, clustering algorithms, and
the future strategy advisor cannot independently confirm identity or merge records.

### 3. Verify optional model runtimes explicitly

The normal core CI remains the fast required quality gate. A separate `all-models` job
installs the pinned LightGBM and PyTorch extras, executes the full test suite, and fails if
any test is skipped.

### 4. Integrate later components through immutable pipeline recipes

The next general orchestration layer will separate model development from new-data
inference.

#### Training and approval

```text
validate configuration
→ profile the linkage job
→ approve candidate retrieval
→ create protected partitions
→ fit the Fellegi-Sunter baseline
→ fit only eligible challengers
→ create out-of-fold ensemble inputs where required
→ select the champion on validation evidence
→ select the calibration method without test access
→ select decision thresholds on the protected decision partition
→ evaluate once on the locked test partition
→ approve an immutable PipelineRecipeArtifact
```

#### New-data inference

```text
load an approved PipelineRecipeArtifact
→ verify schema and artifact compatibility
→ preprocess new data
→ generate candidates
→ score with the approved champion
→ optionally score challengers in shadow mode
→ calibrate
→ rank
→ assign
→ apply the explicit decision policy
→ export restricted review cases and aggregate diagnostics
```

New-data inference must not retrain models, reselect the champion, or alter thresholds
without eligible verified truth and an explicit authorised development run.

### 5. Integrate M3 through M7 as explicit workflows

The integration track will add:

- adjudication import, consensus, promotion, and authorised retraining commands;
- `dedupe_only`, `link_and_dedupe`, many-to-one, one-to-many, and unconstrained runners;
- portfolio configuration for XGBoost, LightGBM, PyTorch, and stacking;
- a source-aware N-dataset evidence-graph pipeline;
- cluster-level review and BCubed evaluation where eligible truth exists.

I1C implements a deliberately smaller part of this track: generated-synthetic dispatch for
exactly `link_only` with many-to-one, one-to-many, or unconstrained assignment;
`dedupe_only` with unconstrained assignment; and `link_and_dedupe` with one-to-one assignment.
It binds decision-only evidence, fitted/reloaded pair-model and calibration artifacts, and
aggregate tamper-evident mode artifacts. It does not implement arbitrary M3–M7 orchestration,
multi-source dispatch, real-data execution, operational approval, or merge authority.

### 6. Add the Linkage Strategy Advisor only as advisory infrastructure

The planned advisor will recommend a small set of complete pipeline recipes from
privacy-safe task profiles and benchmark evidence. It must support uncertainty,
out-of-distribution detection, and abstention.

Its fixed authority is:

```text
recommendation_authority = advisory_only
decision_authority = none
assignment_authority = none
merge_authority = none
```

Synthetic evidence may provide prior recommendations, but operational promotion still
requires a bounded local champion-challenger evaluation and governance approval.

## Consequences

### Positive

- Repository claims become auditable and resistant to status drift.
- Optional model tests exercise their real dependencies.
- Component implementation can progress without pretending that orchestration is complete.
- Training, approval, inference, shadow scoring, adjudication, and retraining have separate
  authority.
- The future strategy advisor can reduce redundant model execution without selecting
  identities or operational thresholds.

### Costs

- The capability registry and generated documentation must be updated with each material
  change.
- General orchestration requires new artifact contracts and migration/version policies.
- All-model CI is slower and uses more storage than the core gate.
- Some existing singular model configuration fields will require a versioned migration to
  represent a portfolio safely.

## Rejected alternatives

### Treating source-code presence as workflow completion

Rejected because it overstates what users can execute and obscures missing integration.

### Running every model on every incoming dataset

Rejected because new data commonly lack verified truth, and repeated retraining or model
selection would be statistically invalid and operationally wasteful.

### Allowing the advisor to select the operational model automatically

Rejected because synthetic similarity and prior benchmark performance cannot substitute
for local validation, calibration, threshold selection, and approval.

### Silently skipping optional runtime tests

Rejected because collected-but-skipped tests do not verify LightGBM or PyTorch behaviour.

## Validation

This ADR is satisfied when:

- the capability registry and generated matrix agree;
- CLI status reports component, workflow, runtime, and operational states separately;
- core CI remains green;
- all-model CI installs LightGBM and PyTorch and reports zero skipped tests;
- documentation no longer labels M3 through M7 as unstarted;
- the repository still makes no claim of real-data operational validation.
