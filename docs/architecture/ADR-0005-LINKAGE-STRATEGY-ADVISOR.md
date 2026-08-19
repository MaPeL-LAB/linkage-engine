# ADR-0005: Linkage Strategy Advisor

**Status:** Accepted  
**Date:** 2026-08-19

## Context

Linkage Engine contains multiple pair-model, ranking, calibration, assignment, adjudication,
and clustering components. Running every possible model and pipeline on every project is
computationally wasteful and statistically invalid when new data have no eligible verified
truth. Choosing one fashionable model without evidence is equally unsafe.

The platform therefore needs an algorithm-selection capability that recommends a small set of
**complete pipeline strategies** for evaluation. It must not make record-level identity
decisions, approve an operational model, select thresholds from synthetic evidence, or convert
a recommendation into an executable approved recipe automatically.

## Decision

### 1. Keep recommendation separate from approval and inference

```text
PipelineRecommendation
    advisory shortlist and evidence statement

PipelineRecipeArtifact
    locally validated and explicitly approved execution contract
```

A recommendation cannot be promoted automatically into a `PipelineRecipeArtifact`.

### 2. Fix the advisor authority in the type system

Every recommendation and structural candidate has immutable literal authority:

```text
recommendation_authority = advisory_only
decision_authority = none
assignment_authority = none
merge_authority = none
automatic_promotion = prohibited
operational_validity = not_established
```

These values are constructor invariants, not ordinary documentation or Python `assert`
statements.

### 3. Make lifecycle intent explicit

Eligibility depends on why advice is requested:

```text
develop_new_recipe
evaluate_challengers
fit_or_select_calibration
infer_with_approved_recipe
shadow_score_challenger
plan_benchmark
```

No eligible verified labels means that supervised **training, calibration, and challenger
selection** are ineligible. It does not prohibit inference with an already approved and
compatible supervised recipe.

### 4. Stage the task profile

The advisor uses three nested aggregate profiles:

1. `PreflightTaskProfile` — configuration and privacy-safe aggregate structure available before
   row-level execution;
2. `CandidateGraphProfile` — aggregate candidate graph evidence available after retrieval;
3. `EvidenceProfile` — aggregate score, disagreement, calibration, review, and assignment
   evidence available after diagnostic execution.

The preflight profile does not fabricate row counts, missingness, uniqueness, candidate-set
sizes, or model evidence that have not yet been observed.

### 5. Implement Stage 1 now

Stage 1 provides:

```text
hard eligibility rules
runtime and governance compatibility
mandatory Fellegi-Sunter baseline retention
structural Pareto analysis
family-diverse bounded shortlist
transparent rule explanations
explicit abstention from empirical ranking
```

Without benchmark evidence, Stage 1 may say which strategies are structurally eligible. It may
not claim that one model will have higher sensitivity, PPV, calibration, or accuracy.

### 6. Separate evidence classes

Advisor outputs distinguish:

```text
global synthetic benchmark evidence
local schema-matched synthetic evidence
local verified validation evidence
local operational monitoring evidence
```

Local synthetic evidence remains synthetic. Local verified evidence does not receive authority
merely because it is local; it must be eligible, current, compatible, independent, protocol
approved, and sufficient for the claim.

### 7. Defer data-driven advisor stages

Nearest-neighbour retrieval, out-of-distribution scoring, learned meta-ranking, and active
benchmark planning are deferred until the benchmark registry satisfies prospective readiness
and held-out-family validation gates.

## Consequences

### Positive

- redundant full-portfolio execution can eventually be reduced;
- the mandatory interpretable baseline remains visible;
- unsupported or weak-evidence jobs cause abstention;
- recommendations expose the evidence that supports them;
- incoming-data inference remains separate from training and model selection;
- no advisor component can acquire identity, assignment, or merge authority.

### Costs

- profiling, benchmark, recommendation, and recipe schemas require versioning;
- benchmark corpus design becomes a substantial scientific milestone;
- Stage 1 may return several eligible strategies without an empirical winner;
- recommendation quality cannot advance faster than benchmark coverage and local validation.

## Rejected alternatives

### Run every model on every incoming dataset

Rejected because incoming data often lack verified truth, repeated selection would be
statistically invalid, and operational inference should load an approved recipe.

### Let the advisor approve the operational champion

Rejected because synthetic similarity cannot replace local candidate recall, calibration,
threshold selection, locked testing, and governance approval.

### Train a meta-ranker before building the corpus

Rejected because it would overfit a small and correlated benchmark and would not establish
generalisation to unseen scenario families.

### Use latent synthetic-generator parameters as advisor inputs

Rejected because those values are unavailable for a real linkage job. Learned advisor inputs
must be observable at recommendation time.

## Validation

This ADR is satisfied for I2A when:

- authority fields cannot be overridden;
- hard eligibility violations cannot be outweighed by utility;
- supervised development is rejected without eligible labels;
- approved inference is distinguished from supervised training;
- the structural shortlist retains the mandatory baseline;
- Stage 1 reports no empirical performance claim;
- profile and recommendation outputs contain no record values, candidate pairs, source columns,
  or local paths;
- CLI commands are deterministic and synthetic/privacy safe.
