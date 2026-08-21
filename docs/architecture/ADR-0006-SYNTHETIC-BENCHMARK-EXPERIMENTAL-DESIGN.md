# ADR-0006: Synthetic Benchmark Experimental Design

**Status:** Accepted  
**Date:** 2026-08-19

## Context

A learned strategy advisor requires a designed and reproducible corpus of linkage tasks and
pipeline results. Opportunistically accumulated synthetic runs are correlated, selectively
reported, and unsuitable for strong generalisation claims.

The benchmark library is therefore the future advisor's evidence corpus and must be governed as
an experiment rather than treated as test-fixture output.

## Decision

### 1. Fix the experimental units

```text
scenario family
    scientifically coherent corruption or linkage regime

scenario instance
    one parameterised point within a family

replicate
    a new seed or population realisation of the same instance

benchmark run
    scenario instance × pipeline recipe × seed × software/environment version
```

Multiple seeds are replicates, not independent scenario families.

### 2. Separate observable features from latent simulator parameters

The registry retains both digests, but only observable profile features may enter an advisor
feature vector. Latent parameters such as the true typo rate or true no-match prevalence may be
used for stratification, mechanism holdout, and simulator-specific contrasts; they may not be
hidden inputs to a model intended for real jobs.

### 3. Use a mixed experimental design

The benchmark programme combines:

1. factorial or fractional-factorial cells for main effects and selected interactions;
2. space-filling coverage for continuous parameter ranges;
3. mechanism-focused stress families;
4. composite realistic families with interacting corruption processes;
5. prospectively held-out mechanisms for out-of-distribution evaluation.

One-factor-at-a-time experiments remain useful for interpretation but are insufficient because
linkage performance depends on interactions such as missingness by frequency skew, candidate-set
size by ranking method, and no-match prevalence by assignment constraint.

### 4. Retain failures as evidence

The aggregate registry retains:

```text
success
failed_fit
timeout
memory_failure
ineligible
abstained
numerical_failure
candidate_budget_failure
```

Dropping failures would bias the evidence toward expensive or brittle pipelines that happened to
complete.

### 5. Split advisor evaluation by scenario family

Replicates and instances from one family cannot be split across advisor training and evaluation
in a way that leaks the same mechanism. Prospectively designated corruption families remain
held out from model and distance-policy tuning.

### 6. Treat family-count ranges as planning aids only

Approximate ranges such as 50–100 families for similarity retrieval and 200–500 for
meta-learning are not encoded as automatic validity thresholds. Readiness depends on profile
coverage, recipe overlap, held-out mechanisms, uncertainty, learning curves, and advisor regret.

### 7. Limit claims to the simulator

Randomised synthetic experiments support simulation-based causal contrasts inside the declared
generator. They do not establish that the same effect holds in a real population or system.

### 8. Bind advisor-v2 prospectively

The first advisor-scale catalogue is `advisor_v2`. It contains 64 scientifically coherent
families and 280 parameterised instances. The family roles are fixed before execution:

```text
40 meta-training families
 8 conformal families
 8 locked-evaluation families
 8 true-mechanism OOD families
```

Families represent main-effect, selected-interaction, composite, stress, or real-mechanism
regimes; parameter rows are instances and are not relabelled as families to increase a count.
The existing seed-v1 catalogue remains exactly 10 families and 19 instances with stable IDs and
digests. Its historical held-out transliteration family is a typo/transposition proxy, so it is
excluded from true OOD readiness. Advisor-v2 uses a versioned cross-script transliteration and
punctuation mechanic for its OOD families.

Sixty-four families are adequate to start the prospectively partitioned programme, but do not
satisfy the higher planning range for strong meta-learning claims by count alone. Readiness still
requires coverage, learning curves, regret, uncertainty calibration, and locked-family evidence.

### 9. Protect truth and comparative metrics

Synthetic truth may enter only protected supervised-training labels and post-score mechanical
evaluation. It cannot construct retrieval, comparison, scoring, ranking, calibration,
assignment, or relationship-decision inputs. The benchmark adapters call the package-owned
Fellegi-Sunter reference, XGBoost classifier, and XGBoost ranker. Dedupe-only, multi-source,
LightGBM, and PyTorch recipes remain ineligible with stable codes until exact truth-safe adapters
are integrated; no placeholder metrics are emitted.

### 10. Require approved append-only heavy execution

The design digest binds a deterministic balanced shard plan. Execution requires an explicit
non-identifying human approval reference bound to both digests. Run evidence is persisted
append-only with commit-marker ordering, exact idempotent resume, dependency/environment
provenance checks, collision rejection, and tamper detection. The executor emits aggregate
summaries only. Large-corpus execution is outside ordinary CI and must use the repository's one
long-run driver.

## Consequences

The benchmark corpus becomes a versioned scientific product with prospective design, failure
retention, environment provenance, and family-level validation. It requires substantial compute
and cannot be compressed into ordinary CI. CI verifies contracts and small benchmark smoke tests;
large corpus production uses an explicitly approved benchmark execution plan.

Acceptance of this ADR establishes the design and execution controls only. The advisor-v2 heavy
corpus is not complete until every approved shard and replicate has retained evidence and the
aggregate completion audit passes. Evidence readiness is stricter than file completion: every
approved scenario-replicate cell must contain successful evidence from all required adapters under
one engine provenance. Family-level overlap cannot hide failed cells.

The first execution-protocol-v1 run completed its file grid but exposed 688 Fellegi-Sunter
materialisation failures. It is retained unchanged as diagnostic evidence. Execution protocol v2
uses stable base-2 logistic scoring, explicit double-precision evidence constants, a new registry
provenance boundary, and a cell-complete readiness contract before Stage-3 fitting.

## Validation

Before Stage 2, the benchmark programme must report:

```text
family, instance, replicate, and run counts
recipe-by-family coverage
pairwise recipe comparison counts
profile-space coverage
held-out mechanism count
failure and abstention rates
nearest-neighbour density
learning-curve stability
```
