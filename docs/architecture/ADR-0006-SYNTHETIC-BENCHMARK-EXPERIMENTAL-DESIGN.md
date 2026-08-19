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

## Consequences

The benchmark corpus becomes a versioned scientific product with prospective design, failure
retention, environment provenance, and family-level validation. It requires substantial compute
and cannot be compressed into ordinary CI. CI verifies contracts and small benchmark smoke tests;
large corpus production uses an explicitly approved benchmark execution plan.

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
