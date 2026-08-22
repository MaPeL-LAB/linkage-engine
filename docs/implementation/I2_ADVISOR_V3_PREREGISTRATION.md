# I2 Advisor-v3 Prospective Preregistration

## Decision and boundary

Advisor-v3 is preregistered as a new synthetic-only experiment. It does not revise, overwrite,
or reinterpret advisor-v2 evidence. No advisor-v2 locked-evaluation or OOD family is reused as a
v3 unit. The canonical outcome-free binding is
[`advisor_v3_preregistration_20260821.json`](../evidence/advisor_v3_preregistration_20260821.json).

The design contains 84 new families and 336 instances, with exactly four instances per family:

The corpus has since completed under this immutable v3 protocol. Its original all-role evidence
readiness gate failed closed. The separately disclosed post-corpus, pre-qualification remediation is
documented in
[`I2_ADVISOR_V31_REMEDIATION.md`](I2_ADVISOR_V31_REMEDIATION.md); it does not rewrite this
preregistration or authorize qualification.

| Role | Families | Purpose |
|---|---:|---|
| Meta-training | 48 | Fit Stage 2 utility summaries and the Stage 3 ridge model |
| Conformal | 12 | Calibrate Stage 3 intervals and the OOD distance threshold |
| Locked evaluation | 12 | Estimate held-out advisory utility after separate approval |
| OOD holdout | 12 | Evaluate distance-based abstention after separate approval |

Family is the statistical unit. Five deterministic replicates and seven portfolio recipes produce
11,760 retained run records. The three required truth-safe adapters contribute 5,040 expected
successes; four currently ineligible recipes contribute 6,720 retained ineligibility records.

## Target and interpretation

The target is family-average synthetic benchmark utility and the corresponding family-level regret
of a recommended recipe relative to the best of the three required adapters. Stage 2 estimates
utility by the unweighted mean over the three nearest meta-training families. Stage 3 fits a ridge
model over the fixed mechanism-feature order and recipe one-hot columns, with split-conformal
intervals calibrated only on conformal families.

This target measures task utility inside the preregistered synthetic design. It is not population
validity, real-data validation, privacy proof, operational safety, relationship-decision authority,
assignment authority, or merge authority.

## Feature-source parity and abstention

Advisor-v3 does not use latent simulator error rates as model features. Its versioned profile is
computed from observable generated aggregates: script variation, punctuation/tokenization,
missingness level and asymmetry, frequency concentration, exact candidate-graph density,
within-dataset duplicate signatures, and the planned supervised training-label budget.

The repository does not yet implement a governed runtime-data producer for this profile. Missing
runtime evidence is represented as unavailable, never silently imputed to zero, and must cause
abstention or the existing v2 fallback. Operational validity therefore remains `not_established`.

## Training-label budget and adapter fairness

The label-budget mechanic is causal only for supervised adapters. A deterministic, class-stratified
budget is applied to the protected training partition for the XGBoost classifier and ranker. The
Fellegi–Sunter adapter retains the full protected training feature surface because its EM evidence
is unsupervised. Validation, calibration, decision, and test partitions are not changed. Both the
full Fellegi–Sunter surface and supervised retained-label authority are digest-bound.

## OOD geometry

The OOD threshold is not a reused numeric threshold from v2. It is the fixed finite-sample 90th
percentile rule over conformal-family nearest-training distances. Locked and OOD families are
prohibited threshold inputs. Across all five preregistered profiling seeds, the outcome-free design
geometry has zero of 12 locked families and all 12 OOD families above the selected threshold. This
is a covariate-coherence check, not evidence that the performance gates will pass.

## Immutable qualification rules

The evaluator specification binds equal-weight family/recipe aggregation, exact regret and oracle
tie formulas, the best-fixed-baseline rule, Stage 2 neighbour ordering, and the raw continuous plus
recipe-one-hot Stage 3 matrix and centered ridge solution. Conformal calibration uses 12 family
units: each score is the maximum absolute residual over the three recipes, and its finite-sample
rank is 12. Leave-one-training-family-out stability is the agreement fraction over exactly
48 omissions by 12 locked targets (576 comparisons); the omitted family leaves only the fit or
neighbour pool, while all conformal families remain fixed. OOD abstention uses strict `>` so equality
does not abstain. Learning-curve fits use nested 12/24/36/48 prefixes ordered by
`sha256(policy_digest:family_id)` and evaluate the same 12 locked families.

The digest also binds every gate conjunction, numeric canonicalization, qualification approval
schema `advisor_v3_locked_qualification_approval_v1`, and aggregate report schema
`advisor_v3_qualification_aggregate_report_v1`. The executable qualification evaluator is
intentionally unavailable in this pre-corpus package. It requires a separate reviewed post-corpus
implementation slice and explicit approval authorizing locked and OOD evaluation. Corpus execution
alone can neither qualify nor promote the advisor.

That later implementation slice now exists under the disclosed v3.1 evidence amendment. This
historical preregistration statement and its digest remain unchanged: implementation availability
does not authorize execution, change a threshold, or convert v3.1 into a prospective protocol.

Performance gates were not relaxed after the advisor-v2 failure:

- Stage 2 regret improvement over the best fixed recipe at least 0.005;
- Stage 3 regret improvement over Stage 2 at least 0.01;
- top-2 oracle coverage at least 0.875 and selection stability at least 0.80;
- locked interval coverage at least 0.80 and mean width at most 0.50;
- OOD detection at least 0.75 and locked false abstention at most 0.125; and
- final learning-curve tail regret range at most 0.02, with no degradation from the first point.

Any missing cell, failed required adapter, provenance conflict, incomplete family, schema drift,
catalogue digest drift, noncanonical preregistration, or unavailable feature causes deterministic
failure or fallback. Qualification cannot run as part of corpus execution and requires a later,
explicit approval bound to the completed registry snapshot.

The execution approval separately binds the exact initial readiness digest and a canonical
execution-provenance digest over the package source digest, dependency-lock digest, and environment
digest. Preparation, every shard worker, and audit recompute both bindings and fail closed on any
source, dependency, environment, or adapter-readiness drift. Aggregate summaries expose only the
combined digests, never the component provenance values. This binding is deliberately downstream
of preregistration and readiness, avoiding a circular readiness/approval contract.

## Execution handoff

The long corpus run is intentionally external to Codex:

```bash
scripts/run_advisor_v3_corpus.sh --dry-run
scripts/run_advisor_v3_corpus.sh \
  --full \
  --approve-execution \
  --approval-reference '<NON_IDENTIFYING_REFERENCE>'
```

The full mode prepares shared governance serially, then runs 42 whole-family shards with ten workers
by default. Workers use disjoint artifact names and an exclusive per-shard process lock. A stopped
run is append-only and resumable. Audit parses held-out metric artifacts only for digest integrity;
it does not use held-out values for design, fitting, threshold selection, or qualification.

## Rejected alternatives

- Reusing revealed v2 locked or OOD units was rejected because it would not be prospective.
- Adding v3 fields to `TaskMetaFeatureVector` was rejected because it would change v2 contracts.
- Latent-rate features were rejected because target-side tasks cannot supply the same inputs.
- A fixed OOD distance of 0.45 was rejected because its meaning changes with feature geometry.
- Applying sparse labels to Fellegi–Sunter EM was rejected because it would bias adapter fairness.
- Parallel instance shards were rejected because workers could race on shared family governance.
- Automatically qualifying or promoting the advisor after corpus completion remains prohibited.
