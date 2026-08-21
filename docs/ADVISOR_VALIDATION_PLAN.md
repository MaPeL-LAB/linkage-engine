# Linkage Strategy Advisor Validation Plan

## Stage 1 — transparent rules

Exit criteria:

- deterministic task-profile digest and recommendation digest;
- zero hard-constraint violations;
- mandatory Fellegi-Sunter baseline retained where eligible;
- supervised development excluded without eligible verified labels;
- approved inference distinguished from training;
- optional dependency failures are explicit;
- stacking requires protected out-of-fold evidence;
- no empirical performance claim without benchmark evidence;
- explicit abstention and local confirmation requirements;
- no rows, identifiers, source fields, paths, candidate pairs, or score vectors in output;
- no identity, assignment, merge, or automatic-promotion authority.

## Stage 2 — similarity retrieval

Stage 2 may begin only after a prospectively designed benchmark registry has adequate profile
coverage. Evaluation is grouped by held-out scenario family and reports:

```text
top-K oracle-recipe coverage
regret relative to the best eligible recipe
coverage and nearest-neighbour density
out-of-distribution detection
abstention accuracy
recommendation stability
performance against always-FS and always-XGBoost baselines
```

## Stage 3 — learned meta-ranking

Stage 3 requires independent Stage-2 validation, sufficient recipe-by-family overlap, and stable
learning curves. Advisor-v2 fixes 40 training, 8 conformal, and 8 locked-evaluation families
before any fit. No family, instance, or replicate may cross those roles. Conformal residuals are
computed only on the conformal families, locked families are evaluated only after fitting, and
the 8 true-mechanism OOD families never enter fitting or interval calibration. The learned ranker
must improve meaningfully over transparent nearest-neighbour retrieval while preserving zero
hard-constraint violations. Every approved scenario-instance and replicate cell must contain
successful evidence from all three required adapters. A failed or missing cell, fewer than five
replicates per instance, mixed engine provenance, or constant metric evidence requires similarity
fallback, not a learned recommendation. Family-level overlap alone is insufficient because it can
exclude difficult failed replicates and create survivor bias.

## Stage 4 — active benchmark planning

Stage 4 requires calibrated uncertainty. Proposed experiments are evaluated prospectively by
whether they reduce coverage gaps, uncertainty, or regret without concentrating only on already
strong pipelines. A fixed core benchmark remains untouched.

Heavy execution also requires the versioned design and shard digests, all three truth-safe core
adapters to be success-capable, and explicit human approval. The completed execution-protocol-v1
registry is retained as diagnostic evidence after exposing 688 Fellegi-Sunter failures; only a
separate execution-v2 registry with a complete three-adapter cell grid can qualify meta-ranking
evidence.

## Evidence language

Synthetic recommendation evidence is a prior. It cannot establish operational validity or
replace local candidate recall, model selection, calibration, threshold selection, locked test
evaluation, and approval.
