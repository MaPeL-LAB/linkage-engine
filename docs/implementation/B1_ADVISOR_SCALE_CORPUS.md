# B1 Advisor-Scale Synthetic Corpus

## Delivered boundary

This increment delivers a versioned experimental design and an approved, resumable execution
route. The first execution-protocol-v1 corpus is retained as diagnostic evidence. The corrected
execution-v2 corpus is complete and supplied the exact evidence grid for the separately governed
advisor qualification.

```text
seed_v1:     10 families / 19 instances (IDs and digests unchanged)
advisor_v2:  64 families / 280 instances

advisor_v2 prospective roles:
  meta-training       40 families / 160 instances
  conformal            8 families /  32 instances
  locked evaluation    8 families /  40 instances
  OOD holdout          8 families /  48 instances
```

Families encode coherent main-effect, interaction, composite, stress, or mechanism regimes.
Instances are parameter points inside those regimes. Replicates are deterministic new
realisations; neither instances nor replicates are counted as additional families.

## Truth and authority boundaries

Synthetic truth is allowed only for protected supervised-training labels and post-score
mechanical evaluation. Candidate generation, comparison features, model scoring, ranking,
calibration, assignment, and relationship decisions cannot receive truth fields. Evaluation
labels are stripped before classifier or ranker scoring.

The benchmark runner uses existing package implementations for:

- DuckDB Fellegi-Sunter reference matching;
- XGBoost pair classification;
- XGBoost candidate ranking.

LightGBM, PyTorch, dedupe-only, and multi-source benchmark entries remain ineligible until exact
truth-safe package adapters exist. Failures and ineligibility are retained. Metrics are never
fabricated to fill a portfolio cell.

Every design, approval, shard, completion, run, and advisor summary retains:

```text
recommendation_authority = advisory_only
decision_authority       = none
assignment_authority     = none
merge_authority          = none
automatic_promotion      = prohibited
operational_validity     = not_established
```

## Advisor partitioning

The family-role manifest is prospective and digest-bound. The meta-ranker requires successful
evidence for all three real adapters in every approved scenario-replicate cell across the intended
training, conformal, locked, and OOD families. Training and conformal families must be disjoint.
Conformal residuals are computed only after fitting on training families. Locked families are evaluated
only after model and interval fitting and cannot tune either. OOD families never enter fitting or
interval calibration.

The legacy `family.held_out_transliteration` is retained for seed-v1 compatibility but excluded
from OOD readiness because its mechanics are only typo and token transposition. Advisor-v2 OOD
families use a versioned deterministic cross-script transliteration and punctuation mechanic.

## Integrity and privacy controls

- The design digest binds the exact 64 families, 280 instances, and prospective roles.
- A deterministic balanced shard plan covers each advisor-v2 instance exactly once.
- Execution requires literal human approval bound to the design and shard-plan digests.
- The registry path must remain project-relative and cannot traverse symbolic links.
- Aggregate artifacts use bounded reads and canonical atomic writes.
- Run metrics or failure evidence are written before the run record, which is the commit marker.
- Exact reruns resume idempotently; conflicting IDs, tamper, or stale environment provenance are
  rejected without overwrite.
- Public summaries contain counts and digests only, never paths, approval references, record
  values, identifiers, candidate pairs, labels, or score vectors.

## Execution handoff

Inspecting the aggregate plan is a quick command:

```text
mapel-linkage plan-advisor-corpus --shards 32 --replicates 5
```

The heavy run must be launched outside Codex through the single repository driver after a human
has reviewed the design digest, shard-plan digest, adapter gaps, compute requirements, and storage
location:

```bash
CORPUS_APPROVAL_REF="replace-with-approved-non-identifying-reference"
scripts/run_advisor_corpus.sh \
  --full \
  --approve-execution \
  --approval-reference "${CORPUS_APPROVAL_REF}"
```

The execution-v2 default registry is under ignored
`private/benchmark_registry/advisor_v2_execution_v2`. The completed diagnostic v1 registry remains
unchanged under its original path. The driver is safe to rerun after interruption. It never
commits, pushes, deletes, resets, or promotes a model.

## Diagnostic v1 result and v2 correction

The 2026-08-21 diagnostic run retained all 9,800 expected records: 3,512 successes, 5,600 expected
ineligible records, and 688 Fellegi-Sunter score-materialisation failures. The failures were caused
by fixed-precision SQL inference for large finite evidence constants and numerically unstable
probability materialisation. Execution v2 casts evidence constants explicitly to double precision,
uses a stable two-tail base-2 logistic expression, and verifies finite bounded aggregate scores.
The retained aggregate diagnostic audit digest is
`ac65c7c326e54d01852a91b3aebb7c4058096f6ebfd9cfde5723bb2baba766db`.

The earlier family-overlap gate is superseded. Readiness v2 requires at least five replicates per
instance, one consistent engine provenance, and successful evidence from all three required
adapters in all 1,400 cells. Failed or missing cells force Stage-2 similarity fallback.

## Corrected v2 completion and qualification handoff

Execution v2 completed on 2026-08-21 with all 9,800 records retained: 4,200 required adapter
successes, 5,600 expected ineligible records, no required adapter failures, and successful
three-adapter evidence in every one of the 1,400 scenario-replicate cells. The aggregate readiness
digest is `4c91c1099c15f226ddded933a3fb5462e23f5ecf8c44914eac87944882d84e76`.

This establishes corpus completeness, not advisor quality. The prospective qualification is
documented separately in
[`I2_ADVISOR_EMPIRICAL_QUALIFICATION.md`](I2_ADVISOR_EMPIRICAL_QUALIFICATION.md); it returned
`not_qualified` and requires similarity fallback. Synthetic evidence cannot establish population
fidelity, fairness, real-world calibration, operational sensitivity or positive predictive value,
or safety for any production linkage task. A human still owns all new experimental-design,
release, and operational-use decisions.
