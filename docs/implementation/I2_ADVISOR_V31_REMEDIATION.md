# I2 Advisor-v3.1 Role-Specific Evidence Remediation

## Decision status

Advisor-v3.1 is a post-corpus, pre-qualification protocol amendment. It is not represented as a
prospective corpus preregistration. The amendment was selected after aggregate adapter-status and
stable failure-code metadata showed that the completed v3 grid contained required-adapter failures
confined to the OOD role. No performance metric value was inspected or used to select the
amendment; the amendment is explicitly performance-metric-blind. The canonical
source-controlled binding is
[`advisor_v31_protocol_amendment_20260821.json`](../evidence/advisor_v31_protocol_amendment_20260821.json).

The immutable advisor-v3 registry remains the source evidence. Advisor-v3.1 writes only amendment,
approval, and readiness manifests to a distinct ignored governance-only registry. It does not copy,
rewrite, repair, delete, or append source benchmark records.

## Role-specific evidence contract

The v3 qualification algorithm uses recipe utility for meta-training, conformal calibration, and
locked evaluation. These 72 family-level units therefore continue to require all three adapters for
all four instances and five replicates: 1,440 complete cells and 4,320 successful adapter runs. Any
missing or non-success cell in these roles fails closed.

OOD qualification is distance-based abstention. It uses the 12 complete observable mechanism
profiles and the preregistered training/conformal distance geometry. OOD recipe utilities are not an
input to model fitting, conformal calibration, threshold selection, or OOD qualification. Their
adapter statuses remain aggregate diagnostic integrity evidence only; OOD metric use is prohibited.
This exclusion applies to the entire OOD role and every adapter status, not only to the observed
failure code, avoiding failure-specific metric fabrication or selective replacement.

Family remains the statistical unit. The catalogue, roles, seeds, replicates, utility function,
distance threshold, performance gates, and authority boundaries are unchanged.

## Why another heavy corpus is not run

A second 10-worker corpus would repeat the same family catalogue, seeds, label budgets, and adapter
conditions. It would not add a qualification input because OOD recipe utilities are prohibited under
the amended contract. Running after source changes would also create a different engine provenance
and an avoidable duplicate experiment rather than repairing the frozen v3 evidence.

The scientifically relevant check is therefore an integrity bridge over the completed source, not a
rerun. The bridge validates all 11,760 run and sidecar artifacts, rejects orphan, extra, non-JSON,
or symbolic-link artifacts, validates 42 whole-family completion manifests, the persisted v3
approval, the original execution-provenance tuple, the terminal v3 readiness digest, the exact
family/instance catalogue, and the resulting registry snapshot. It also recomputes the observable
mechanism-profile geometry and binds the coherence digest. Current v3.1 source, dependency lock,
and environment are bound as a separate combined analysis-provenance digest and are never
substituted for the original engine provenance.

## Post-hoc boundary and provenance limitation

The amendment was motivated by role-scoped adapter-status and failure-code metadata observed after
corpus execution. Future reports must identify the evidence contract as advisor-v3.1 and must not
describe it as the original v3 preregistration. The amendment cannot relax a required-role failure,
alter a performance threshold, or authorize qualification.

The source snapshot receives its first explicit digest binding in the v3.1 governance bridge. The
earlier v3 approval, completion, sidecar, provenance, and terminal-readiness bindings provide local
tamper evidence, but no external signature or timestamp authority was preregistered. This residual
integrity limitation must remain disclosed in qualification review.

## Fail-closed handoff

The remediation audit is a bounded local integrity pass, not a heavy benchmark run:

```bash
mapel-linkage audit-advisor-v31-remediation \
  --project-root . \
  --source-registry-dir private/benchmark_registry/advisor_v3_execution_v1 \
  --remediation-registry-dir private/benchmark_registry/advisor_v31_remediation_v1 \
  --amendment docs/evidence/advisor_v31_protocol_amendment_20260821.json \
  --approve-remediation \
  --approval-reference '<NON_IDENTIFYING_REFERENCE>'
```

The command fails closed on current-source substitution, mixed source provenance, missing source
records, sidecar tampering, catalogue drift, incomplete whole-family completions, path traversal,
symbolic links, source/destination aliasing, non-governance destination content, or conflicting
idempotent output.

Even a ready v3.1 bridge does not run or authorize locked/OOD qualification, qualify an advisor,
promote a model, release an artifact, decide a relationship, assign an edge, or merge records. An
executable evaluator requires a separate reviewed implementation slice and a new human approval
bound to the v3.1 amendment, source snapshot, readiness digest, and then-current evaluator source.
