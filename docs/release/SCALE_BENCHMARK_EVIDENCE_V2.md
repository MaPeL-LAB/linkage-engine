# Synthetic Scale Benchmark Evidence V2

## Review status

The aggregate v2 evidence was explicitly approved by the repository owner on 2026-08-22 for the
bounded development envelope described below. This approval closes only
`scale_evidence_not_completed`. It does not authorize release, publication, deployment, migration,
real-data execution, model promotion, or operational use.

## Immutable evidence binding

| Field | Reviewed value |
|---|---|
| benchmark | `m8_complete_synthetic_scale_v2` |
| plan digest | `442d59215bf6572979bb96ce1b3881c88b7974e627bd731a083d20b2eb05a48d` |
| summary digest | `4d4c015b1f1e289c57516a76b6b3730d277a761e2310b53be1f76fab651f7465` |
| configuration digest | `9c4b3b630316cb6802aaddcd61e9bb712184274aec06988981c9d5bb71f3eb06` |
| implementation digest | `29efe1b55ca83d12063da5520c776dc0c0ad70319a6faf32d8e1816b5b8e098c` |
| seed | `20260816` |
| workers | 10 |
| matrix | 100, 200, 300, 400, and 500 entities; two repetitions each |
| completed cases | 10 of 10 |
| resume check | 10 resumed; 0 newly executed; summary digest unchanged |

The summary digest binds the ordered case-report digests. Detailed case reports remain in the
ignored aggregate-synthetic artifact directory and are not repository content.

## Aggregate observations

| Measure | Reviewed value |
|---|---:|
| median elapsed time | 11.448434 seconds |
| maximum elapsed time | 18.661615 seconds |
| maximum resident set size | 753,040 KiB |

Every retained report asserts `contains_record_data=false`, `contains_identifiers=false`,
`contains_candidate_pairs=false`, `contains_local_paths=false`, and
`operational_validity=not_established`.

## Superseded proposal

The v1 proposal included 1,000- and 2,000-entity cases. Both sizes deterministically crossed the
package-owned 100,000-pair candidate budget and failed closed with `ML-CANDIDATE-008`. The budget
was not increased. V2 instead limits the reviewed envelope to at most 500 entities and uses a new
benchmark identifier, output directory, plan digest, and evidence chain. V1 diagnostic case reports
cannot satisfy or be mixed with v2 evidence.

## Interpretation boundary

This evidence establishes bounded generated-synthetic runtime and memory behaviour on the reviewed
local software envelope only. It does not establish population accuracy, linkage validity,
fairness, calibration, real-system capacity, cross-platform compatibility, privacy guarantees for
operational data, or fitness for deployment.
