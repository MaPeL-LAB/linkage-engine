# Synthetic Rollback Drill Evidence

## Decision

The repository owner explicitly approved this bounded drill on 2026-08-22. The final execution
completed every prespecified check and closed only `rollback_drill_not_completed`. Release remains
blocked by five independent gates, and this evidence does not authorize release, publication,
deployment, real-data execution, or model promotion.

## Immutable scope

| Evidence item | Bound value |
|---|---|
| Drill ID | `m8_synthetic_rollback_v1` |
| Candidate commit | `81762675996eae77ccb16210936d630f092a3e7b` |
| Restored baseline commit | `5050626583236fe1a7778eabc363a31764385285` |
| Candidate wheel SHA-256 | `974e76df6f6f4783eeec0f9f4f499f565417cdd8aaea8d2d653ca483f024a926` |
| Baseline wheel SHA-256 | `492c9c30c76059e316e9d477792ddef796669024483bc8b368e2d0a2dbd9b475` |
| Candidate package-tree digest | `ecbc7fa8cddea0de30f0caf33f19a3a82be598b6fd11ba3d2dc360c0a953e221` |
| Restored package-tree digest | `b106ede07796592fa8c799400d25e42ab31ef801a8995b32ef5b10db566cbf5c` |
| Constraints digest | `a527f3013c3e076e804f757e17b6d3c64eaf4a514c8706ac17ea38e83f017423` |
| Synthetic configuration digest | `9c4b3b630316cb6802aaddcd61e9bb712184274aec06988981c9d5bb71f3eb06` |
| Authority-contract digest | `4de937aca72fc9e4275ee8cb1ab03979948bc1c3c159eefdc1f1e70abe5c6f4c` |
| Dependency-environment digest | `711ee5cc4b885cab1d997074c0ca17b9ef1f8f69b041f0e484a8c2bcb5661508` |
| Drill implementation digest | `248c80f27ac61b0c205d1b25574db391f955415c64a865b3494ca6c26686e15d` |
| Canonical report-body digest | `fd664bebd3d8e8d328812ea2dffaee80894838d112d4d7dcd0b2d9f389771f89` |
| Canonical summary-file digest | `0515cd83ec9b9e9abd91ebe7cb660c6e6d788ca1e7642bc65a7cf46927ab0763` |

The local ignored evidence directory retains both wheels separately and the canonical aggregate
summary. It contains no record data, identifiers, candidate pairs, or local filesystem paths.

## Exercised sequence

1. The harness resolved both full commit digests and proved the baseline is an ancestor of the
   candidate.
2. It built reproducible wheels from Git archives rather than a mutable working tree.
3. It created an isolated virtual-environment package surface, reused the exact verified Python
   3.12 dependency layer bound by digest, and proved the imported package came from that surface.
4. It installed and verified the 140-file candidate tree, ran the aggregate environment doctor,
   and completed a deterministic 100-entity generated-synthetic smoke test with seed `20260816`.
5. An approved synthetic drill trigger initiated rollback without overwriting candidate evidence.
6. The harness force-reinstalled the retained baseline wheel, proved the one candidate-only file
   was removed, and exactly matched the 139-file baseline tree and package version.
7. The restored environment passed the same doctor and generated-synthetic smoke test. Both
   retained wheel digests were then rechecked.

All 13 checks passed. Candidate and baseline doctor-output digests matched, and their safe
synthetic-smoke output digests matched. Decision, assignment, recommendation, and merge authority
contracts were unchanged across the two immutable snapshots.

## Fail-closed attempt record

The first attempt stopped before writing evidence because the macOS temporary-directory spelling
traversed a system symlink and the engine rejected the managed path. The harness was corrected to
use the resolved physical temporary root, focused tests were rerun, and only then was the final
drill executed. This was a harness-path failure, not a candidate or restore failure.

Before commit, integrity review added exact dependency-pin validation and mandatory retained-wheel
rehashing on idempotent replay. The earlier passing evidence directory was preserved locally under
a superseded name, and the complete drill was rerun with the hardened implementation. The digests
in this document bind only the final hardened run.

## Boundary and limitation

The package installation surface was isolated, but the dependency layer was deliberately reused
read-only from the verified Python 3.12 environment and bound by digest. Dependency packages were
not independently reinstalled. The candidate and baseline used identical constraint files, so no
dependency change was being rolled back. This drill establishes bounded synthetic restore
behaviour only. It records `operational_validity=not_established`, all publication/deployment/
release authorities as `none`, and automatic publication as `prohibited`.
