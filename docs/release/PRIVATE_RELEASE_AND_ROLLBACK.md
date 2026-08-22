# Private Release and Rollback

## Boundary

This runbook packages a private candidate for an authorised environment. It does not publish to a
package index, change repository visibility, select a licence, deploy an operational service, or
establish operational validity. Those actions require separate owner approval.

## Candidate preparation

1. Require a clean, reviewed commit with `quality` and `all-models` checks passing.
2. Run `python scripts/verify_release_readiness.py --expect-blocked` and confirm every blocker is
   truthful; do not mislabel a blocked Phase 1 candidate as releasable.
3. Run the synthetic scale benchmark separately and review aggregate runtime/memory evidence.
4. Run `python scripts/build_local_handoff.py --project-root .` in the exact Python 3.12 envelope.
5. Review dependency audit, SBOM, distribution inspection, checksums, and the aggregate handoff
   manifest under ignored local directories.
6. Record the approved Git commit, constraints digest, wheel digest, source-distribution digest,
   reviewer, destination environment, and expiry/rollback decision outside the repository.

## Transfer prohibition

Never place real data, completed configurations, labels, adjudications, candidate pairs, models,
linkage outputs, credentials, or restricted operational logs in a candidate bundle. A release
candidate contains package source/distributions and aggregate software evidence only.

## Rollback

1. Stop new executions through the authorised operational change-control mechanism.
2. Select the previously approved immutable wheel/source bundle by retained checksum; do not rebuild
   it from a mutable branch.
3. Restore the corresponding constraints, configuration digest, artifact set, and package version
   inside the restricted environment without overwriting the failed candidate evidence.
4. Run the aggregate environment doctor and generated-synthetic smoke test.
5. Verify retained artifact/configuration digests and confirm decision/assignment authority remains
   unchanged.
6. Record the rollback reason, affected candidate digest, restored digest, verification evidence,
   and human approval outside the repository.

The bounded synthetic drill was completed on 2026-08-22 against immutable candidate and baseline
snapshots. Its aggregate evidence is bound in
[`ROLLBACK_DRILL_EVIDENCE.md`](ROLLBACK_DRILL_EVIDENCE.md), so
`rollback_drill_not_completed` is closed. This does not establish operational rollback readiness;
an authorised deployment environment must retain and exercise its own change-control, backup,
restore, access, and approval procedures before operational use.
