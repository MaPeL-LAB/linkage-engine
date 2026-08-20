# I1C Configuration-Driven Linkage Modes

## Status

Integrated with core-CI coverage for bounded generated-synthetic software behaviour; CI
evidence is assessed per commit. Operational validity, real-data approval, arbitrary mode
dispatch, release readiness, and merge authority are not established.

## Exact dispatch boundary

`SyntheticModeWorkflowRunner` and `mapel-linkage run-linkage-mode` accept only:

| Linkage mode | Assignment constraint | Output authority |
|---|---|---|
| `link_only` | `many_to_one` | decision policy may emit relationship status |
| `link_only` | `one_to_many` | decision policy may emit relationship status |
| `link_only` | `unconstrained` | decision policy may emit relationship status |
| `dedupe_only` | `unconstrained` | aggregate cluster evidence only |
| `link_and_dedupe` | `one_to_one` | aggregate cross-source assignment and same-source cluster evidence only |

When `mode_orchestration` is present, configuration validation accepts only these five exact
combinations. Project shapes without that opt-in retain the broader base-schema vocabulary,
but `run-linkage-mode` rejects them before execution because they have no allow-listed dispatch
key. The runner additionally requires seed `20260816`, an enabled XGBoost pair model,
protected calibration bound to that pair model, and the exact package-owned implementation,
artifact schema, assignment, and deduplication allow-list entries.

## Evidence and provenance boundaries

1. Package-owned candidate retrieval creates the candidate surface and never decides
   identity.
2. Same-table retrieval removes self-pairs and canonicalises mirrored pairs before comparison
   features are constructed.
3. Verified truth creates protected labels and mechanical evaluation only. It is never used
   as a score or fabricated probability.
4. Training, validation, calibration, decision, and locked-test authority remain partitioned
   by entity/household-connected components.
5. The pair classifier and calibrator are fitted, serialized, strictly reloaded, and then
   applied to feature-only decision evidence.
6. Assignment selects compatible scored edges only. A lower-ranked selected edge cannot be
   automatically confirmed.
7. Only the relationship policy can emit cross-source relationship status, and this I1C
   policy stage is used only by the three `link_only` combinations. `dedupe_only` and
   `link_and_dedupe` emit no relationship statuses and have `decision_authority: none` and
   `merge_authority: none`.
8. Locked-test rows are evaluated mechanically after freezing and are not used as scoring,
   selection, calibration, assignment, or decision evidence. Inference scoring and
   assignment consume feature-only decision rows. Package synthetic-provenance verification
   re-reads the full generated bundle, including truth and protected provenance, solely to
   authenticate the synthetic attestation; those data are not supplied as model or decision
   evidence.

`link_and_dedupe` uses one globally partitioned, surface-tagged training, validation, and
calibration matrix spanning cross-source, intra-source A, and intra-source B evidence. The
three surface batches must have byte-identical partition-assignment provenance. One strictly
bound champion and calibrator are fitted over the combined evidence. A cross-source-only
calibration artifact cannot authorise either same-source clustering surface.

## Immutable outputs

The mode orchestration and run artifacts contain only schema/version metadata, fixed
authority declarations, aggregate counts, and cryptographic provenance digests. They reject
unknown fields, duplicate identifiers, unsupported schemas, wrong seeds, inconsistent
surface sets, artifact or evidence digest drift, and provenance tampering. They contain no
record rows, pair identifiers, source values, raw identifiers, or local filesystem paths.

The link-only modes reuse a strict `PipelineRecipeArtifact` because their persisted champion,
calibrator, recipe assignment/decision, schema, and evidence bindings support replay.
Candidate ranks are derived deterministically from calibrated decision scores; no ranker is
persisted for this route and `ranking_artifact_digest` is `None`. The dedupe aggregate artifact
is not represented as a recipe and conveys no executable decision or merge authority.

## CLI and verification

```text
mapel-linkage run-linkage-mode --config CONFIG --project-root ROOT \
  --synthetic-demo --entity-count 120
scripts/run_i1c_linkage_modes.sh --dry-run
scripts/run_i1c_linkage_modes.sh --dry-run --full
scripts/run_i1c_linkage_modes.sh
scripts/run_i1c_linkage_modes.sh --full
```

The CLI refuses execution without `--synthetic-demo` and returns aggregate metadata only.
The shell driver provides a safe outside-Codex preflight, focused I1C qualification, and an
opt-in longer repository-wide gate. Verification caches remain outside the repository, the
distribution check runs from an external copy of tracked and nonignored candidate files, and
the driver fails if candidate content or Git/index state changes. It does not read ignored
private-data or artifact content. Required schema, capability-matrix, and repository-manifest
regeneration remains an explicit maintainer action.

## Residual risks and human gates

- Synthetic generation does not establish population fidelity, anonymity, fairness,
  sensitivity, positive predictive value, calibration transport, or operational fitness.
- Local owners must approve data governance, threat controls, population/subgroup
  validation, calibration, thresholds, review procedures, monitoring, rollback, retention,
  deletion, and any operational recipe.
- Cluster size and candidate-edge budgets are safety bounds, not operational parameters.
- Strict least-privilege data-access isolation is not established: package provenance
  authentication currently re-reads the complete generated-synthetic bundle, including
  protected truth, even though only feature-only decision rows cross the scoring, assignment,
  and decision-evidence boundary.
- Exact canonical-byte enforcement applies to the I1C mode orchestration and run artifacts.
  Inherited generic model and calibrator loaders validate schemas and digests but do not
  universally require serializer-byte identity; the I1C persistence wrappers compensate with
  full semantic artifact equality after reload. Universal canonical encoding for inherited
  loaders remains outside this milestone.
- The repository runner cannot ingest real rows, construct a master record, silently merge,
  mutate labels, retrain automatically, or grant release authority.
