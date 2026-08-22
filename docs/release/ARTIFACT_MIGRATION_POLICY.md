# Artifact Migration Policy

## Current status

The fail-closed migration utility is implemented for the first reviewed path: aggregate run
manifest schema `0.1` to schema `1`. The utility exposes separate dry-run and plan-bound execution
through `mapel-linkage migrate-artifact`; it does not discover or execute configuration-provided
transformations. The implementation evidence is retained in
[`ARTIFACT_MIGRATION_EVIDENCE.md`](ARTIFACT_MIGRATION_EVIDENCE.md).

`artifact_migration_tool_not_implemented` is closed. This does not approve migration of any other
artifact kind or version, and it does not grant artifact approval, release, publication,
deployment, decision, assignment, or merge authority.

## Invariants

- Migration never edits or replaces a source artifact in place.
- The original artifact, bytes, digest, schema version, and authority literals remain retained.
- A migration requires an explicit source-version to target-version allow-list entry.
- The target is written once, canonically serialized, reloaded, and digest-verified.
- Unknown fields, unsupported versions, symbolic links, path escape, digest mismatch, and
  conflicting output fail closed.
- Migration cannot create approval, promotion, release, decision, assignment, or merge authority.
- Restricted operational artifacts remain local and ignored; repository-safe fixtures are
  deterministic synthetic or aggregate-only.

## Required migration evidence

| Evidence | Requirement |
|---|---|
| source binding | immutable source digest and schema version |
| target binding | canonical target digest and schema version |
| transformation | package-owned named migration with source review |
| dry run | required before any target write |
| idempotence | exact replay accepted; conflicting replay rejected |
| rollback | retain and reload the original artifact without transformation |
| privacy | no row values, identifiers, candidate pairs, secrets, or local paths in reports |

No migration should be implemented merely to make an incompatible artifact load. The scientific
and governance meaning must be demonstrably preserved or the artifact must remain unsupported.

## CLI workflow

First emit and review the canonical no-write plan:

```bash
mapel-linkage migrate-artifact \
  --project-root . \
  --source artifacts/legacy-run-manifest.json \
  --target artifacts/run-manifest-v1.json \
  --artifact-kind run_manifest \
  --target-version 1 \
  --dry-run > artifacts/run-manifest-migration-plan.json
```

Then execute only that exact plan:

```bash
mapel-linkage migrate-artifact \
  --project-root . \
  --source artifacts/legacy-run-manifest.json \
  --target artifacts/run-manifest-v1.json \
  --artifact-kind run_manifest \
  --target-version 1 \
  --plan-file artifacts/run-manifest-migration-plan.json
```

Both files remain local under ignored artifact roots. Do not place operational artifacts or plans
in source control.
