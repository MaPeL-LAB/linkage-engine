# Artifact Migration Policy

## Current status

Artifact schemas are versioned and strictly loaded, but a general migration utility is not yet
implemented. `artifact_migration_tool_not_implemented` therefore remains a release blocker.

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
