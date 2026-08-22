# Artifact Migration Evidence

## Review status

The repository owner explicitly instructed the M8 release-readiness work to proceed on
2026-08-22. This record closes only `artifact_migration_tool_not_implemented` after the bounded
implementation and verification below. It does not authorize release, publication, deployment,
model promotion, operational artifact migration, or real-data use. Operational validity remains
`operational_validity=not_established`.

## Supported path

| Binding | Reviewed value |
|---|---|
| artifact kind | `run_manifest` |
| source | `source_schema_version=0.1` |
| target | `target_schema_version=1` |
| package transformation | `run_manifest_0_1_to_1` |
| plan schema | `1` |
| report classification | `aggregate_only` |
| migration authority | `none` |
| release authority | `none` |
| decision authority | `none` |
| assignment authority | `none` |
| merge authority | `none` |

The transformation changes only the run-manifest schema identifier. Every other strictly
validated aggregate field is preserved. Unsupported artifact kinds or version pairs remain
unsupported rather than being coerced into the target contract.

## Integrity evidence

- A canonical, digest-bound dry-run plan is required before a target write.
- Source and target paths must be distinct and remain under the fixed local `artifacts/` or
  `private/` envelope; path escape and symbolic links fail closed.
- The source is bounded to one megabyte, strictly parsed without duplicate or unknown fields,
  and bound by SHA-256 digest before transformation.
- The target is canonically serialized, strictly reloaded as schema `1`, and digest-verified.
- The source bytes are re-read and digest-verified after target creation, providing the retained
  rollback artifact without an inverse transformation.
- Target creation never replaces an existing file. Exact replay is idempotent; conflicting replay
  is rejected.
- Plans, results, and public errors contain aggregate contract metadata only and exclude artifact
  values and local paths.

## Verification boundary

Deterministic synthetic aggregate fixtures use seed `20260816`. Focused tests cover plan
serialization, dry-run separation, canonical target reload, semantic field preservation, source
retention, rollback reload, exact replay, conflicts, stale and tampered plans, duplicate and
unknown fields, unsupported versions, size bounds, path escape, source and target symbolic links,
privacy-safe errors, and the CLI plan/execute split.

The complete repository verification includes Ruff formatting and linting, strict mypy, the full
pytest suite, the generated error-code catalogue, fail-closed release verification, repository
hygiene, and distribution inspection. Synthetic checks establish bounded software behaviour only.
