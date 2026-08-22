# API Stability Policy

## Current status

The package is `0.2.0.dev3` and pre-1.0. Public compatibility is versioned but not frozen. The
release-readiness policy therefore retains `api_stability_not_frozen` as a blocker.

## Public surfaces

The controlled public surfaces are:

- the `mapel-linkage` CLI command names, required flags, exit codes, and aggregate outputs;
- package exports documented in `__all__` and exercised by distribution tests;
- the generated configuration JSON Schema;
- recipe, model, calibration, stage, benchmark-governance, and qualification artifact schemas;
- stable safe error codes from the generated catalogue.

Everything else remains internal and may change before 1.0. Internal status does not weaken
privacy, partition, authority, path, or no-silent-merge invariants.

## Change rules

1. A breaking public change requires a new schema or contract version and migration assessment.
2. Readers remain strict: unknown fields, stale digests, invalid authority literals, and
   unsupported versions fail closed.
3. A producer must not silently emit a new interpretation under an old schema version.
4. Deprecations require documentation, tests for the replacement, and at least one retained
   development release before removal unless a security flaw requires immediate rejection.
5. Safe error codes remain stable within a versioned contract; arbitrary rejected values and
   local paths never become part of their public message.
6. Pre-1.0 flexibility never authorizes weakening calibration isolation, decision authority,
   assignment constraints, synthetic-only repository policy, or advisory-only boundaries.

## Freeze gate

API stability can be marked frozen only after the exported-surface inventory, compatibility
matrix, migration tooling, rollback drill, model cards, and private candidate acceptance evidence
are complete and separately approved.
