# ADR-0001: Configuration-Compiled Privacy-Bounded Engine

- **Status:** Accepted
- **Decision date:** 2026-08-16
- **Implemented in:** M1 / `0.1.0.dev1`

## Context

The engine must support heterogeneous datasets without embedding source schemas in Python code. It must expose configuration while preventing configuration from becoming an arbitrary execution channel. Configuration values and identifiers may themselves be sensitive.

## Decision

The package will:

1. parse YAML with a bounded `SafeLoader` subclass or JSON with duplicate-key and non-finite-number rejection;
2. validate frozen Pydantic models with immutable mapping fields and unknown fields forbidden;
3. hide submitted values and arbitrary mapping keys from user-facing validation errors;
4. perform cross-field semantic validation;
5. compile configuration into an immutable `ExecutionPlan`;
6. resolve operations through package-owned immutable allow-list registries;
7. expose a small typed blocking/comparison DSL;
8. prohibit raw SQL, arbitrary imports, configured callables, shell commands, `eval`, and `exec`;
9. resolve local paths under configured roots and a separate host-approved envelope;
10. reject remote URI schemes, UNC paths, home expansion, out-of-root paths, and project-root widening by default;
11. canonicalise configuration and registry contents into SHA-256 digests;
12. keep operational paths out of public object representations and unrestricted manifests.

## Consequences

Benefits include a stable public contract, safer execution, backend replacement, deterministic testing, centralized governance, and value-safe errors.

Costs include compiler and adapter code, duplicated parity tests, a deliberately smaller configuration language than underlying libraries, and the need to maintain a generated JSON Schema.

## Rejected alternatives

- raw Splink or DuckDB SQL in project configuration;
- dotted Python object paths;
- user plugins loaded from configuration;
- project configuration self-authorising arbitrary filesystem roots;
- exposing backend-native objects as the public package API;
- returning raw Pydantic or backend exceptions to the CLI.

## Acceptance evidence

M1 tests cover unknown and duplicate fields, YAML alias/merge/depth controls, non-finite and non-JSON scalars, raw SQL/callable/module attempts, unverified truth, enabled-model calibration, remote paths, out-of-root paths, host-envelope escape, root widening, incompatible operations and levels, non-disableable safeguards, invalid thresholds, hidden values and map keys, immutable registries and configuration mappings, output deny-by-default, safe logging, safe manifests, deterministic synthetic generation, safe synthetic writes, and generated-schema parity.

Full acceptance for the remote repository additionally requires its Python 3.12 CI job to pass formatting, linting, typing, tests, build, and distribution inspection.
