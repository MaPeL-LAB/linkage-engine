# ADR-0001: Configuration-Compiled Privacy-Bounded Engine

- **Status:** Proposed
- **Date:** 2026-08-16

## Context

The engine must support heterogeneous datasets without embedding source schemas in Python code. It must expose configuration while preventing configuration from becoming an arbitrary execution channel.

## Decision

The package will:

1. parse YAML with `safe_load` or JSON with the standard parser;
2. validate strict Pydantic models with unknown fields forbidden;
3. hide submitted values from user-facing validation errors;
4. perform cross-field semantic validation;
5. compile configuration into an immutable `ExecutionPlan`;
6. resolve operations through package-owned allow-list registries;
7. expose a small typed blocking/comparison DSL;
8. prohibit raw SQL, arbitrary imports, configured callables, shell commands, `eval`, and `exec`;
9. resolve local paths under approved roots only;
10. reject remote URI schemes by default.

## Consequences

Benefits include a stable public contract, safer execution, backend replacement, deterministic testing, and centralized governance.

Costs include compiler/adaptor code, duplicated parity tests, and a deliberately smaller configuration language than the underlying libraries.

## Rejected alternatives

- raw Splink or DuckDB SQL in project configuration;
- dotted Python object paths;
- user plugins loaded from configuration;
- exposing backend-native objects as the public package API.

## Acceptance

This ADR is accepted when synthetic tests prove rejection of unknown fields, raw SQL, callable paths, remote URIs, out-of-root paths, invalid cross-field combinations, and value-bearing validation errors.
