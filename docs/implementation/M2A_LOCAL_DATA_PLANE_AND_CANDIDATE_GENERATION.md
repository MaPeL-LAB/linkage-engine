# M2A — Local data plane and candidate generation

**Status:** Implementation candidate  
**Parent milestone:** M2 — smallest complete synthetic vertical slice

## Purpose

M2A establishes the first row-bearing execution boundary without introducing a pair model or an identity decision. It provides:

- an opaque `TableRef` contract;
- a local DuckDB store with parameterised row insertion;
- fixed SQL type and identifier allow-lists;
- typed blocking predicates (`Exact`, `PrefixEqual`, `AllOf`, and `AnyOf`);
- a package-owned compiler from the predicate tree to quoted DuckDB SQL;
- union-and-deduplicate candidate retrieval;
- candidate-pair budget enforcement before materialisation;
- aggregate-only candidate diagnostics.

## Authority boundary

Candidate generation answers only:

> Which pairs should be compared further?

It does not estimate match probabilities, rank candidates, solve assignments, classify relationships, or merge records. Candidate presence is not evidence that two records describe the same entity.

## Privacy boundary

The data plane deliberately does not expose dataframe previews or convenience methods that print rows. Public representations contain only structural metadata. Internal exceptions are translated into stable public error codes without carrying submitted row values, identifiers, paths, or generated candidate pairs.

The committed tests construct synthetic rows in memory. No record-level fixture file is committed.

## Configuration integration

This slice accepts a canonical variable-to-column mapping and typed rule objects. The next M2 increment must adapt the compiled M1 `ExecutionPlan` into these runtime inputs. Project YAML will never supply raw SQL.

## Accepted scope

- Two-table `link_only` candidate retrieval
- Exact and prefix-equality predicates
- DuckDB local execution
- Explicit candidate budget
- Aggregate retrieval diagnostics

## Deferred scope

- project dataset ingestion from configured local paths;
- canonical normalisation tables;
- deterministic-anchor evidence;
- richer comparison functions;
- Splink/Fellegi–Sunter scoring;
- supervised pair classification;
- ranking, calibration, assignment, adjudication, and relationship decisions;
- deduplication and multi-source graph resolution.

## Verification

Acceptance requires repository verification, Ruff, mypy, unit tests, package build, restricted-distribution inspection, and synthetic-only GitHub Actions CI.
