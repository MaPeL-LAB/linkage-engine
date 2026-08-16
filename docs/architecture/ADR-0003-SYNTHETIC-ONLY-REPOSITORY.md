# ADR-0003: Synthetic-Only Repository and CI Boundary

- **Status:** Proposed
- **Date:** 2026-08-16

## Context

Record linkage commonly processes direct and quasi-identifiers. Repository, CI, issue, and agent systems are not approved operational data environments.

## Decision

Only generated synthetic record-level data may appear in source control, CI, tests, examples, documentation, notebooks, issues, pull requests, or agent conversations.

Real data, de-identified real data, real configurations, identifiers, candidate pairs, adjudication records, models, secrets, and outputs remain local under ignored directories.

CI generates synthetic fixtures during each job and uploads no row-level artifacts.

## Consequences

The repository can test architecture, privacy controls, and software behaviour without handling participant data. Real operational validation must occur separately under approved local governance.

## Rejected alternatives

- committing “anonymized” or hashed real records;
- using a real crosswalk as a test fixture;
- uploading local model artifacts to CI;
- relying on log masking after sensitive values are emitted.

## Acceptance

Privacy sentinel tests prove that synthetic record values and identifiers do not appear in logs, errors, unrestricted manifests, or package distributions. Repository verification rejects common row-level and model-artifact file types.
