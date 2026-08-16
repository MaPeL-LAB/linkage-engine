# ADR-0002: Separate Retrieval, Scoring, Ranking, Assignment, and Decision Authority

- **Status:** Proposed
- **Date:** 2026-08-16

## Context

A record pair can receive a high local score while still conflicting with other pairs, failing a global capacity constraint, or lacking sufficient evidence for confirmation. Candidate ranking optimizes ordering, not identity certainty.

## Decision

The pipeline will maintain separate interfaces for:

- candidate generation;
- comparison feature construction;
- pair scoring;
- candidate ranking;
- probability calibration;
- global assignment;
- relationship decision policy;
- adjudication.

A ranker may emit score, rank, top-K membership, and provenance only. An assignment solver selects real or no-match edges under constraints but does not determine `confirmed` status. Decision policy combines calibrated probability, margin, assignment, anchor evidence, data-quality state, and explicit thresholds.

No interface exposes an implicit `merge()` or master-record operation.

## Consequences

- candidate recall and model discrimination can be evaluated independently;
- one-to-one constraints can change the chosen candidate without rewriting model scores;
- unresolved and no-match remain distinct;
- model authority is testable and auditable.

## Rejected alternatives

- accept top-ranked candidate automatically;
- treat a classifier threshold as global assignment;
- convert absence of a selected edge into no match;
- merge source rows as a side effect of prediction.

## Acceptance

Tests prove that ranker output has no relationship status, no model can call a merge operation, assignment has zero capacity violations, and decision outcomes are exhaustive and policy-derived.
