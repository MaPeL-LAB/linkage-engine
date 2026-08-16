# Human Adjudication Workflow

## Purpose

Adjudication resolves uncertain evidence without granting automated models authority to silently merge records.

## Review reasons

A pair may enter review because of:

- probability within the review region;
- low top-two margin;
- model disagreement;
- conflicting deterministic anchors;
- assignment contention;
- missing critical evidence;
- incomplete or budget-limited candidate search;
- source-specific policy requirement.

## Restricted export

Review files remain local and ignored. Only explicitly approved review fields are included. Candidate pairs and review values are never logged.

## Outcomes

```text
match
nonmatch
uncertain
insufficient_information
duplicate_review
```

## Append-only event

Each event retains:

- adjudication event ID;
- pair surrogate;
- outcome;
- reviewer pseudonymous/local role ID;
- timestamp;
- protocol version;
- source run/artifact digests;
- optional second-review status;
- optional superseded event ID.

Corrections create superseding events rather than mutating history.

## Training eligibility

An adjudication outcome does not automatically become training truth. The label-provenance policy separately determines eligibility for training, validation, calibration, decision selection, and testing.

## Active learning

Later queue ordering may combine uncertainty, expected information gain, diversity, graph conflict, and review cost [@christen2020active; @primpeli2021almser]. Selection policy and sampling probabilities must be recorded to support bias assessment.

## Test protection

Review of locked test cases cannot inform model selection unless a new evaluation version and partition design is declared.
