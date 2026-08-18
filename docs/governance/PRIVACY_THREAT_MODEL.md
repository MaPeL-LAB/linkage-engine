# Privacy Threat Model

**M2 development-candidate status:** configuration, path, error, aggregate logging, manifest,
output allow-list, synthetic-sentinel, local DuckDB, protected-label partition, native model
artifact, calibration, ranking, assignment, decision, and restricted-review boundaries are
implemented for the complete synthetic vertical slice. Real-data validation, operational
approval, deployment hardening, and external security assessment are not established.

## Protected material

Protected material includes real records, direct and quasi-identifiers, project configurations, source-system metadata classified as sensitive, candidate pairs, comparison values, labels, adjudication events, model artifacts, linkage outputs, logs, secrets, and credentials.

Protected material must never be committed or processed in repository CI.

## Threat boundaries and controls

| Boundary | Threat | Required control |
|---|---|---|
| configuration | code execution or SQL injection | strict schema, typed DSL, allow-list registries, no raw SQL/callables |
| filesystem | path traversal or arbitrary file access | real-path resolution under approved roots; local paths only |
| DuckDB | filesystem/network/extension access | package-owned SQL, restricted settings, approved extensions, OS/container controls [@duckdbsecurity2026] |
| logging | record or identifier leakage | typed aggregate events; reject row-like objects |
| exceptions | backend or validation values exposed | safe error codes; hidden input values; traceback disabled by default |
| model artifacts | memorized values or unsafe deserialization | native formats, local trust, manifests, no untrusted pickle loading |
| output | non-whitelisted fields exported | deny-by-default schema generation |
| labels | weak evidence promoted to truth | provenance eligibility enforcement |
| adjudication | sensitive fields or decisions leaked | restricted local files, append-only audit, no Git/CI |
| CI | real data or secret exposure | generated synthetic data, read-only token, no row artifact uploads |
| supply chain | mutable actions/dependencies | pinned GitHub Action SHAs; tested constraints; dependency audit |
| synthetic demo | operational configuration causes real input access | require synthetic truth and the exact generated two-source fixture paths before any generation, dataset read, or write |
| intermediate tables | same-schema materialisation overwrites prior evidence | derive table identity from both schema and package-owned input-table provenance; retain coverage assertions |

## Logging policy

Allowed log fields are structural and aggregate, such as:

```text
event
stage
run_id
model_id
rule_id
count
duration_ms
status
version
digest
safe_error_code
```

Prohibited fields include record values, identifiers, candidate pairs, configuration payloads, SQL with values, adjudication values, secrets, and model training rows.

## DuckDB boundary

DuckDB is not a sandbox. Even package-controlled database settings do not replace operating-system or container isolation when processing untrusted inputs [@duckdbsecurity2026]. The engine should establish security settings before data registration and prevent project configuration from changing them.

## Artifact policy

Unrestricted manifests may contain versions, digests, aggregate counts, timing, seeds, and metrics. They must not contain row previews, original IDs, pair lists, sensitive configuration values, or adjudication content.

## Model privacy

A model can encode information from training data even when raw rows are absent. Operational model artifacts therefore remain local and restricted. The first neural model uses derived comparison features rather than raw identifying text.

## PPRL boundary

Hashing or Bloom-filter encoding does not automatically satisfy privacy requirements. Known cryptanalytic attacks make Bloom-filter PPRL a separate, threat-model-driven research track [@kuzu2011cryptanalysis; @vidanage2020graphattack].

## Verification

Privacy tests generate unique synthetic sentinels and assert their absence from validation errors, CLI errors, typed log construction, emitted logs, manifests, and unrestricted representations. Validation locations replace arbitrary user-provided mapping keys with `*`. Distribution inspection rejects row-level and model-artifact file types, including JSONL/NDJSON.

The `--synthetic-demo` flag is not, by itself, sufficient authority to use an arbitrary
configuration. The runner rejects non-synthetic label authority, any configured label-source
path, non-generated dataset paths, and seed mismatch before creating fixtures or opening a
dataset. This is a software privacy boundary, not evidence of population fidelity or operational
fitness.
