# Benchmark Registry

## Scope

The global registry stores aggregate results from generated synthetic linkage scenarios. Local
schema-matched synthetic and local verified evidence remain under local ignored roots and are
not uploaded to the repository.

## Registry units

| Unit | Definition |
|---|---|
| Scenario family | A scientifically coherent mechanism or corruption regime |
| Scenario instance | One parameterised point inside a family |
| Replicate | One seed/population realisation of an instance |
| Benchmark run | One instance, pipeline recipe, seed, and software/environment version |

## Required aggregate provenance

Each run records:

```text
family, instance, replicate, and run IDs
task-profile digest
pipeline-recipe digest
engine commit
dependency-lock and environment digests
seed
run status and stable failure code
aggregate-metrics digest
runtime and peak memory
stage-artifact manifest digest
```

The registry contains no records, source identifiers, source fields, candidate pairs, local
paths, model-training rows, or unrestricted artifacts.

## Failure retention

Unsuccessful fits, timeouts, memory failures, ineligible recipes, abstentions, numerical errors,
and candidate-budget failures are retained. A pipeline that does not complete is part of the
comparative evidence.

## Evidence hierarchy

```text
global synthetic
local schema-matched synthetic
local verified validation
local operational monitoring
```

Evidence classes are not pooled silently. Local synthetic evidence remains synthetic, and local
verified evidence is eligible only under an approved protocol.

## Advisor-v2 implementation boundary

The versioned advisor-scale design contains 64 families and 280 instances, partitioned
prospectively into 40 meta-training, 8 conformal, 8 locked-evaluation, and 8 real-mechanism OOD
families. The 10-family/19-instance seed-v1 catalogue remains stable. Its historical
transliteration family is retained for compatibility but excluded from true OOD readiness.

Three package-owned link-only adapters can produce comparative success evidence: the
Fellegi-Sunter reference, XGBoost classifier, and XGBoost ranker. Synthetic truth is restricted to
protected supervised training labels and post-score mechanical evaluation. Dedupe-only,
multi-source, LightGBM, and PyTorch recipes retain stable ineligible evidence until real adapters
exist; their metrics are never fabricated.

The CLI can inspect the aggregate design and execute one explicitly approved deterministic shard.
Manifests and run evidence are append-only, digest-bound, idempotently resumable, and rejected on
tamper, collision, or environment drift.

The 2026-08-21 execution-protocol-v1 diagnostic registry completed 9,800 records but retained 688
Fellegi-Sunter scoring failures. Its family-overlap readiness result is superseded: difficult
failed replicates cannot be silently omitted from meta-training. Execution protocol v2 uses a new
registry provenance boundary and requires five or more replicates per instance plus successful
Fellegi-Sunter, XGBoost-classifier, and XGBoost-ranker evidence in every scenario-replicate cell.
The diagnostic registry is never overwritten, and advisor validation remains unestablished until
the corrected execution-v2 audit passes.
