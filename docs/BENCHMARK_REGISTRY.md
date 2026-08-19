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

## Current implementation boundary

I2A provides immutable registry contracts and deterministic snapshot construction. It does not
run the model portfolio, populate the global corpus, retrieve nearest scenarios, or train a
meta-recommender. Those belong to B1 and later I2 stages.
