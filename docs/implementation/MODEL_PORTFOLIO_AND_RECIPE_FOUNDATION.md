# Model Portfolio and Approved Recipe Foundation

## Status

Implemented as a bounded post-audit foundation. This milestone does not provide an operational data runner and does not approve any model for real-world linkage.

## Purpose

The earlier synthetic vertical slice represented one Fellegi–Sunter model, one boosted challenger, and one candidate ranker. The platform now needs a stable plural contract before configuration and orchestration are expanded to run LightGBM, PyTorch, and stacking candidates in a protected tournament.

`ModelPortfolioDeclaration` therefore represents:

- one mandatory baseline;
- zero or more bounded pair-model challengers;
- optional stacking candidates with explicit base-model IDs;
- zero or more candidate-ranker alternatives;
- a maximum challenger count;
- explicit shadow-scoring permission;
- a prohibition on using the locked test partition for portfolio selection;
- no decision, assignment, recommendation, or merge authority.

`compile_model_portfolio()` converts the current backward-compatible project configuration into this plural downstream contract. A later schema revision can expose the plural structure directly without changing the orchestration interface.

## Recipe persistence

The initial `PipelineRecipeArtifact` can be serialized to and restored from canonical JSON. The loader:

- applies a strict size limit;
- rejects duplicate and unknown keys;
- rejects unsupported schema versions;
- reconstructs only package-owned typed fields;
- verifies the stored recipe digest;
- contains no arbitrary code, import path, SQL, row value, identifier, candidate pair, or filesystem path.

A recipe remains unusable for inference unless it is explicitly `approved_for_inference` and records `locally_established` operational validation. Synthetic validation alone cannot grant inference authority.

## Authority boundary

```text
portfolio recommendation / selection -> evidence planning only
pair models                           -> evidence only
rankers                               -> ordering only
recipe                               -> binds approved artifacts and policy
assignment                           -> global selection only
relationship policy                  -> explicit status decision
merge authority                      -> none
```

No recipe creates a consolidated master record.

## I1B integration status

Plural candidates are exposed through the backward-compatible configuration schema and execute
in the I1B generated-synthetic workflow. The runner produces group-protected OOF manifests,
validation-only selection, calibration-only fitting, locked-test evaluation, strict artifact
reload, recipe-v1 binding, and typed recipe replay. Operational approval and optional shadow
challenger scoring remain separate future work; synthetic execution cannot supply either.
