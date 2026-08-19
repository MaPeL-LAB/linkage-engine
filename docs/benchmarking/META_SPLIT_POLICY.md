# Advisor Meta-Split Policy

## Principle

Advisor development and evaluation split at the **scenario-family** level. Instances and
replicates from one family cannot leak across meta-training, model selection, calibration, or
final advisor evaluation.

## Prospective partitions

```text
meta-training families
meta-validation families
coverage/OOD calibration families
locked advisor-test families
prospectively unseen corruption mechanisms
```

The split is fixed before inspecting pipeline-performance outcomes.

## Prohibitions

- splitting seeds from one instance across training and testing;
- selecting distance weights on locked advisor-test families;
- using latent synthetic parameters unavailable at recommendation time;
- dropping failed or ineligible pipeline runs;
- declaring readiness from a raw family-count threshold;
- using operational records in the global registry.

## Advancement gates

A later advisor advances only when held-out-family regret, oracle coverage, OOD detection,
abstention, and stability meet prospectively documented criteria and improve on simple fixed
pipeline baselines.
