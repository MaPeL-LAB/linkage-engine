# Unresolved Research Questions

The initial research supports the core architecture but does not settle every methodological or operational decision. These questions should be resolved through focused research, synthetic experiments, and later approved local validation.

## Candidate generation

- Which safe blocking DSL operations provide sufficient recall without permitting arbitrary SQL?
- When should sorted neighbourhood, canopy, LSH, ANN, or learned blocking become supported challengers?
- How should candidate budgets fail: hard stop, policy-defined truncation, or unresolved output?
- How should multi-source blocking preserve source-specific constraints?

## Comparison functions

- Which multilingual name normalisation and phonetic methods are appropriate for the intended jurisdictions and languages?
- How should address parsing be modularised without embedding country-specific assumptions in core code?
- Which Unicode confusables and transliteration operations are safe, reversible, and auditable?

## Labels and adjudication

- What verification protocol makes an adjudication outcome eligible for training, calibration, threshold selection, or testing?
- When is dual review required?
- How should reviewer disagreement and corrected decisions be represented?
- How should active-learning selection avoid oversampling uncertain cases in a way that distorts evaluation?

## Calibration and decisions

- When should beta calibration be added as a supported option?
- How large must calibration and decision partitions be for stable threshold selection?
- How should no-match utility be estimated and sensitivity-tested?
- What operational loss function should govern false links, missed links, and review burden?

## Multi-source entity resolution

- Should the first multi-source implementation use constrained graph clustering, partition models, correlation clustering, or source-aware optimization?
- How will cannot-link evidence and within-source uniqueness constraints be enforced?
- How will contradictory cycles be exposed for review rather than silently resolved?

## Privacy-preserving linkage

- What parties, threat model, and trust assumptions would govern PPRL?
- Is a trusted linkage unit acceptable, or is secure multiparty computation required?
- Which cryptographic protocol can support approximate matching without relying on vulnerable Bloom-filter encodings?
- How will key management, frequency attacks, collusion, replay, and disclosure risk be assessed?
- Should PPRL be a separate package rather than a core module?

## Validation and fairness

- Which protected or operational strata can be evaluated lawfully and safely?
- How should small cells be suppressed in aggregate reports?
- How should linkage error be propagated into downstream statistical analyses?
- What external gold-standard or clerical-review design is feasible for real validation?

## Performance and deployment

- What dataset sizes and candidate densities define the first performance target?
- Is local DuckDB sufficient, or is a distributed backend required later?
- Which operating-system sandbox and offline-execution controls are mandatory?
- What artifact-retention and deletion policies apply in operational deployments?

## Licensing and publication

- Which licence does MaPeL-LAB approve?
- Will the package remain private, become source-available, or be published to PyPI?
- What dependency and model-licence review is required before a release?

Each material resolution should produce or update an ADR.
