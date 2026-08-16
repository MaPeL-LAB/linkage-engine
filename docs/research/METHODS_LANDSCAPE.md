# Methods Landscape

## Scope and terminology

This document surveys methods that may be implemented, wrapped, benchmarked, or deferred. “Established” means supported by substantial methodological or operational literature; it does not mean universally correct for every dataset.

## 1. Deterministic linkage and anchor evidence

Deterministic rules compare exact or tightly normalized values. They are useful for:

- high-specificity anchor evidence;
- seed examples for model diagnostics;
- unit and synthetic regression tests;
- operational rules that have been independently validated.

Risks include collisions, shared household values, copied identifiers, data-entry defaults, source-system reuse, and false confidence from apparently unique combinations. Every anchor needs uniqueness tests, contradiction checks, rule provenance, and a declared authority level. The default is `evidence_only`.

## 2. Candidate generation and blocking

Candidate generation controls computational scale and places an upper bound on downstream recall [@papadakis2020blocking].

| Method | Strength | Primary risk | Initial status |
|---|---|---|---|
| Exact multi-field blocking | Fast, interpretable | Misses corrupted true links | Implement first |
| Prefix and date-window blocking | Handles limited variation | Rule explosion, uneven block size | Implement first |
| Sorted neighbourhood | Useful when near values sort together [@hernandez1995mergepurge] | Sensitive to sort key/window | Implement as challenger |
| Canopy methods | Cheap approximate preclustering [@mccallum2000canopies] | Tuning and metric dependence | Later |
| Phonetic keys | Helpful for selected name systems | Language/culture dependence | Optional allow-listed transform |
| Locality-sensitive hashing | Scalable similarity retrieval | Collision/tuning complexity | Research track |
| ANN/embedding retrieval | Flexible semantic retrieval | Representation privacy and recall audits | Deferred |
| Learned/adaptive blocking | Can reduce comparisons | Training leakage and opacity | Later challenger |
| Post-hoc Bayesian blocking | Supports sophisticated inference [@mcveigh2019blocking] | Complex implementation | Deferred |

Each candidate pair must retain all retrieving rule IDs, candidate-set size, source-pair identity, and run-local provenance. Candidate budgets and block-size limits should fail safely rather than silently truncate without an `unresolved` signal.

## 3. Normalisation and comparison functions

### Strings

Useful measures include exact agreement, Jaro/Jaro–Winkler, Levenshtein and Damerau–Levenshtein distance, q-gram similarity, token-set overlap, and affine-gap variants. Comparison choice must be type- and language-aware. No single name metric is universally appropriate.

### Dates

Date comparison should distinguish exact agreement, plausible day/month transposition, bounded day/month/year differences, partial dates, and impossible values. Transformations must not silently “correct” ambiguous dates.

### Numeric and geographic values

Use absolute, relative, or scaled differences with explicit units. Geographic distance may be useful only where coordinates and governance permit it. The package should not infer location semantics from column names.

### Categorical and rare values

Exact agreement can be weighted by rarity. Fellegi–Sunter term-frequency adjustment provides one principled mechanism; supervised features may also include category frequency, but frequency statistics must be fitted within training boundaries.

## 4. Missingness

Missingness can be uninformative, informative, source-specific, or caused by linkage-related processes. The engine should support:

- explicit missing comparison levels;
- left-missing, right-missing, and both-missing indicators;
- source-specific missingness diagnostics;
- no implicit interpretation of two missing values as agreement;
- stratified performance by missingness pattern.

A model may learn from missingness indicators, but operational confirmation policy should not rely on a pattern that has not been validated in the relevant source systems.

## 5. Fellegi–Sunter and latent-class baselines

The classical model compares the probability of a comparison pattern among matches with its probability among nonmatches [@fellegi1969theory]. Practical implementations estimate `m` and `u` parameters using labelled data, random sampling, EM-style procedures, or combinations thereof [@winkler1991application; @fortini2020improved].

Strengths:

- interpretable evidence weights;
- unsupervised or weakly supervised estimation paths;
- natural review region;
- mature operational history.

Limitations:

- conditional-independence assumptions may be violated;
- pairwise scores do not automatically satisfy global matching constraints;
- parameter estimation can be sensitive to blocking and population shift;
- “unsupervised” does not mean validated.

## 6. Bayesian and structural models

Bayesian bipartite linkage models enforce one-to-one structure and quantify posterior uncertainty [@sadinle2017bayesian; @mcveigh2017practical]. Multi-file extensions model partitions or matching patterns [@sadinle2013generalized].

These methods are scientifically attractive for uncertainty-aware linkage but may be computationally demanding and harder to integrate into an initial general-purpose package. Interfaces should leave room for them without making them MVP dependencies.

## 7. Supervised pair classification

### Logistic regression

A strong interpretable baseline for comparison features. It supports odds-based reasoning but may miss nonlinear interactions unless engineered explicitly.

### Random forests

Robust nonlinear baseline; probability calibration may require care. Model artifacts can be less portable if stored through Python pickle.

### Gradient boosting

XGBoost is the initial default; LightGBM is the first challenger. CatBoost is deferred until categorical-treatment benefits are demonstrated. Gradient boosting can model nonlinear interactions between similarity, rarity, missingness, source pair, anchor evidence, and candidate-set context.

Training labels must be eligible verified truth. Unknown pairs remain unknown. Hard negatives should come from verified nonmatches that are plausible competitors, not arbitrary unlabeled pairs.

## 8. Learning to rank

Ranking groups candidates by query record and optimizes ordering. Suitable metrics include recall@K, mean reciprocal rank, true-match rank, and NDCG where graded relevance exists. XGBoost provides a documented ranking interface [@xgboostltr2026].

Ranker authority is deliberately limited:

```text
allowed: ranking_score, rank, top_k_membership, ranker_version
prohibited: confirmed, no_match, merged_entity_id
```

A true match absent from the candidate set cannot be recovered by ranking; candidate recall remains a separate gate.

## 9. Neural and representation models

Deep entity-matching systems can learn sequence and field interactions [@mudgal2018deep]. Transformer pair classifiers can be strong for text-rich product and bibliographic matching [@li2020ditto].

Initial risks:

- raw identifying text may be encoded or memorized in artifacts;
- pre-trained model provenance and licences add supply-chain complexity;
- GPU/CPU behaviour and reproducibility vary [@pytorchrepro2026];
- probability calibration is not guaranteed [@guo2017calibration];
- benchmark gains may not transfer to population linkage.

The first optional neural model should therefore use tabular comparison features and run offline. Raw-text or transformer models require a separate ADR and privacy review.

## 10. Calibration

Discrimination and probability calibration are different properties. Candidate methods:

| Method | Strength | Limitation | Initial status |
|---|---|---|---|
| Sigmoid/Platt | Stable parametric baseline | May impose unsuitable shape | Implement first |
| Isotonic | Flexible monotonic mapping | Can overfit small calibration sets | Implement first |
| Beta calibration | Richer parametric family [@kull2017beta] | Additional dependency/implementation | Challenger |
| Cross-fitted calibration | Uses data efficiently | More orchestration complexity | Later |

Calibrators must be fitted on data disjoint from base-model training [@sklearncalibration2026]. A score should not be named a probability until the artifact contract states how it was calibrated and evaluated.

## 11. Ensemble and champion–challenger strategy

The initial policy is champion–challenger, not automatic stacking:

1. train eligible model families;
2. compare discrimination, calibration, stability, and subgroup behaviour on validation data;
3. choose a champion under a documented policy;
4. fit calibration on a separate partition;
5. select thresholds on a decision partition;
6. evaluate once on the locked test partition.

Stacking or weighted-logit ensembles require out-of-fold predictions and their own provenance to avoid leakage.

## 12. Assignment and constraints

| Constraint | Interpretation |
|---|---|
| `one_to_one` | each source and target has capacity one |
| `many_to_one` | source capacity one; target capacity greater than one |
| `one_to_many` | target capacity one; source capacity greater than one |
| `unconstrained` | pair decisions without a global capacity solver |

OR-Tools minimum-cost flow is the initial sparse solver [@ortoolsassignment2026]. SciPy rectangular assignment is the small synthetic oracle [@scipyassignment2026]. More general side constraints may require CP-SAT or mixed-integer programming.

Use deterministic integer costs and private dummy no-match arcs. Assignment results do not by themselves imply confirmation; the decision policy still checks probability, margin, calibration, data quality, and review conditions.

## 13. Multi-source and collective entity resolution

Pairwise chaining can create contradictory clusters. Multi-source linkage must consider transitivity, source-specific duplicate constraints, cannot-link evidence, and graph consistency [@sadinle2013generalized; @zhang2015graph].

Possible later approaches include:

- constrained connected components;
- correlation clustering;
- graph partition models;
- collective/relational evidence;
- source-aware assignment and partition inference.

The package should not label sequential pairwise merges as a complete `multi_source` implementation.

## 14. Human review and active learning

Review queues should prioritize uncertainty, low margins, model disagreement, assignment conflicts, retrieval warnings, or policy-defined risk. Active learning can reduce annotation effort [@christen2020active; @primpeli2021almser].

Every review event is append-only and versioned. Training eligibility is a separate governance decision. Test-set adjudication must not flow back into model selection without a new versioned evaluation design.

## 15. Privacy-preserving record linkage

Bloom-filter PPRL enables approximate comparisons over encoded q-grams [@schnell2009pprl], but known attacks exploit frequency, constraints, and graph structure [@kuzu2011cryptanalysis; @vidanage2020graphattack]. Evaluations also show that utility and privacy depend strongly on protocol and context [@randall2022blindedeval].

Consequently:

- hashing is not anonymisation;
- Bloom filters are not a default secure transport format;
- PPRL requires explicit parties, attacker model, key management, protocol, disclosure analysis, and legal governance;
- cryptographic PSI/secure computation and trusted linkage units should be evaluated separately;
- no PPRL implementation belongs in the initial vertical slice.

## 16. Synthetic evaluation

Synthetic truth is generated before source corruption. Required corruption families include character edits, token reordering, Unicode/case variation, date perturbation, missingness, duplicates, shared values, contradictory fields, no-match entities, near ties, and assignment conflicts.

Synthetic benchmark results are regression tests of software behaviour, not population-valid performance estimates.
