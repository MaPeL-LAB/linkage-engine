# Research Synthesis and Evidence-Based Recommendations

## Purpose

This synthesis translates the initial research review into repository decisions for a reusable Python package supporting record linkage, entity resolution, multi-source integration, and deduplication. It is not an implementation report and does not claim that any model has been validated on real populations.

## Executive findings

Record linkage is not a single binary-classification task. A defensible system separates candidate retrieval, comparison construction, pair evidence, ranking, calibration, assignment, decision policy, human review, and validation. The original Fellegi–Sunter framework explicitly allowed link, non-link, and possible-link regions rather than forcing every pair into a binary outcome [@fellegi1969theory]. Later statistical work models the matching structure itself, including bipartite constraints and unresolved portions of a linkage [@sadinle2017bayesian; @sadinle2013generalized].

The package should therefore produce relationships and evidence, not silently synthesize a master record. Candidate rank is not identity. Pair probability is not global assignment. Absence of a selected candidate is not automatically a defensible no-match conclusion.

### Recommended architectural baseline

1. **Compile configuration into an execution plan.** Pydantic validates a small schema; package-owned registries and typed DSL nodes are the only configurable operations. Raw SQL and configured Python callables remain prohibited.
2. **Use DuckDB as the local execution substrate.** It supports efficient local analytical workflows, but security settings and operating-system boundaries are required because unrestricted SQL can access files and external resources [@duckdbsecurity2026].
3. **Use Splink as the Fellegi–Sunter baseline behind an adapter.** Splink is designed for scalable probabilistic linkage and supports term-frequency-aware comparison weights and unsupervised estimation [@linacre2022splink; @splinkdocs2026]. Its public SQL expressiveness must not become the package configuration language.
4. **Use XGBoost first for supervised pair classification and learned ranking.** Native JSON/UBJSON model IO and documented learning-to-rank support align with model-governance and candidate-ranking requirements [@xgboostmodelio2026; @xgboostltr2026].
5. **Use LightGBM as the first challenger, not an immediate required dependency.** It provides classification and ranking objectives and explicit deterministic-mode controls, with platform and implementation caveats [@lightgbmparams2026].
6. **Keep PyTorch optional and challenger-only initially.** Reproducibility is constrained by release, platform, hardware, and algorithm choices [@pytorchrepro2026]. The first neural model should consume comparison features rather than raw identifying text.
7. **Use OR-Tools for sparse constrained assignment.** SciPy provides a small-problem reference oracle; NetworkX supports components, graph diagnostics, and validation grouping [@ortoolsassignment2026; @scipyassignment2026].
8. **Separate model training, selection, calibration, threshold selection, and final testing.** Calibration must use data disjoint from base-model fitting [@sklearncalibration2026; @guo2017calibration]. Beta calibration is a credible later challenger when sigmoid calibration is structurally inadequate [@kull2017beta].
9. **Split by entity and household connected groups before pair construction.** Random pair splits can leak the same entity, household, or near-duplicate evidence across partitions and produce over-optimistic estimates.
10. **Treat candidate recall as a first-class metric.** Blocking can irreversibly remove true links before scoring; it must be evaluated independently [@papadakis2020blocking].

## Established methods to implement first

### Deterministic anchors

Exact or highly specific combinations can provide high-confidence evidence, but uniqueness, contradiction checks, and provenance are essential. Anchors remain evidence-only by default and are never automatically promoted to training truth.

### Fellegi–Sunter baseline

Fellegi–Sunter comparison weights remain the foundational interpretable baseline [@fellegi1969theory]. Practical extensions include EM estimation, frequency-aware evidence, missing comparison levels, and operational review regions [@winkler1991application; @winkler2006overview; @fortini2020improved].

### Supervised comparison-feature models

Logistic regression provides an interpretable baseline. Gradient-boosted trees are the first high-capacity supervised models because they handle nonlinear feature interactions, missingness indicators, blocking provenance, and candidate-set context without requiring raw-text embeddings.

### Candidate ranking

A ranker groups target candidates by query record and optimizes ordering. Its outputs are rank, top-K membership, and ranker provenance. A classifier/calibrator and assignment/decision layer retain final authority.

### Explicit assignment and no-match

One-to-one and capacity-constrained linkage should be solved globally over the candidate graph. Each query record receives an explicit private dummy no-match option. Global assignment and local pair scoring are separate because a set of individually plausible edges may be jointly inconsistent [@sadinle2017bayesian; @zhang2015graph].

### Human adjudication

Human review is a versioned evidence process rather than a spreadsheet side effect. Active-learning research supports prioritizing informative cases, but adjudication events should not automatically enter every training or test set [@christen2020active; @primpeli2021almser].

## Alternatives that should remain available

- **Python Record Linkage Toolkit** is valuable as a modular research comparator with blocking, sorted neighbourhood, comparisons, supervised classifiers, and ECM-style unsupervised classification [@debruin2019recordlinkage]. It is not selected as the engine core because the proposed architecture needs stricter configuration, table-reference, provenance, assignment, and privacy contracts.
- **dedupe** offers mature active-learning workflows and learned blocking/matching for structured data [@gregg2022dedupe]. Its interactive training concepts should inform adjudication and active-learning design without making it the public architecture.
- **Zingg** provides Spark-oriented entity-resolution workflows and a Python API [@zinggdocs2026]. Its Spark runtime is heavier than the initial local Python/DuckDB scope.
- **hlink** is a configuration-driven probabilistic linkage system designed for scale [@ipums2026hlink]. It is an important architectural comparator, especially for hierarchical blocking and large historical linkage, but introduces a distinct Spark-oriented stack and licence.
- **CatBoost** is a possible later classifier/ranker challenger; it should not expand the initial dependency surface before evidence shows a material benefit.

## Emerging or deferred methods

### Transformer and representation models

DeepMatcher-style neural architectures and transformer pair classifiers such as Ditto demonstrate strong performance in some text-rich benchmarks [@mudgal2018deep; @li2020ditto]. They are deferred because they increase dependency size, artifact complexity, hardware variability, calibration burden, and privacy risks from raw textual representations.

### Bayesian and partition models

Bayesian bipartite and multi-file models provide coherent uncertainty and structural matching constraints [@sadinle2017bayesian; @mcveigh2017practical; @sadinle2013generalized]. They should inform interfaces and later research, but are not required for the smallest synthetic vertical slice.

### Privacy-preserving record linkage

Bloom-filter PPRL established an influential approximate encrypted-linkage approach [@schnell2009pprl], but subsequent cryptanalysis demonstrated that encoding identifiers into Bloom filters does not by itself provide a sufficient security guarantee [@kuzu2011cryptanalysis; @vidanage2020graphattack]. PPRL is therefore a separate future capability requiring a threat model, cryptographic expertise, protocol review, and deployment assumptions. It is not a configuration switch added to the MVP [@vatsalan2013taxonomy; @randall2022blindedeval].

## Validation consequences

Linked-data research warns that false links, missed links, and non-random linkage error can bias downstream analyses [@harron2017guide; @harron2017challenges]. Evaluation must therefore include:

- candidate recall@K and zero-candidate rate;
- sensitivity and positive predictive value;
- false-link and missed-link rates;
- precision–recall curves and average precision;
- reliability diagrams, Brier score, calibration slope/intercept;
- rank-based metrics;
- global assignment accuracy and constraint violations;
- performance by missingness pattern, candidate-set size, source pair, and blocking provenance;
- synthetic corruption and edge-case regression tests.

Synthetic testing demonstrates software behaviour under controlled assumptions. It does not establish operational validity, fairness, calibration, or public-health fitness.

## Final recommendation

Proceed with a documentation-governed, configuration-compiled architecture and implement the smallest complete two-source synthetic vertical slice before adding multi-source clustering, raw-text neural models, PPRL, or a browser-based review interface. The initial slice should prove privacy boundaries, provenance, retrieval recall, model authority separation, calibration discipline, explicit no-match assignment, and deterministic regression behaviour.
