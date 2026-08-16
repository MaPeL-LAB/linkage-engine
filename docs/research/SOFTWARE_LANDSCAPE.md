# Software and Tooling Landscape

## Decision criteria

Software is assessed for methodological fit, extensibility, local/offline operation, Python 3.12 compatibility, artifact governance, licence, maintenance, scalability, and alignment with the package-owned privacy/configuration boundary.

## Recommended implementation stack

| Component | Role | Why selected | Boundary |
|---|---|---|---|
| Pydantic | configuration validation | typed unions, cross-field validation, forbidden extras, safe errors | configuration still requires package-owned semantic validation |
| PyYAML | YAML parsing | mature safe loader | use `safe_load`; no custom Python tags |
| DuckDB | local table execution | embedded analytical engine, strong SQL performance | package-generated SQL only; filesystem/network restrictions [@duckdbsecurity2026] |
| Splink | Fellegi–Sunter baseline | scalable probabilistic linkage, term-frequency support [@linacre2022splink; @splinkdocs2026] | adapter only; no raw Splink SQL in project config |
| scikit-learn | calibration and metrics | mature calibration and evaluation APIs [@sklearncalibration2026] | do not persist arbitrary untrusted pickles |
| XGBoost | pair classifier and ranker | classifier/ranking APIs; native model format [@xgboostmodelio2026; @xgboostltr2026] | comparison features only initially |
| LightGBM | challenger | strong classification/ranking; deterministic controls [@lightgbmparams2026] | optional extra |
| PyTorch | optional neural matcher | flexible modelling | deferred; offline, feature-based first [@pytorchrepro2026] |
| OR-Tools | global assignment | sparse min-cost flow and richer optimization options [@ortoolsassignment2026] | integer costs and deterministic tie rules |
| SciPy | reference assignment and metrics | rectangular assignment [@scipyassignment2026] | small-problem oracle, not primary sparse backend |
| NetworkX | graph diagnostics | connected components and graph QA | not the production optimization solver |

## Existing record-linkage frameworks

### Splink

**Adopt and wrap.** Splink is the preferred Fellegi–Sunter baseline and an important source of model diagnostics. The engine should own configuration, candidate provenance, output governance, validation, assignment, and decision policy rather than expose Splink as the public architecture.

### Python Record Linkage Toolkit

**Benchmark and learn from; do not make core.** The toolkit provides modular indexing, sorted neighbourhood, comparisons, supervised classifiers, ECM-style unsupervised classification, and evaluation [@debruin2019recordlinkage]. It is particularly useful as a small/medium-data reference implementation. Its pandas-centric API and direct column-oriented workflow are less aligned with the proposed opaque table-reference and strict configuration boundary.

### dedupe

**Learn from active learning and learned blocking.** `dedupe` is a mature Python library for structured-data deduplication and record linkage using human training data and active learning [@gregg2022dedupe]. Its review interaction and learned predicates should inform later adjudication and candidate-generation challengers. The project should not copy its assumption that interactive training examples automatically become sufficient truth for every evaluation purpose.

### Zingg

**Defer as a Spark-scale comparator.** Zingg exposes entity-resolution, identity-resolution, record-linkage, and deduplication workflows through a Python/Spark stack [@zinggdocs2026]. It may be relevant when distributed Spark execution is justified, but it adds a Java/Spark runtime and a configuration ecosystem outside the initial local Python/DuckDB boundary.

### hlink

**Architectural comparator for large hierarchical linkage.** hlink is configuration-driven and supports preprocessing, filtering, training, blocking, feature generation, and scoring at scale [@ipums2026hlink]. It is especially relevant to historical census linkage. Its Spark orientation and MPL-2.0 licence mean it should be evaluated as a separate adapter or source of design lessons rather than embedded casually.

### FEBRL

**Historical reference only.** FEBRL influenced later Python record-linkage toolkits and remains important to the history of preprocessing, indexing, comparison, classification, and evaluation. Its age and legacy stack make it unsuitable as a new dependency; use modern packages and the methodological literature instead [@christen2012data; @debruin2019recordlinkage].

## Model libraries

### XGBoost

Initial default for boosted pair classification and ranking. Persist native model files with a package manifest; do not use arbitrary Python snapshots as the canonical format [@xgboostmodelio2026].

### LightGBM

Initial challenger. Validate determinism within a declared environment and record thread counts, parameters, compiler/platform, and library version [@lightgbmparams2026].

### CatBoost

Research comparator. Its categorical handling may be useful, but much of the proposed architecture already transforms source fields into comparison features. Add only after a benchmark demonstrates benefit that justifies the dependency and artifact surface.

### PyTorch

Optional neural adapter only. The package must document that exact reproducibility is not guaranteed across all versions/hardware [@pytorchrepro2026]. Raw-text embeddings require a new privacy ADR.

## Assignment and graph tools

OR-Tools is primary because candidate graphs are sparse and assignment constraints may expand. SciPy is the correctness oracle for small matrices. NetworkX is the graph-analysis layer for connected components, split construction, source/entity graph diagnostics, and conflict analysis.

## Storage and serialization

| Artifact | Preferred format |
|---|---|
| source/intermediate tables | restricted DuckDB or Parquet under ignored roots |
| configuration | original local file plus digest; redacted structural manifest only |
| Splink model | Splink-native JSON/settings plus engine manifest |
| XGBoost | `.ubj` or `.json` plus engine manifest |
| LightGBM | native model text/file plus engine manifest |
| calibrator | package-defined JSON/array representation where feasible |
| PyTorch | controlled architecture manifest and local weights; never untrusted arbitrary object loading |
| metrics | aggregate JSON/Markdown |
| adjudication | restricted local file with append-only event semantics |

## Licensing implications

The project licence remains undecided. Dependency licences must be inventoried before release. Alternative frameworks should not be copied into the repository merely because they are open source; licence compatibility and attribution must be reviewed.

## Adoption summary

| Software | Decision |
|---|---|
| Splink | adopt behind adapter |
| DuckDB | adopt as local execution engine |
| XGBoost | adopt as initial boosted classifier/ranker |
| LightGBM | optional challenger |
| PyTorch | optional later neural matcher |
| OR-Tools | adopt for global assignment |
| SciPy | adopt for reference assignment and metrics |
| NetworkX | adopt for graph diagnostics |
| recordlinkage | benchmark/reference |
| dedupe | learn from active learning/learned blocking |
| Zingg | defer distributed comparator |
| hlink | architectural comparator; possible later adapter |
| FEBRL | historical reference only |
| CatBoost | defer pending benchmark |
