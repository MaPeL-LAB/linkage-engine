# Roadmap

## Status vocabulary

Linkage Engine reports four independent states:

```text
specified
component implemented
workflow integrated
operationally validated
```

Source-code presence and unit tests do not by themselves make a component reachable from the
configuration-driven CLI. The normative status table is
[`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md).

## Completed integrated foundations

- **M0:** research, architecture, ADRs, governance, bibliography, package shell, and
  publication guard;
- **M1:** strict configuration compilation, path and logging controls, manifests, generated
  schema, and deterministic synthetic generation;
- **M2:** complete generated-synthetic two-source `link_only`, `one_to_one` workflow,
  including canonical preparation, anchors, bounded candidates, comparison features,
  Fellegi-Sunter and XGBoost pair models, validation-only champion selection, sigmoid,
  isotonic and Beta calibration, XGBoost ranking, explicit no-match assignment, four-status
  decisions, restricted review export, evaluation, and deterministic orchestration.

M2 is merged and CI-verified as software behaviour on generated synthetic data. It is not
operationally validated.

## Integrated milestones and bounded residuals

### M3 — adjudication and label lifecycle

The append-only adjudication ledger, bounded review import, disagreement handling,
multi-reviewer consensus, label-promotion controls, and active-learning ordering are integrated
and core-CI verified. Automatic retraining remains prohibited. Operational reviewer roles,
protocol approval, and any real-data retraining decision remain local governance work rather
than repository-granted authority.

### M4 — extended linkage modes

Many-to-one, one-to-many, unconstrained assignment, single-source deduplication, and combined
link-and-dedupe components are integrated with core-CI coverage; CI evidence is assessed per
commit. I1C adds bounded synthetic orchestration for exactly `link_only` with `many_to_one`,
`one_to_many`, or `unconstrained`;
`dedupe_only` with `unconstrained`; and `link_and_dedupe` with `one_to_one`. This does not
authorise arbitrary M4 combinations, multi-source or real-data dispatch, or operational use.

### M5 — broader model portfolio

I1B integrates the mandatory native Splink baseline with configured XGBoost, LightGBM,
PyTorch, stacking, XGBoost ranking, and LightGBM ranking candidates. The synthetic workflow
uses source-side entity/household-connected OOF groups for supervised stacking inputs,
validation-only champion/ranker selection, calibration-only fitting, and locked-test
evaluation after the champion and calibrator are frozen. It persists and strictly reloads the
champion bundle, calibrator, executable source-query ranker, and recipe-v1 binding before two
disjoint decision-partition replays. Target-query rankers are trained and reported but cannot
be silently used by the source-to-target assignment contract.

Operational model cards, promotion, threshold approval, and real-population validation remain
local human decisions. Native Splink is never treated as a generic boosted-feature or stacking
base artifact.

### M6 — optional neural matcher

The feature-based PyTorch matcher has bounded epochs, learning rate, weight decay, training
pair budget, CPU-device, thread, and deterministic-mode configuration. Its protected
tournament, immutable artifact reload, calibration, and recipe-bound synthetic replay path is
integrated and executed by all-models CI. It has no raw-text, identity, decision, or merge
authority.

### M7 — multi-source entity resolution

Source-aware N-dataset entity resolution and global crosswalk workflow integration are
core-CI verified. Correlation clustering, constrained agglomerative clustering, BCubed,
purity, pairwise metrics, cannot-link enforcement, and violation diagnostics remain declared
as component-only where the capability matrix says so. Building a production N-source graph
from locally approved pairwise recipes remains outside the complete synthetic CLI and does not
gain merge authority.

## I1A/I1B/I1C — orchestration, configured model portfolio, and bounded modes

I1 now establishes, for the bounded generated-synthetic two-source workflow:

- immutable artifact-to-artifact stage execution;
- protected portfolio training and out-of-fold stacking evidence;
- protected train, select, calibrate, locked-test, persist/reload, and infer boundaries;
- approved-recipe new-data inference;
- shadow challengers with no decision authority;
- configuration-driven native and optional-model portfolio execution.
- typed allow-list dispatch for the five I1C generated-synthetic combinations;
- canonical same-table retrieval plus fitted/reloaded pair-model and calibration evidence;
- feature-only decision rows for scoring and assignment, plus aggregate, tamper-evident
  mode-run provenance.

This does not claim one general CLI for every M3–M7 capability or any operational approval.
Package synthetic-attestation authentication still re-reads the complete generated bundle;
strict least-privilege data-access isolation is not established.

See
[`architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md`](architecture/ADR-0004-CAPABILITY-STATUS-AND-PIPELINE-INTEGRATION.md).

## I2A — Stage-1 Linkage Strategy Advisor

I2A is implemented as an advisory structural workflow:

```text
configuration-only preflight task profile
→ lifecycle-aware hard eligibility rules
→ mandatory Fellegi-Sunter baseline
→ structural Pareto frontier
→ family-diverse bounded shortlist
→ transparent explanations
→ explicit abstention from empirical ranking
```

The advisor fixes recommendation, decision, assignment, merge, and automatic-promotion
authority in immutable contracts. It makes no sensitivity, PPV, calibration, or operational
performance claim without benchmark evidence.

## B1 — synthetic benchmark evidence library

B1 is integrated: the repository includes the parametric scenario generator, benchmark
portfolio runner, failure/status contracts, and file-backed aggregate registry persistence.
Generated registries are synthetic evidence and are not an operational corpus or a claim of
population fidelity. A specific locally generated registry still needs adequate scenario
coverage before an advisor may rely on it. The corrected advisor-v2 execution completed its exact
9,800-run grid and passed evidence readiness; this closes the corpus-completeness gate but not the
separate strategy-quality gate.

## I2B — similarity and coverage advisor

I2B is integrated with nearest-family retrieval, weighted distance, coverage and
out-of-distribution checks, performance-distribution aggregation, uncertainty, and
abstention. It must abstain when the supplied registry lacks adequate coverage. Its first
prospective locked-family qualification returned `not_qualified`: it tied the best fixed recipe
rather than improving on it, and the current meta-features failed the OOD-detection gate.

```text
observable feature standardisation
nearest-family retrieval
coverage scoring
out-of-distribution detection
uncertainty and abstention
held-out-family regret and oracle-coverage evaluation
```

Family counts such as 50–100 are planning ranges, not automatic validity gates.

## I2C — learned meta-ranking advisor

I2C is integrated as a learned meta-regressor with conformal uncertainty and similarity
fallback. Its presence does not establish that a particular registry has adequate overlap or
that learned ranking outperforms transparent retrieval for an operational population. Its learned
utilities now order supported candidates, but the first prospective qualification did not improve
over Stage 2 and missed the locked interval-coverage gate by one of 24 family-recipe cells. Stage 3
therefore remains in fail-safe similarity fallback.

## I2D — active benchmark planning

I2D is integrated with snapshot-bound active synthetic benchmark planning, explicit human
execution approval, append-only evidence checks, and advisory refit. It cannot execute a
benchmark, promote a recipe, classify a relationship, or establish operational validity on
its own.

## M8 — release hardening

Phase 1 is integrated with a machine-readable fail-closed private-candidate policy, compatibility
and API/artifact governance, model cards, a generated error-code catalogue, security/dependency
controls, rollback procedures, and a resumable aggregate-only scale runner using 10 workers by
default. The bounded v2 synthetic scale evidence is complete and owner-approved for the development
envelope. A dry-run-gated, allow-listed aggregate run-manifest migration from schema `0.1` to `1`
is implemented and verified without artifact approval or release authority. The release policy
remains explicitly `blocked`.

Remaining exit work:

- freeze the pre-1.0 public surface after compatibility review;
- add dedicated macOS and Windows compatibility jobs where approved;
- perform a synthetic rollback drill and retain its aggregate evidence;
- obtain external security, licence, publication, visibility, and operational approvals separately.

Privacy-preserving record linkage remains a separate research stream requiring its own threat
model. No milestone implies operational validity or publication without explicit approval.
