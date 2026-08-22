# Synthetic Scale Benchmark Policy

## Purpose

The M8 scale runner measures elapsed time and peak resident memory for the complete generated-
synthetic vertical slice at increasing entity counts. It cannot establish population accuracy,
operational capacity, fairness, calibration, or deployment fitness.

## Fixed safeguards

- package-owned `configs/examples/synthetic_link_only.yaml` only;
- deterministic synthetic seed `20260816`;
- aggregate reports under ignored `artifacts/` only;
- no record values, identifiers, candidate pairs, local paths, or configuration payloads in reports;
- one isolated temporary project root per case;
- single-threaded numerical/model runtimes inside each case;
- 10 concurrent workers by default and at most 10 workers;
- at most 500 entities per case under the package-owned 100,000-pair candidate budget;
- exact plan-digest resume with conflicting evidence rejected;
- dry-run planning performs no writes;
- interruption preserves completed case reports for resume.

## Default matrix

| Entity count | Repetitions | Cases |
|---:|---:|---:|
| 100 | 2 | 2 |
| 200 | 2 | 2 |
| 300 | 2 | 2 |
| 400 | 2 | 2 |
| 500 | 2 | 2 |

The superseded v1 proposal used 100, 250, 500, 1,000, and 2,000 entities. Its two larger
sizes deterministically exceeded the package-owned 100,000-pair candidate budget and are not
accepted release evidence. The v2 matrix narrows the entity-count envelope instead of weakening
that safety control. V1 case reports remain ignored local diagnostic evidence and cannot be mixed
with the v2 plan.

Inspect the deterministic plan first:

```bash
scripts/run_m8_scale_benchmarks.sh --dry-run
```

Run or resume the long benchmark outside Codex:

```bash
scripts/run_m8_scale_benchmarks.sh
```

Use a new ignored output directory for a different matrix. Existing plans and completed cases are
accepted only when their canonical content and digests match exactly. The v2 default matrix is
complete and owner-approved for this bounded development envelope; its immutable aggregate binding
is recorded in [`SCALE_BENCHMARK_EVIDENCE_V2.md`](SCALE_BENCHMARK_EVIDENCE_V2.md). A future package,
configuration, implementation, matrix, or environment requires new evidence and review.
