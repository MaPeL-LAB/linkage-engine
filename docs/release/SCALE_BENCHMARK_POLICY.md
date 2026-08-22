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
- exact plan-digest resume with conflicting evidence rejected;
- dry-run planning performs no writes;
- interruption preserves completed case reports for resume.

## Default matrix

| Entity count | Repetitions | Cases |
|---:|---:|---:|
| 100 | 2 | 2 |
| 250 | 2 | 2 |
| 500 | 2 | 2 |
| 1,000 | 2 | 2 |
| 2,000 | 2 | 2 |

Inspect the deterministic plan first:

```bash
scripts/run_m8_scale_benchmarks.sh --dry-run
```

Run or resume the long benchmark outside Codex:

```bash
scripts/run_m8_scale_benchmarks.sh
```

Use a new ignored output directory for a different matrix. Existing plans and completed cases are
accepted only when their canonical content and digests match exactly. `scale_evidence_not_completed`
remains a release blocker until the default matrix completes and its aggregate results receive
human review.
