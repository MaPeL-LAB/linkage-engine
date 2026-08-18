# Local Handoff Checklist

## Repository and installation

- [ ] clone or unpack the checksummed approved repository snapshot;
- [ ] verify the Git commit and archive SHA-256 digest;
- [ ] install Python 3.12 and the tested constraints set;
- [ ] run repository verification, tests, build, and distribution inspection where permitted;
- [ ] run `mapel-linkage doctor`;
- [ ] run the complete generated-synthetic smoke test.

## Authorised workspace

- [ ] confirm the approved local project root and operating-system access controls;
- [ ] initialise ignored `private/`, `data/`, and `artifacts/` subdirectories;
- [ ] confirm encryption, backup, retention, deletion, and incident policies;
- [ ] confirm that Git remotes and automated sync tools cannot see restricted paths.

## Configuration and data

- [ ] prepare the completed project configuration only locally;
- [ ] map source columns to canonical variables without changing package model logic;
- [ ] document source quality, missingness, duplication, temporal coverage, and constraints;
- [ ] verify local path and output allow-lists;
- [ ] validate configuration before processing records.

## Truth and validation

- [ ] identify eligible verified truth and classify unverified references correctly;
- [ ] construct entity/household connected components;
- [ ] create protected training, validation, calibration, decision, and test partitions;
- [ ] verify zero cross-partition leakage;
- [ ] retain the locked test partition until all development decisions are frozen.

## Model and decision governance

- [ ] validate candidate retrieval before pair models;
- [ ] fit and compare models using approved partitions only;
- [ ] calibrate on the independent calibration partition;
- [ ] approve thresholds and no-match utility using the decision partition;
- [ ] run locked final test evaluation;
- [ ] approve the restricted review workflow and output recipients;
- [ ] record model, calibrator, ranker, assignment, configuration, and run digests.

## Prohibited transfers

- [ ] no real records, completed configs, identifiers, candidate pairs, adjudication rows, operational artifacts, or outputs are sent to ChatGPT, GitHub issues/PRs, repository examples, documentation, notebooks, or CI;
- [ ] no unverified crosswalk is used as training, calibration, threshold-selection, or test truth;
- [ ] no row values or candidate pairs appear in logs or unrestricted reports;
- [ ] no model or workflow silently creates a merged master record.

## Reproducible package handoff

Inside the verified Python 3.12 environment, run:

```bash
python scripts/build_local_handoff.py --project-root .
```

The builder fails closed when `dist/` is non-empty. Review its contents first; only then may an
operator explicitly add `--replace-build-output`.

The builder regenerates the schema and repository manifest, verifies the repository, runs all tests, builds wheel and source distributions, inspects distribution contents, runs a strict local dependency audit, creates a CycloneDX JSON software bill of materials, and writes SHA-256 checksums plus an aggregate local handoff manifest. Generated reports and distributions remain under ignored `artifacts/` and `dist/` paths and contain no record-level data.
