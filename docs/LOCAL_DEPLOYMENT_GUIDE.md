# Local Deployment Guide

## Purpose

This guide prepares an authorised local environment for Linkage Engine without placing real records, configurations, identifiers, labels, adjudication material, model artifacts, or linkage outputs in Git, ChatGPT, issues, pull requests, documentation, examples, or CI.

## Supported starting environment

- Python `3.12.x`;
- local filesystem access to an approved project root;
- sufficient disk space for local DuckDB databases and restricted artifacts;
- operating-system controls appropriate to the data classification;
- no requirement for cloud or external model services.

The repository build does not provide host-level sandboxing. The data custodian remains responsible for authorised accounts, encryption, backups, network policy, malware protection, access logging, and retention.

## Bootstrap

On macOS or Linux:

```bash
./scripts/bootstrap_local.sh /approved/path/linkage-project
```

On Windows PowerShell:

```powershell
./scripts/bootstrap_local.ps1 -ProjectRoot C:\approved\linkage-project
```

The scripts verify Python 3.12, create a local virtual environment, install the tested dependency set, initialise ignored local directories, and run the aggregate-only environment doctor. They do not create participant-like example rows.

## Local workspace

`mapel-linkage init-local-project` creates the following ignored structure:

```text
private/config/
private/labels/
private/adjudication/
private/outputs/
data/raw/
data/derived/
artifacts/models/
artifacts/runs/
artifacts/reports/
```

Do not weaken `.gitignore` to expose these paths. Do not copy completed operational configuration into `configs/examples/`.

## Environment doctor

```bash
mapel-linkage doctor --project-root /approved/path/linkage-project
```

The doctor reports only aggregate environment state: Python and dependency availability, required local-directory state, write permissions, publication guard, and schema availability. It does not inspect or print records, source columns, identifiers, candidate pairs, or completed configuration content.

## Configuration preparation

Begin from `configs/templates/local_project.template.yaml` and the worksheets in `docs/local_templates/`. Complete them only inside the authorised local environment.

Required local decisions include:

- dataset roles and local paths;
- source identifier columns;
- canonical variable mappings;
- allow-listed normalisation;
- blocking and comparison rules;
- eligible label provenance;
- protected split policy;
- candidate, model, and solver budgets;
- output allow-list;
- decision thresholds approved for the local project.

Run configuration validation before opening any dataset:

```bash
mapel-linkage validate-config \
  --config private/config/project.yaml \
  --project-root /approved/path/linkage-project
```

## Synthetic smoke test

Run the repository example before any operational execution:

```bash
mapel-linkage run \
  --config configs/examples/synthetic_link_only.yaml \
  --project-root . \
  --synthetic-demo
```

The result establishes installation and software behaviour only.

## Operational execution boundary

The repository CLI deliberately requires `--synthetic-demo` for row-level examples. Operational execution should be enabled only through a separately reviewed local integration after the governance and validation runbook is completed. That integration must preserve the same path, logging, output, artifact, and no-silent-merge controls.

## Rollback

Keep each approved engine version, configuration digest, label snapshot digest, model/calibrator/ranker manifests, and run manifest. To roll back:

1. stop new runs;
2. restore the last approved package and constraints file;
3. restore the last approved local configuration and artifact set;
4. rerun the synthetic smoke test;
5. verify artifact digests and environment doctor status;
6. record the rollback event under the local change-control process.

## Build the local handoff package

After the bootstrap and synthetic smoke test pass:

```bash
python scripts/build_local_handoff.py --project-root .
```

The builder refuses to replace a non-empty `dist/` directory. After reviewing that directory
and explicitly approving replacement, rerun with `--replace-build-output`.

The command creates ignored local outputs:

```text
dist/SHA256SUMS
artifacts/reports/dependency-audit.json
artifacts/reports/software-bill-of-materials.cdx.json
artifacts/reports/local_handoff_manifest.json
```

Review these before moving the package into an authorised restricted-analysis environment. They describe software dependencies and package files only; they must never include local source records, completed project configuration, adjudication values, candidate pairs, or linkage outputs.
