# Agent Instructions

## Canonical names

- Developer / organisation: `MaPeL-LAB`
- Repository: `linkage-engine`
- Distribution: `mapel-linkage-engine`
- Import package: `mapel_linkage`
- CLI: `mapel-linkage`

Do not rename the repository to `mapel-linkage-engine`.

## Privacy boundary

Never request, inspect, generate from, upload, commit, quote, paste, print, or log real participant or operational record data. Use synthetic data exclusively in source-controlled files, tests, examples, notebooks, CI, issues, pull requests, and agent conversations.

Real data, identifiers, configurations, adjudication records, candidate pairs, secrets, models, and outputs remain local and ignored. De-identified, masked, hashed, tokenised, sampled, or perturbed real data are not repository-safe synthetic data.

Never log record values, identifiers, candidate pairs, comparison values, training rows, secrets, or adjudication values.

## Truth and model authority

- Unknown pairs are not nonmatches.
- Unverified crosswalks are not training truth.
- Deterministic anchors are evidence-only by default.
- A ranker may output rank and top-K only.
- Pair models score evidence; assignment enforces constraints; decision policy emits the relationship status.
- No component may silently merge records.

## Configuration safety

Configuration is data, not code. Prohibit raw SQL, `eval`, `exec`, imports, dotted callable paths, shell commands, and unrestricted plugins. Only package-owned allow-list registries and typed DSL nodes are executable.

## Working method

1. Inspect repository and relevant ADRs.
2. Define a bounded change and checks.
3. Use the smallest complete synthetic slice.
4. Run applicable formatting, typing, tests, privacy, security, and end-to-end checks.
5. Report files changed, verification, limitations, and next step.
6. Never claim real-world validation from synthetic tests.
7. Do not create repositories, change remotes/visibility, publish, commit, push, or merge without explicit authorisation.

## Source-column rule

Dataset-specific source column names may appear only in project configuration, synthetic fixture generation, and IO-mapping tests. They must not appear in model, comparison, assignment, calibration, decision, or orchestration logic.

## M1 configuration invariant

- Add configuration behavior through typed Pydantic nodes and immutable package registries only.
- Do not resolve modules, functions, plugins, SQL, or shell content from project configuration.
- Keep configuration values and arbitrary mapping keys out of public validation errors.
- A configured filesystem root must also fit within the trusted host path envelope.
- Regenerate `schemas/linkage-config.schema.json` after every configuration-model change.
- Maintain parity between the committed schema and `LinkageConfig.model_json_schema()`.
- Use `SafeLogEvent`/`build_safe_log_event`; do not add free-form row-bearing log calls.
- Keep row-bearing dataclass representations value-hidden.
