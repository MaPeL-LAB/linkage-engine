# Security and Dependency Review

## Automated controls

The `quality` CI job currently:

- uses read-only repository contents permission;
- pins GitHub Actions to immutable commit SHAs;
- installs through the Python 3.12 constraints file;
- runs repository privacy and manifest verification;
- runs Ruff, strict mypy, pre-commit, and the complete tests;
- runs strict `pip-audit` against installed third-party dependencies;
- generates a CycloneDX JSON software bill of materials in temporary CI storage;
- builds and inspects the wheel and source distribution for restricted content.

The private local handoff builder repeats repository verification, tests, distribution inspection,
dependency audit, SBOM creation, checksums, and an aggregate handoff manifest. These are supply-chain
and software-integrity controls; they are not an external security assessment.

## Required Phase 2 review

| Domain | Required evidence | Current status |
|---|---|---|
| dependency vulnerabilities | strict audit for the exact candidate constraints | automated per CI; candidate sign-off pending |
| software bill of materials | candidate-bound CycloneDX document and digest | generated locally/CI; retained sign-off pending |
| package contents | wheel and sdist restricted-file inspection | automated |
| filesystem isolation | path traversal, symbolic-link, and output-root tests | automated; host controls remain external |
| configuration execution | strict typed DSL and package-owned registries | automated |
| secrets and CI token | read-only token, no protected dataset processing | automated configuration review |
| model privacy | restricted local artifacts and no unsafe untrusted pickle load | internal controls; external review pending |
| threat model | architecture/privacy review against intended deployment | external review pending |

## Residual boundaries

DuckDB is not a sandbox. The package does not provide host encryption, malware protection,
identity and access management, network isolation, backup, deletion, or incident response.
Operational models may encode information from their training data and remain restricted local
artifacts. Real data, completed configurations, candidate pairs, adjudications, models, and outputs
must never enter repository CI or source control.

`external_security_review_not_approved` remains a release blocker until a named human review owns
the intended private deployment envelope and records an approval outside the repository.
