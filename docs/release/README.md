# M8 Release Hardening

M8 defines reproducible private-candidate packaging and release evidence. Phase 1 installs
fail-closed controls; it does not authorize publication, deployment, real-data execution, model
promotion, or operational use.

The machine-readable authority is
[`RELEASE_READINESS_POLICY.json`](RELEASE_READINESS_POLICY.json). Its current status is `blocked`,
publication and deployment authority are `none`, and operational validity is `not_established`.

## Phase 1 controls

- [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md)
- [`API_STABILITY_POLICY.md`](API_STABILITY_POLICY.md)
- [`ARTIFACT_MIGRATION_POLICY.md`](ARTIFACT_MIGRATION_POLICY.md)
- [`SECURITY_AND_DEPENDENCY_REVIEW.md`](SECURITY_AND_DEPENDENCY_REVIEW.md)
- [`MODEL_CARDS.md`](MODEL_CARDS.md)
- [`ERROR_CODE_CATALOGUE.md`](ERROR_CODE_CATALOGUE.md)
- [`PRIVATE_RELEASE_AND_ROLLBACK.md`](PRIVATE_RELEASE_AND_ROLLBACK.md)
- [`SCALE_BENCHMARK_EVIDENCE_V2.md`](SCALE_BENCHMARK_EVIDENCE_V2.md)
- [`SCALE_BENCHMARK_POLICY.md`](SCALE_BENCHMARK_POLICY.md)

Verify the controls without claiming release readiness:

```bash
python scripts/generate_error_code_catalogue.py --check
python scripts/verify_release_readiness.py --expect-blocked
```

Calling the verifier without `--expect-blocked` fails while any release blocker remains. That
failure is deliberate and must not be bypassed by changing a status string without its required
evidence and human approval.
