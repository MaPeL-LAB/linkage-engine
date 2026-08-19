"""Materialize the checksum-controlled I2A source batch and update repository metadata."""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath

PAYLOAD_DIGEST = "b7b1f9c472d0dd2d49eceb638f850c7f848ffea00f383307e819a4f89705bd1e"
PARTS_DIRECTORY = Path("scripts/i2a_payload_parts")
ERRORS_PATH = Path("src/mapel_linkage/domain/errors.py")


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"Expected one update marker in {path}.")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def materialize_payload() -> None:
    parts = sorted(PARTS_DIRECTORY.glob("part-*.b64"))
    if not parts:
        raise SystemExit("No I2A transfer parts were found.")
    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise SystemExit("The I2A transfer payload is not valid Base64.") from error
    if hashlib.sha256(payload).hexdigest() != PAYLOAD_DIGEST:
        raise SystemExit("The I2A transfer payload checksum does not match.")

    live_errors = ERRORS_PATH.read_text(encoding="utf-8")
    if "class AdvisorError(" in live_errors:
        raise SystemExit("The live error registry already contains AdvisorError.")

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise SystemExit("The I2A transfer payload is empty.")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit("The I2A transfer payload contains an unsafe path.")
            if not member.isfile():
                raise SystemExit("The I2A transfer payload may contain regular files only.")
        archive.extractall(path=Path.cwd(), filter="data")

    advisor_error = (
        "\n\nclass AdvisorError(LinkageRuntimeError):\n"
        '    """Raised by advisory-only eligibility and recommendation boundaries."""\n'
    )
    ERRORS_PATH.write_text(live_errors.rstrip() + advisor_error + "\n", encoding="utf-8")


def apply_repository_updates() -> None:
    replace_once(
        "pyproject.toml",
        'version = "0.2.0.dev2"',
        'version = "0.2.0.dev3"',
    )
    replace_once(
        "src/mapel_linkage/__init__.py",
        '__version__ = "0.2.0.dev2"',
        '__version__ = "0.2.0.dev3"',
    )
    replace_once(
        "src/mapel_linkage/capabilities.py",
        '''    Capability(
        "linkage_strategy_advisor",
        "Linkage Strategy Advisor",
        "I2",
        ComponentStatus.PLANNED,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.NOT_VERIFIED,
        "Advisory-only pipeline recommendation with coverage checks and abstention.",
    ),
''',
        '''    Capability(
        "stage1_linkage_strategy_advisor",
        "Stage-1 Linkage Strategy Advisor",
        "I2A",
        ComponentStatus.IMPLEMENTED,
        WorkflowStatus.INTEGRATED,
        RuntimeVerificationStatus.CORE_CI,
        "Configuration-only profiling, hard eligibility, structural Pareto shortlisting, "
        "transparent explanations, and explicit empirical abstention.",
    ),
    Capability(
        "synthetic_benchmark_registry",
        "Synthetic Benchmark Registry",
        "B1",
        ComponentStatus.PARTIAL,
        WorkflowStatus.COMPONENT_ONLY,
        RuntimeVerificationStatus.CORE_CI,
        "Versioned aggregate family, instance, replicate, run, failure, and snapshot "
        "contracts exist; corpus generation and portfolio execution remain pending.",
    ),
    Capability(
        "linkage_strategy_advisor",
        "Evidence-Based Linkage Strategy Advisor",
        "I2B-I2D",
        ComponentStatus.PLANNED,
        WorkflowStatus.NOT_INTEGRATED,
        RuntimeVerificationStatus.NOT_VERIFIED,
        "Nearest-family retrieval, OOD detection, learned meta-ranking, and active "
        "benchmark planning remain evidence-gated future stages.",
    ),
''',
    )

    readme_section = '''## Stage-1 Linkage Strategy Advisor

The advisory-only Stage-1 workflow can build a privacy-safe preflight task profile, apply hard
lifecycle and runtime eligibility rules, retain the mandatory Fellegi-Sunter baseline, construct
a structural Pareto shortlist, explain every applied rule, and abstain from empirical ranking
while the benchmark registry is empty.

```text
mapel-linkage profile-job --config CONFIG --project-root ROOT
mapel-linkage recommend-pipeline --config CONFIG --project-root ROOT
```

A recommendation is not a `PipelineRecipeArtifact`, cannot approve a model, cannot use the locked
test partition, and has no identity, assignment, threshold, or merge authority. See
[`docs/architecture/ADR-0005-LINKAGE-STRATEGY-ADVISOR.md`](
docs/architecture/ADR-0005-LINKAGE-STRATEGY-ADVISOR.md).

'''
    replace_once("README.md", "## Command line\n", readme_section + "## Command line\n")

    changelog_section = '''### Added — I2A Stage-1 Linkage Strategy Advisor

- Staged privacy-safe preflight, candidate-graph, and evidence task-profile contracts.
- Lifecycle-aware hard eligibility rules that distinguish training, calibration, approved
  inference, shadow scoring, and benchmark planning.
- Structural Pareto shortlisting with mandatory-baseline retention, family diversity,
  transparent applied-rule explanations, and explicit abstention from empirical ranking.
- A separate immutable `PipelineRecommendation` contract with advisory-only authority and
  prohibited automatic promotion.
- Aggregate benchmark family, instance, replicate, run, failure, and registry-snapshot
  contracts that retain unsuccessful executions as evidence.
- Architecture, experimental-design, evidence-hierarchy, profile, registry, policy, and advisor
  validation documentation.
- `profile-job` and `recommend-pipeline` CLI commands that emit aggregate JSON only.

'''
    replace_once(
        "CHANGELOG.md",
        "## [Unreleased]\n\n",
        "## [Unreleased]\n\n" + changelog_section,
    )


if __name__ == "__main__":
    materialize_payload()
    apply_repository_updates()
