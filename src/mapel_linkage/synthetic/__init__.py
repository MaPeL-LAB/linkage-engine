"""Deterministic synthetic data generation for tests and examples."""

from __future__ import annotations

from mapel_linkage.synthetic.generator import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    SyntheticProvenance,
    SyntheticRecord,
    SyntheticTruthRecord,
    generate_synthetic_bundle,
    write_synthetic_bundle,
)

__all__ = [
    "SyntheticBundle",
    "SyntheticGenerationConfig",
    "SyntheticProvenance",
    "SyntheticRecord",
    "SyntheticTruthRecord",
    "generate_synthetic_bundle",
    "write_synthetic_bundle",
]
