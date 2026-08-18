from __future__ import annotations

import hashlib
from typing import cast

import numpy as np

from mapel_linkage.validation import evaluate_stratified_pair_performance


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_stratified_pair_metrics_report_missingness_and_candidate_size() -> None:
    report = evaluate_stratified_pair_performance(
        labels=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8),
        probabilities=np.asarray([0.05, 0.40, 0.70, 0.96, 0.10, 0.85]),
        missingness_patterns=(
            "pattern_000",
            "pattern_001",
            "pattern_000",
            "pattern_000",
            "pattern_001",
            "pattern_010",
        ),
        candidate_set_sizes=(1, 4, 4, 12, 12, 7),
        diagnostic_threshold=0.5,
        partition_manifest_digest=digest("test-partition"),
    )
    summary = report.safe_summary()
    missingness_strata = cast(list[dict[str, object]], summary["missingness_pattern_strata"])
    candidate_size_strata = cast(list[dict[str, object]], summary["candidate_set_size_strata"])
    assert report.pair_count == 6
    assert {item["stratum"] for item in missingness_strata} == {
        "pattern_000",
        "pattern_001",
        "pattern_010",
    }
    assert {item["stratum"] for item in candidate_size_strata} == {
        "size_1",
        "size_2_to_5",
        "size_6_to_10",
        "size_11_plus",
    }
