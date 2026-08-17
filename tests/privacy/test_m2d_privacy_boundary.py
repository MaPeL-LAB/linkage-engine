from __future__ import annotations

from mapel_linkage.models import (
    FellegiSunterComparisonParameters,
    FellegiSunterLevelParameters,
    FellegiSunterModelArtifact,
)


def test_model_artifact_repr_hides_comparison_identifiers_and_contains_no_pairs() -> None:
    sentinel = "SYNTHETIC-SENSITIVE-COMPARISON-ID"
    parameters = FellegiSunterComparisonParameters(
        comparison_id=sentinel,
        levels=(
            FellegiSunterLevelParameters(0, 0.5, 0.5, 0.0),
            FellegiSunterLevelParameters(1, 0.5, 0.5, 0.0),
        ),
    )
    artifact = FellegiSunterModelArtifact(
        model_id="fs_baseline",
        model_version="m2d-reference-v1",
        prior_probability=0.1,
        random_seed=7,
        u_sample_pair_count=10,
        em_candidate_pair_count=4,
        em_iterations=2,
        converged=True,
        smoothing=0.5,
        convergence_tolerance=0.0001,
        feature_schema_digest="a" * 64,
        parameter_digest="b" * 64,
        comparisons={sentinel: parameters},
    )

    rendered = repr(artifact)
    assert sentinel not in rendered
    assert "left_record_key" not in rendered
    assert "right_record_key" not in rendered
    assert "model_posterior_uncalibrated" in rendered
    assert "evidence_only" in rendered
