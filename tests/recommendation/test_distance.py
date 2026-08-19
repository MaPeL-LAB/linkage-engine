from __future__ import annotations

import pytest

from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
)
from mapel_linkage.configuration import compile_config, load_config
from mapel_linkage.profiling import build_preflight_task_profile
from mapel_linkage.recommendation.distance import (
    MetaFeatureDistanceComputer,
    TaskMetaFeatureVector,
    extract_family_meta_features,
)
from tests.helpers import EXAMPLE_CONFIG, ROOT


def test_meta_feature_extraction_from_preflight_profile() -> None:
    loaded = load_config(EXAMPLE_CONFIG)
    plan = compile_config(loaded.config, project_root=ROOT)
    profile = build_preflight_task_profile(plan)

    vec = TaskMetaFeatureVector.from_profile(profile)

    assert vec.linkage_mode == "link_only"
    assert vec.assignment_constraint == "one_to_one"
    assert vec.label_volume_class == "sparse"
    assert 0.0 <= vec.record_count_log_ratio <= 1.0
    assert 0.0 <= vec.entropy_estimate <= 1.0
    assert 0.0 <= vec.variable_count_scale <= 1.0
    assert 0.0 <= vec.comparison_count_scale <= 1.0
    assert 0.0 <= vec.blocking_rule_count_scale <= 1.0


def test_meta_feature_extraction_from_latent_spec() -> None:
    spec = ScenarioLatentSpec(
        family_id="family.typo_stress",
        instance_id="instance.typo_high",
        typo_rate=0.3,
        token_transposition_rate=0.3,
        date_shift_rate=0.1,
        missingness_rate=0.05,
        label_volume=500,
    )
    vec = TaskMetaFeatureVector.from_latent_spec(spec)

    assert vec.linkage_mode == "link_only"
    assert vec.label_volume_class == "dense"
    assert vec.error_estimate_approx > 0.3
    assert vec.label_volume_scale > 0.5
    assert vec.missingness_mean == 0.05


def test_distance_computer_properties() -> None:
    computer = MetaFeatureDistanceComputer()

    v1 = TaskMetaFeatureVector(
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        label_volume_class="sparse",
        record_count_log_ratio=0.5,
        missingness_mean=0.1,
        missingness_max=0.2,
        entropy_estimate=0.5,
        candidate_edge_budget_scale=0.4,
        error_estimate_approx=0.2,
        label_volume_scale=0.5,
        variable_count_scale=0.2,
        comparison_count_scale=0.2,
        blocking_rule_count_scale=0.1,
    )

    # 1. Identity
    assert computer.compute_distance(v1, v1, metric="gower") == 0.0
    assert computer.compute_distance(v1, v1, metric="weighted_euclidean") == 0.0

    v2 = TaskMetaFeatureVector(
        linkage_mode="dedupe_only",
        assignment_constraint="unconstrained",
        label_volume_class="none",
        record_count_log_ratio=0.5,
        missingness_mean=0.5,
        missingness_max=0.8,
        entropy_estimate=0.1,
        candidate_edge_budget_scale=0.8,
        error_estimate_approx=0.8,
        label_volume_scale=0.0,
        variable_count_scale=0.8,
        comparison_count_scale=0.8,
        blocking_rule_count_scale=0.5,
    )

    # 2. Symmetry
    d_gower_12 = computer.compute_distance(v1, v2, metric="gower")
    d_gower_21 = computer.compute_distance(v2, v1, metric="gower")
    assert pytest.approx(d_gower_12) == d_gower_21
    assert 0.0 < d_gower_12 <= 1.0

    d_euc_12 = computer.compute_distance(v1, v2, metric="weighted_euclidean")
    d_euc_21 = computer.compute_distance(v2, v1, metric="weighted_euclidean")
    assert pytest.approx(d_euc_12) == d_euc_21
    assert 0.0 < d_euc_12 <= 1.0


def test_find_nearest_families() -> None:
    gen = BenchmarkScenarioGenerator()
    family_vectors = extract_family_meta_features(gen)
    computer = MetaFeatureDistanceComputer()

    # Query with a typo-heavy link_only vector
    query_vec = TaskMetaFeatureVector(
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        label_volume_class="sparse",
        record_count_log_ratio=0.5,
        missingness_mean=0.0,
        missingness_max=0.0,
        entropy_estimate=0.6,
        candidate_edge_budget_scale=0.4,
        error_estimate_approx=0.25,
        label_volume_scale=0.3,
        variable_count_scale=0.15,
        comparison_count_scale=0.15,
        blocking_rule_count_scale=0.1,
    )

    nearest = computer.find_nearest_families(query_vec, family_vectors, k=3)
    assert len(nearest) == 3
    distances = [dist for _, dist in nearest]
    assert distances == sorted(distances)
    assert all(0.0 <= d <= 1.0 for d in distances)
