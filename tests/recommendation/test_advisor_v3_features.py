from __future__ import annotations

import pytest
from pydantic import ValidationError

from mapel_linkage.benchmarking.advisor_v3_catalogue import build_advisor_v3_generator
from mapel_linkage.benchmarking.advisor_v3_features import (
    AdvisorV3MechanismProfile,
    advisor_v3_feature_source_policy_digest,
    build_advisor_v3_mechanism_profile,
)
from mapel_linkage.benchmarking.generator import ScenarioLatentSpec
from mapel_linkage.recommendation.distance_v3 import (
    AdvisorV3FeatureUnavailableError,
    MechanismAwareMetaFeatureDistanceComputer,
    MechanismAwareTaskMetaFeatureVector,
    advisor_v3_feature_model_schema_digest,
    select_advisor_v3_ood_distance_threshold,
)
from mapel_linkage.recommendation.qualification_v3 import (
    AdvisorV3QualificationPolicy,
    advisor_v3_evaluation_algorithm_digest,
)
from mapel_linkage.synthetic.generator import SyntheticRecord


def _vector(*, script_variation_rate: float = 0.0) -> MechanismAwareTaskMetaFeatureVector:
    return MechanismAwareTaskMetaFeatureVector(
        linkage_mode="link_only",
        assignment_constraint="one_to_one",
        label_volume_class="sparse",
        script_variation_rate=script_variation_rate,
        punctuation_variation_rate=0.0,
        tokenization_variation_rate=0.0,
        missingness_mean=0.1,
        missingness_asymmetry=0.0,
        frequency_concentration=0.2,
        candidate_ambiguity_scale=0.2,
        duplicate_signature_rate=0.0,
        planned_training_label_budget_scale=0.5,
    )


def test_v3_missing_mechanism_features_require_abstention_or_fallback() -> None:
    generator = build_advisor_v3_generator()
    profile = generator.build_task_profile("instance.advisor_v3.f001.p01")
    unavailable = AdvisorV3MechanismProfile.unavailable()

    with pytest.raises(AdvisorV3FeatureUnavailableError, match="abstain"):
        MechanismAwareTaskMetaFeatureVector.from_profiles(profile, unavailable)
    assert unavailable.script_variation_rate is None
    assert unavailable.complete is False


def test_v3_rejects_latent_only_feature_shortcut() -> None:
    with pytest.raises(AdvisorV3FeatureUnavailableError, match="Latent simulator"):
        MechanismAwareTaskMetaFeatureVector.from_latent_spec(
            ScenarioLatentSpec(family_id="family.synthetic", instance_id="instance.synthetic")
        )


def test_v3_feature_schema_has_exact_profile_vector_parity() -> None:
    generator = build_advisor_v3_generator()
    instance_id = "instance.advisor_v3.f001.p01"
    bundle = generator.generate(instance_id, seed=20260816)
    mechanism = generator.build_advisor_v3_mechanism_profile(
        instance_id, seed=20260816, generated_bundle=bundle
    )
    vector = MechanismAwareTaskMetaFeatureVector.from_profiles(bundle.task_profile, mechanism)

    assert tuple(
        name for name in type(mechanism).model_fields if name in vector.CONTINUOUS_FEATURES
    ) == (vector.CONTINUOUS_FEATURES)
    assert set(vector.to_dict()) == set((*vector.CATEGORICAL_FEATURES, *vector.CONTINUOUS_FEATURES))
    assert len(advisor_v3_feature_model_schema_digest()) == 64
    assert mechanism.contains_truth_values is False
    assert mechanism.contains_outcomes is False


def test_v3_duplicate_rate_is_within_dataset_not_cross_source() -> None:
    generator = build_advisor_v3_generator()
    task_profile = generator.build_task_profile("instance.advisor_v3.f001.p01")
    left = SyntheticRecord("A000001", "beka-00001", "2000-01-01", "G01")
    right = SyntheticRecord("B000001", "beka-00001", "2000-01-01", "G01")
    profile = build_advisor_v3_mechanism_profile(
        datasets={"left": (left,), "right": (right,)},
        task_profile=task_profile,
        planned_training_label_budget=25,
    )

    assert profile.duplicate_signature_rate == 0.0
    assert profile.candidate_ambiguity_scale == 1.0


def test_v3_feature_source_policy_and_exact_observable_formulas_are_frozen() -> None:
    generator = build_advisor_v3_generator()
    task_profile = generator.build_task_profile("instance.advisor_v3.f001.p01")
    duplicate_left = SyntheticRecord("A000001", "Ána-Smith", "2000-01-01", "G01")
    profile = build_advisor_v3_mechanism_profile(
        datasets={
            "left": (
                duplicate_left,
                SyntheticRecord("A000002", "Ána-Smith", "2000-01-01", "G01"),
                SyntheticRecord("A000003", None, "2001-01-01", "G02"),
            ),
            "right": (
                SyntheticRecord("B000001", "Ána-Smith", "2000-01-01", "G01"),
                SyntheticRecord("B000002", "Name.One Two", "2002-01-01", "G03"),
            ),
        },
        task_profile=task_profile,
        planned_training_label_budget=25,
    )

    assert advisor_v3_feature_source_policy_digest() == (
        "a82a000c41e001f43bf12aefc19da2cfa5711048f3830028319bccde3524da9d"
    )
    assert profile.script_variation_rate == pytest.approx(0.75)
    assert profile.punctuation_variation_rate == pytest.approx(0.25)
    assert profile.tokenization_variation_rate == pytest.approx(0.0)
    assert profile.missingness_mean == pytest.approx(1.0 / 6.0)
    assert profile.missingness_asymmetry == pytest.approx(1.0 / 3.0)
    assert profile.frequency_concentration == pytest.approx(0.6)
    assert profile.candidate_ambiguity_scale == pytest.approx((1.0 / 3.0) ** 0.5)
    assert profile.duplicate_signature_rate == pytest.approx(0.4)


def test_v3_profile_cannot_claim_complete_with_missing_feature() -> None:
    payload = AdvisorV3MechanismProfile.unavailable().model_dump(mode="json")
    payload["complete"] = True
    with pytest.raises(ValidationError, match="completeness"):
        AdvisorV3MechanismProfile.model_validate(payload)


def test_v3_ood_threshold_uses_only_fixed_geometry_and_detects_axis_shift() -> None:
    training = {
        f"train.{index}": _vector(script_variation_rate=index * 0.005) for index in range(4)
    }
    conformal = {"conformal.1": _vector(script_variation_rate=0.012)}
    shifted = _vector(script_variation_rate=1.0)
    threshold = select_advisor_v3_ood_distance_threshold(
        training_vectors=training,
        conformal_vectors=conformal,
    )
    distance = min(
        MechanismAwareMetaFeatureDistanceComputer().compute_distance(shifted, item)
        for item in training.values()
    )

    assert distance > threshold
    with pytest.raises(TypeError):
        select_advisor_v3_ood_distance_threshold(  # type: ignore[call-arg]
            training_vectors=training,
            conformal_vectors=conformal,
            distance_computer=MechanismAwareMetaFeatureDistanceComputer(weights={}),
        )


def test_v3_policy_and_complete_evaluator_specification_are_fixed() -> None:
    policy = AdvisorV3QualificationPolicy()
    assert policy.learning_curve_family_counts == (12, 24, 36, 48)
    assert policy.performance_thresholds_relaxed_from_v2 is False
    assert policy.ood_distance_policy_status == "training_conformal_geometry_only"
    assert policy.evaluation_algorithm_digest == advisor_v3_evaluation_algorithm_digest()
    assert policy.operational_validity == "not_established"
    with pytest.raises(ValidationError, match="prospectively fixed"):
        AdvisorV3QualificationPolicy(minimum_ood_detection_rate=0.5)
