from __future__ import annotations

from collections import Counter

import pytest

from mapel_linkage.benchmarking.advisor_catalogue import (
    AdvisorCorpusReadinessManifest,
    BenchmarkShardPlan,
    advisor_v2_family_roles,
    build_advisor_corpus_design,
    build_advisor_corpus_readiness,
    build_advisor_v2_generator,
    build_benchmark_shard_plan,
)
from mapel_linkage.benchmarking.generator import BenchmarkScenarioGenerator


def test_seed_v1_ids_and_digests_remain_stable() -> None:
    generator = BenchmarkScenarioGenerator()
    family_digests = {item.family_id: item.family_digest for item in generator.list_families()}
    instance_digests = {
        item.instance_id: item.instance_digest for item in generator.list_instances()
    }

    assert len(family_digests) == 10
    assert len(instance_digests) == 19
    assert family_digests["family.held_out_transliteration"] == (
        "6a2104ec6c7000d1c42487d7046207d4f7c5cdf72963440766230b86e7a896dc"
    )
    assert instance_digests["instance.transliteration_base"] == (
        "73d48a442268394752fdfea34faea5c9637e1b4aa0d57b4e60c68d1ad777da85"
    )
    assert not any(item.startswith("family.advisor_v2.") for item in family_digests)


def test_advisor_v2_design_has_prospective_family_roles_and_exact_scale() -> None:
    first = build_advisor_v2_generator()
    second = build_advisor_v2_generator()
    roles = advisor_v2_family_roles()
    counts = Counter(role for _, role in roles)

    assert len(first.list_families()) == 74
    assert len(first.list_instances()) == 299
    assert len(roles) == 64
    assert counts == {
        "meta_training": 40,
        "conformal": 8,
        "locked_evaluation": 8,
        "ood_holdout": 8,
    }
    assert [item.family_digest for item in first.list_families()] == [
        item.family_digest for item in second.list_families()
    ]
    assert [item.instance_digest for item in first.list_instances()] == [
        item.instance_digest for item in second.list_instances()
    ]

    design = build_advisor_corpus_design()
    assert design.family_count == 64
    assert design.instance_count == 280
    assert design.synthetic_only is True
    assert design.recommendation_authority == "advisory_only"
    assert design.decision_authority == "none"
    assert design.assignment_authority == "none"
    assert design.merge_authority == "none"
    assert design.automatic_promotion == "prohibited"
    assert design.operational_validity == "not_established"


def test_advisor_v2_ood_uses_real_cross_script_mechanics() -> None:
    generator = build_advisor_v2_generator()
    bundle = generator.generate("instance.advisor_v2.f57.p06", seed=20260816)
    left = bundle.datasets["source_a"]
    right = bundle.datasets["source_b"]

    assert len(left) == len(right)
    assert any(
        source.label_value != target.label_value
        and target.label_value is not None
        and any(ord(character) > 127 for character in target.label_value)
        for source, target in zip(left, right, strict=True)
    )


def test_readiness_fails_closed_until_all_required_real_adapters_exist() -> None:
    incomplete = build_advisor_corpus_readiness(
        adapter_statuses={
            "recipe.fellegi_sunter_reference": "success_capable",
            "recipe.xgboost_classifier": "success_capable",
            "recipe.xgboost_ranker": "ineligible",
        }
    )
    complete = build_advisor_corpus_readiness(
        adapter_statuses={
            "recipe.fellegi_sunter_reference": "success_capable",
            "recipe.xgboost_classifier": "success_capable",
            "recipe.xgboost_ranker": "success_capable",
            "recipe.single_source_dedupe": "ineligible",
        }
    )

    assert incomplete.execution_ready is False
    assert incomplete.success_capable_required_adapter_count == 2
    assert complete.execution_ready is True
    assert complete.readiness_schema_version == "2"
    assert complete.execution_protocol_id == "advisor_corpus_execution_v2"
    assert complete.expected_run_count == 9_800
    assert complete.required_evidence_cell_count == 1_400
    assert complete.expected_required_adapter_run_count == 4_200
    assert complete.missing_required_adapter_run_count == 4_200
    assert complete.advisor_evidence_ready is False
    assert complete.success_capable_required_adapter_count == 3
    assert complete.incomplete_mode_adapters == ("dedupe_only", "multi_source")
    assert complete.recommendation_authority == "advisory_only"
    assert complete.decision_authority == "none"
    assert complete.assignment_authority == "none"
    assert complete.merge_authority == "none"
    assert complete.automatic_promotion == "prohibited"


def test_readiness_requires_every_required_adapter_in_every_replicate_cell() -> None:
    initial = build_advisor_corpus_readiness(
        adapter_statuses={
            "recipe.fellegi_sunter_reference": "success_capable",
            "recipe.xgboost_classifier": "success_capable",
            "recipe.xgboost_ranker": "success_capable",
        },
        planned_replicates_per_instance=5,
    )
    complete_payload = {
        **initial.model_dump(mode="json"),
        "execution_status": "complete",
        "completed_run_count": initial.expected_run_count,
        "successful_evidence_cell_count": initial.required_evidence_cell_count,
        "successful_required_adapter_run_count": (initial.expected_required_adapter_run_count),
        "failed_required_adapter_run_count": 0,
        "missing_required_adapter_run_count": 0,
        "successful_overlap_family_count": 64,
        "advisor_evidence_ready": True,
    }
    complete = AdvisorCorpusReadinessManifest.model_validate(complete_payload)
    assert complete.advisor_evidence_ready is True

    incomplete_payload = {
        **complete_payload,
        "successful_evidence_cell_count": initial.required_evidence_cell_count - 1,
        "successful_required_adapter_run_count": (initial.expected_required_adapter_run_count - 1),
        "failed_required_adapter_run_count": 1,
    }
    with pytest.raises(ValueError, match="fail closed"):
        AdvisorCorpusReadinessManifest.model_validate(incomplete_payload)

    incomplete_payload["advisor_evidence_ready"] = False
    incomplete = AdvisorCorpusReadinessManifest.model_validate(incomplete_payload)
    assert incomplete.failed_required_adapter_run_count == 1
    assert incomplete.advisor_evidence_ready is False


def test_shard_plan_is_deterministic_balanced_and_collision_free() -> None:
    first = build_benchmark_shard_plan(shard_count=32)
    second = build_benchmark_shard_plan(shard_count=32)
    ids = [instance_id for shard in first.shards for instance_id in shard.instance_ids]

    assert first == second
    assert first.plan_digest == second.plan_digest
    assert len(ids) == len(set(ids)) == 280
    assert max(map(len, (item.instance_ids for item in first.shards))) == 9
    assert min(map(len, (item.instance_ids for item in first.shards))) == 8

    payload = first.model_dump(mode="json")
    payload["shards"][1]["instance_ids"][0] = payload["shards"][0]["instance_ids"][0]
    with pytest.raises(ValueError, match="exactly once"):
        BenchmarkShardPlan.model_validate(payload)
