"""Tests for the parametric synthetic benchmark scenario generator."""

from __future__ import annotations

from mapel_linkage.benchmarking.generator import (
    BenchmarkScenarioGenerator,
    ScenarioLatentSpec,
)
from mapel_linkage.profiling.contracts import LabelEvidenceClass


def test_standard_families_and_instances_registration() -> None:
    generator = BenchmarkScenarioGenerator()
    families = generator.list_families()
    instances = generator.list_instances()

    assert len(families) >= 10
    assert len(instances) >= 15

    family_ids = {fam.family_id for fam in families}
    assert "family.typo_stress" in family_ids
    assert "family.missingness_regime" in family_ids
    assert "family.date_variation" in family_ids
    assert "family.frequency_skew" in family_ids
    assert "family.duplicate_density" in family_ids
    assert "family.label_scarcity" in family_ids
    assert "family.dedupe_only" in family_ids
    assert "family.multi_source" in family_ids
    assert "family.composite_realistic" in family_ids
    assert "family.held_out_transliteration" in family_ids

    # Verify held-out family
    translit = generator.get_family("family.held_out_transliteration")
    assert translit.prospectively_held_out is True
    assert len(translit.family_digest) == 64


def test_scenario_instance_manifest_and_profile_digests() -> None:
    generator = BenchmarkScenarioGenerator()
    inst = generator.get_instance("instance.typo_low")

    assert inst.family_id == "family.typo_stress"
    assert inst.instance_id == "instance.typo_low"
    assert len(inst.family_digest) == 64
    assert len(inst.latent_parameter_manifest_digest) == 64
    assert len(inst.observable_profile_digest) == 64
    assert len(inst.instance_digest) == 64


def test_typo_stress_generation() -> None:
    generator = BenchmarkScenarioGenerator()
    bundle_low = generator.generate("instance.typo_low", seed=42)
    bundle_high = generator.generate("instance.typo_high", seed=42)

    assert "source_a" in bundle_low.datasets
    assert "source_b" in bundle_low.datasets
    assert len(bundle_low.datasets["source_a"]) > 0
    assert len(bundle_low.datasets["source_b"]) > 0
    assert len(bundle_low.truth) > 0

    # High typo rate should have more corrupted strings in source_b than low
    src_a_lookup = {
        r.record_key.replace("A", ""): r.label_value
        for r in bundle_high.datasets["source_a"]
        if r.label_value
    }
    diff_count_high = sum(
        1
        for r in bundle_high.datasets["source_b"]
        if (
            r.label_value
            and r.record_key.replace("B", "") in src_a_lookup
            and r.label_value != src_a_lookup[r.record_key.replace("B", "")]
        )
    )
    src_a_low = {
        r.record_key.replace("A", ""): r.label_value
        for r in bundle_low.datasets["source_a"]
        if r.label_value
    }
    diff_count_low = sum(
        1
        for r in bundle_low.datasets["source_b"]
        if (
            r.label_value
            and r.record_key.replace("B", "") in src_a_low
            and r.label_value != src_a_low[r.record_key.replace("B", "")]
        )
    )
    assert diff_count_high > diff_count_low


def test_missingness_regime_generation() -> None:
    generator = BenchmarkScenarioGenerator()
    bundle_zero = generator.generate("instance.missing_zero", seed=101)
    bundle_high = generator.generate("instance.missing_high", seed=101)

    # Missing zero has no null values in source_a
    assert all(r.label_value is not None for r in bundle_zero.datasets["source_a"])
    # Missing high has null values
    null_count = sum(r.label_value is None for r in bundle_high.datasets["source_a"]) + sum(
        r.label_value is None for r in bundle_high.datasets["source_b"]
    )
    assert null_count > 0


def test_frequency_skew_zipfian() -> None:
    generator = BenchmarkScenarioGenerator()
    bundle_zipf = generator.generate("instance.zipf_extreme", seed=202)

    groups = [r.group_value for r in bundle_zipf.datasets["source_a"] if r.group_value]
    from collections import Counter

    counts = Counter(groups)
    most_common_count = counts.most_common(1)[0][1]
    least_common_count = counts.most_common()[-1][1]
    # In extreme Zipfian skew, top group frequency dominates
    assert most_common_count > least_common_count * 2


def test_label_scarcity_dimensions() -> None:
    generator = BenchmarkScenarioGenerator()

    bundle_zero = generator.generate("instance.labels_zero", seed=303)
    assert bundle_zero.task_profile.verified_labels_available is False
    assert bundle_zero.task_profile.label_evidence_class == LabelEvidenceClass.NONE

    bundle_dense = generator.generate("instance.labels_dense", seed=303)
    assert bundle_dense.task_profile.verified_labels_available is True
    assert bundle_dense.task_profile.label_evidence_class == LabelEvidenceClass.SYNTHETIC_TRUTH


def test_linkage_modes_bundles() -> None:
    generator = BenchmarkScenarioGenerator()

    # 1. Dedupe only
    bundle_dedupe = generator.generate("instance.dedupe_standard", seed=404)
    assert set(bundle_dedupe.datasets.keys()) == {"source_a"}
    assert bundle_dedupe.task_profile.linkage_mode == "dedupe_only"

    # 2. Multi source
    bundle_multi = generator.generate("instance.tri_source_standard", seed=505)
    assert set(bundle_multi.datasets.keys()) == {"source_a", "source_b", "source_c"}
    assert bundle_multi.task_profile.linkage_mode == "multi_source"


def test_custom_family_registration() -> None:
    generator = BenchmarkScenarioGenerator()
    fam = generator.register_family(
        family_id="family.custom_test",
        mechanism_tags=("character_substitution", "custom_tag"),
        prospectively_held_out=True,
        instances=(
            ScenarioLatentSpec(
                family_id="family.custom_test",
                instance_id="instance.custom_test_01",
                typo_rate=0.40,
            ),
        ),
    )
    assert fam.family_id == "family.custom_test"
    inst = generator.get_instance("instance.custom_test_01")
    assert inst.instance_id == "instance.custom_test_01"
