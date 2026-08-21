"""Parametric scenario generator for the Linkage Engine synthetic benchmark library."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from mapel_linkage.benchmarking.contracts import (
    ScenarioFamilyManifest,
    ScenarioInstanceManifest,
)
from mapel_linkage.profiling.contracts import (
    CountBand,
    LabelEvidenceClass,
    PreflightTaskProfile,
    ProfileScope,
    VariableTypeCount,
)
from mapel_linkage.synthetic.generator import (
    SyntheticRecord,
    SyntheticTruthRecord,
)

_SYLLABLES = (
    "beka",
    "daro",
    "feni",
    "guma",
    "hilo",
    "jari",
    "keto",
    "luma",
    "mavi",
    "noro",
    "peta",
    "riso",
    "suna",
    "tavi",
    "welo",
    "zari",
)


def _digest_object(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ScenarioLatentSpec(BaseModel):
    """Latent parameters governing generation mechanics for one instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: StrictStr
    instance_id: StrictStr
    entity_count: Annotated[StrictInt, Field(ge=8, le=100_000)] = 120
    linkage_mode: Literal["link_only", "dedupe_only", "multi_source"] = "link_only"
    typo_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    token_transposition_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    date_shift_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    date_ambiguity_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    missingness_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    informative_missingness: StrictBool = False
    zipf_skew_parameter: Annotated[StrictFloat, Field(ge=0.0, le=5.0)] = 0.0
    duplicate_density: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    label_volume: Annotated[StrictInt, Field(ge=0, le=100_000)] = 100
    planned_replicates: Annotated[StrictInt, Field(ge=1, le=10_000)] = 5


class ScenarioMechanicExtension(BaseModel):
    """Versioned mechanics added without changing legacy seed-v1 instance digests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanic_schema_version: Literal["2"] = "2"
    unicode_transliteration_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0
    punctuation_change_rate: Annotated[StrictFloat, Field(ge=0.0, le=1.0)] = 0.0


@dataclass(frozen=True, slots=True, repr=False)
class BenchmarkScenarioBundle:
    """Generated datasets, truth records, and profile for one scenario instance realization."""

    family_id: str
    instance_id: str
    seed: int
    family_manifest: ScenarioFamilyManifest
    instance_manifest: ScenarioInstanceManifest
    task_profile: PreflightTaskProfile
    datasets: dict[str, tuple[SyntheticRecord, ...]] = field(repr=False)
    truth: tuple[SyntheticTruthRecord, ...] = field(repr=False)
    latent_parameters: dict[str, Any] = field(repr=False)

    def __repr__(self) -> str:
        sizes = {name: len(records) for name, records in self.datasets.items()}
        return (
            f"BenchmarkScenarioBundle(family={self.family_id}, "
            f"instance={self.instance_id}, seed={self.seed}, "
            f"datasets={sizes}, truth_count={len(self.truth)})"
        )


def _mutate_typo(value: str, rng: random.Random) -> str:
    if len(value) < 2:
        return value + "x"
    characters = list(value)
    op = rng.choice(("substitute", "insert", "delete", "transpose"))
    if op == "substitute":
        pos = rng.randrange(len(characters))
        characters[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    elif op == "insert":
        pos = rng.randrange(len(characters) + 1)
        characters.insert(pos, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif op == "delete" and len(characters) > 2:
        pos = rng.randrange(len(characters))
        del characters[pos]
    else:  # transpose
        pos = rng.randrange(len(characters) - 1)
        characters[pos], characters[pos + 1] = characters[pos + 1], characters[pos]
    return "".join(characters)


def _mutate_token_transposition(value: str, rng: random.Random) -> str:
    parts = value.split("-")
    if len(parts) >= 2 and rng.random() < 0.8:
        return f"{parts[1]}-{parts[0]}"
    tokens = value.split(" ")
    if len(tokens) >= 2:
        tokens[0], tokens[1] = tokens[1], tokens[0]
        return " ".join(tokens)
    return _mutate_typo(value, rng)


_TRANSLITERATED_SYLLABLES: Mapping[str, str] = {
    "beka": "μπέκα",
    "daro": "даро",
    "feni": "фени",
    "guma": "гума",
    "hilo": "χίλο",
    "jari": "джари",
    "keto": "κέτο",
    "luma": "лума",
    "mavi": "мави",
    "noro": "норо",
    "peta": "πέτα",
    "riso": "рисо",
    "suna": "суна",
    "tavi": "тави",
    "welo": "βέλο",
    "zari": "зари",
}


def _mutate_unicode_transliteration(value: str) -> str:
    """Apply a deterministic cross-script synthetic transliteration mechanic."""

    prefix, separator, suffix = value.partition("-")
    translated = _TRANSLITERATED_SYLLABLES.get(prefix, prefix)
    return f"{translated}{separator}{suffix}" if separator else translated


def _mutate_punctuation(value: str) -> str:
    """Change separators without adding any source-derived values."""

    return re.sub(r"[- ]+", "'", value)


def _sample_zipf_index(num_elements: int, s: float, rng: random.Random) -> int:
    if s <= 0.0 or num_elements <= 1:
        return rng.randrange(num_elements)
    weights = [1.0 / ((i + 1) ** s) for i in range(num_elements)]
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if upto >= r:
            return i
    return num_elements - 1


class BenchmarkScenarioGenerator:
    """Instantiates scenario families, parameterised instances, and synthetic datasets."""

    def __init__(self) -> None:
        self._families: dict[str, tuple[ScenarioFamilyManifest, dict[str, Any]]] = {}
        self._instances: dict[str, tuple[ScenarioInstanceManifest, ScenarioLatentSpec]] = {}
        self._instance_extensions: dict[str, ScenarioMechanicExtension] = {}
        self._register_standard_corpus()

    def _register_standard_corpus(self) -> None:
        # 1. Typo Stress Family
        self.register_family(
            family_id="family.typo_stress",
            mechanism_tags=("character_substitution", "token_transposition"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.typo_stress",
                    instance_id="instance.typo_low",
                    typo_rate=0.05,
                    token_transposition_rate=0.05,
                ),
                ScenarioLatentSpec(
                    family_id="family.typo_stress",
                    instance_id="instance.typo_moderate",
                    typo_rate=0.15,
                    token_transposition_rate=0.15,
                ),
                ScenarioLatentSpec(
                    family_id="family.typo_stress",
                    instance_id="instance.typo_high",
                    typo_rate=0.30,
                    token_transposition_rate=0.30,
                ),
            ),
        )

        # 2. Missingness Regime Family
        self.register_family(
            family_id="family.missingness_regime",
            mechanism_tags=("source_specific_missingness", "informative_missingness"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.missingness_regime",
                    instance_id="instance.missing_zero",
                    missingness_rate=0.0,
                ),
                ScenarioLatentSpec(
                    family_id="family.missingness_regime",
                    instance_id="instance.missing_low",
                    missingness_rate=0.10,
                ),
                ScenarioLatentSpec(
                    family_id="family.missingness_regime",
                    instance_id="instance.missing_high",
                    missingness_rate=0.30,
                    informative_missingness=True,
                ),
            ),
        )

        # 3. Date Variation Family
        self.register_family(
            family_id="family.date_variation",
            mechanism_tags=("date_shift", "date_ambiguity"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.date_variation",
                    instance_id="instance.date_shift_low",
                    date_shift_rate=0.10,
                    date_ambiguity_rate=0.05,
                ),
                ScenarioLatentSpec(
                    family_id="family.date_variation",
                    instance_id="instance.date_shift_high",
                    date_shift_rate=0.30,
                    date_ambiguity_rate=0.20,
                ),
            ),
        )

        # 4. Frequency Skew Family (Zipfian)
        self.register_family(
            family_id="family.frequency_skew",
            mechanism_tags=("frequency_skew", "common_value_collisions"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.frequency_skew",
                    instance_id="instance.zipf_moderate",
                    zipf_skew_parameter=1.0,
                ),
                ScenarioLatentSpec(
                    family_id="family.frequency_skew",
                    instance_id="instance.zipf_extreme",
                    zipf_skew_parameter=1.8,
                ),
            ),
        )

        # 5. Duplicate Density Family
        self.register_family(
            family_id="family.duplicate_density",
            mechanism_tags=("within_source_duplication", "shared_household_attributes"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.duplicate_density",
                    instance_id="instance.duplicate_low",
                    duplicate_density=0.05,
                ),
                ScenarioLatentSpec(
                    family_id="family.duplicate_density",
                    instance_id="instance.duplicate_high",
                    duplicate_density=0.25,
                ),
            ),
        )

        # 6. Label Scarcity Family
        self.register_family(
            family_id="family.label_scarcity",
            mechanism_tags=("label_scarcity", "label_noise"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.label_scarcity",
                    instance_id="instance.labels_zero",
                    label_volume=0,
                ),
                ScenarioLatentSpec(
                    family_id="family.label_scarcity",
                    instance_id="instance.labels_sparse",
                    label_volume=50,
                ),
                ScenarioLatentSpec(
                    family_id="family.label_scarcity",
                    instance_id="instance.labels_dense",
                    label_volume=500,
                ),
            ),
        )

        # 7. Dedupe Only Family
        self.register_family(
            family_id="family.dedupe_only",
            mechanism_tags=("single_source_deduplication", "within_source_duplication"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.dedupe_only",
                    instance_id="instance.dedupe_standard",
                    linkage_mode="dedupe_only",
                    duplicate_density=0.20,
                ),
            ),
        )

        # 8. Multi Source Family
        self.register_family(
            family_id="family.multi_source",
            mechanism_tags=("multi_source_resolution", "graph_contradictions"),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.multi_source",
                    instance_id="instance.tri_source_standard",
                    linkage_mode="multi_source",
                ),
            ),
        )

        # 9. Composite Realistic Family
        self.register_family(
            family_id="family.composite_realistic",
            mechanism_tags=(
                "character_substitution",
                "date_shift",
                "source_specific_missingness",
                "within_source_duplication",
            ),
            prospectively_held_out=False,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.composite_realistic",
                    instance_id="instance.composite_base",
                    typo_rate=0.15,
                    date_shift_rate=0.15,
                    missingness_rate=0.10,
                    duplicate_density=0.10,
                ),
            ),
        )

        # 10. Held-Out Transliteration Family
        self.register_family(
            family_id="family.held_out_transliteration",
            mechanism_tags=("unicode_variation", "transliteration", "punctuation_changes"),
            prospectively_held_out=True,
            instances=(
                ScenarioLatentSpec(
                    family_id="family.held_out_transliteration",
                    instance_id="instance.transliteration_base",
                    typo_rate=0.20,
                    token_transposition_rate=0.20,
                ),
            ),
        )

    def register_family(
        self,
        *,
        family_id: str,
        mechanism_tags: tuple[str, ...],
        prospectively_held_out: bool = False,
        instances: tuple[ScenarioLatentSpec, ...],
        instance_extensions: Mapping[str, ScenarioMechanicExtension] | None = None,
    ) -> ScenarioFamilyManifest:
        latent_scenario_payload = {
            "family_id": family_id,
            "mechanism_tags": mechanism_tags,
            "prospectively_held_out": prospectively_held_out,
            "instance_count": len(instances),
        }
        extensions = dict(instance_extensions or {})
        if set(extensions) - {spec.instance_id for spec in instances}:
            raise ValueError("Scenario mechanic extensions must bind registered instances.")
        if extensions:
            latent_scenario_payload["mechanic_schema_version"] = "2"
            latent_scenario_payload["mechanic_extension_digest"] = _digest_object(
                {
                    instance_id: extension.model_dump(mode="json")
                    for instance_id, extension in sorted(extensions.items())
                }
            )
        latent_scenario_digest = _digest_object(latent_scenario_payload)
        family_manifest = ScenarioFamilyManifest(
            family_id=family_id,
            mechanism_tags=mechanism_tags,
            latent_scenario_manifest_digest=latent_scenario_digest,
            prospectively_held_out=prospectively_held_out,
        )
        self._families[family_id] = (family_manifest, latent_scenario_payload)

        for spec in instances:
            profile = self.build_task_profile(spec)
            latent_param_payload = spec.model_dump(mode="json")
            extension = extensions.get(spec.instance_id)
            if extension is not None:
                latent_param_payload["mechanic_extension"] = extension.model_dump(mode="json")
            instance_manifest = ScenarioInstanceManifest(
                family_id=family_id,
                instance_id=spec.instance_id,
                family_digest=family_manifest.family_digest,
                latent_parameter_manifest_digest=_digest_object(latent_param_payload),
                observable_profile_digest=profile.profile_digest,
                planned_replicates=spec.planned_replicates,
            )
            self._instances[spec.instance_id] = (instance_manifest, spec)
            if extension is not None:
                self._instance_extensions[spec.instance_id] = extension

        return family_manifest

    def get_family(self, family_id: str) -> ScenarioFamilyManifest:
        if family_id not in self._families:
            raise KeyError(f"Unknown scenario family ID: {family_id}")
        return self._families[family_id][0]

    def list_families(self) -> tuple[ScenarioFamilyManifest, ...]:
        return tuple(fam[0] for fam in self._families.values())

    def get_instance(self, instance_id: str) -> ScenarioInstanceManifest:
        if instance_id not in self._instances:
            raise KeyError(f"Unknown scenario instance ID: {instance_id}")
        return self._instances[instance_id][0]

    def get_latent_spec(self, instance_id: str) -> ScenarioLatentSpec:
        """Return the immutable package-owned simulator specification for planning only.

        Latent values remain outside persisted benchmark manifests and advisor feature
        vectors. This accessor lets experimental-design code stratify the package-owned
        synthetic catalogue without generating or exposing record-level material.
        """
        if instance_id not in self._instances:
            raise KeyError(f"Unknown scenario instance ID: {instance_id}")
        return self._instances[instance_id][1]

    def list_instances(self, family_id: str | None = None) -> tuple[ScenarioInstanceManifest, ...]:
        if family_id is not None:
            return tuple(
                inst[0] for inst in self._instances.values() if inst[0].family_id == family_id
            )
        return tuple(inst[0] for inst in self._instances.values())

    def build_task_profile(
        self, spec_or_instance_id: ScenarioLatentSpec | str
    ) -> PreflightTaskProfile:
        if isinstance(spec_or_instance_id, str):
            spec = self._instances[spec_or_instance_id][1]
        else:
            spec = spec_or_instance_id

        assignment_constraint: Literal["one_to_one", "many_to_one", "one_to_many", "unconstrained"]
        if spec.linkage_mode == "link_only":
            dataset_count = 2
            source_count = 1
            target_count = 1
            assignment_constraint = "one_to_one"
        elif spec.linkage_mode == "dedupe_only":
            dataset_count = 1
            source_count = 1
            target_count = 0
            assignment_constraint = "unconstrained"
        else:  # multi_source
            dataset_count = 3
            source_count = 3
            target_count = 0
            assignment_constraint = "unconstrained"

        has_labels = spec.label_volume > 0
        evidence_class = (
            LabelEvidenceClass.SYNTHETIC_TRUTH if has_labels else LabelEvidenceClass.NONE
        )

        variable_types = (
            VariableTypeCount(data_type="categorical", count=1),
            VariableTypeCount(data_type="date", count=1),
            VariableTypeCount(data_type="string", count=1),
        )

        return PreflightTaskProfile(
            profile_scope=ProfileScope.GLOBAL_SYNTHETIC,
            linkage_mode=spec.linkage_mode,
            assignment_constraint=assignment_constraint,
            dataset_count=dataset_count,
            source_count=source_count,
            target_count=target_count,
            reference_count=0,
            auxiliary_count=0,
            variable_count=3,
            variable_type_counts=variable_types,
            restricted_variable_count=0,
            transformation_count=0,
            blocking_rule_count=1,
            comparison_count=3,
            record_count_band=CountBand.NOT_OBSERVED,
            candidate_pair_budget_band=CountBand.SMALL,
            label_evidence_class=evidence_class,
            verified_labels_available=has_labels,
        )

    def generate(
        self,
        instance_id: str,
        *,
        seed: int = 20260816,
    ) -> BenchmarkScenarioBundle:
        """Deterministically generate the dataset bundle for a scenario instance and seed."""
        if instance_id not in self._instances:
            raise KeyError(f"Unknown scenario instance ID: {instance_id}")

        instance_manifest, spec = self._instances[instance_id]
        extension = self._instance_extensions.get(instance_id)
        family_manifest = self._families[spec.family_id][0]
        profile = self.build_task_profile(spec)
        rng = random.Random(seed)

        bases: list[tuple[str, str, str, str, str]] = []
        for index in range(spec.entity_count):
            entity_key = f"E{index:06d}"
            household_key = f"H{index // 3:05d}"
            if spec.zipf_skew_parameter > 0.0:
                syl_idx = _sample_zipf_index(len(_SYLLABLES), spec.zipf_skew_parameter, rng)
                label = f"{_SYLLABLES[syl_idx]}-{index:05d}"
                group_idx = _sample_zipf_index(7, spec.zipf_skew_parameter, rng)
                group = f"G{group_idx:02d}"
            else:
                label = f"{rng.choice(_SYLLABLES)}-{index:05d}"
                group = f"G{index % 7:02d}"

            base_date = date(1980 + (index % 30), 1 + (index % 12), 1 + (index % 27))
            date_text = base_date.isoformat()
            bases.append((entity_key, household_key, label, date_text, group))

        datasets: dict[str, list[SyntheticRecord]] = {}
        truth: list[SyntheticTruthRecord] = []

        if spec.linkage_mode == "link_only":
            datasets["source_a"] = []
            datasets["source_b"] = []

            for index, (entity_key, household_key, label, date_text, group) in enumerate(bases):
                key_a = f"A{index:06d}"
                val_a = label
                if rng.random() < spec.missingness_rate:
                    val_a = None  # type: ignore[assignment]
                datasets["source_a"].append(
                    SyntheticRecord(
                        record_key=key_a,
                        label_value=val_a,
                        date_value=date_text,
                        group_value=group,
                    )
                )
                truth.append(SyntheticTruthRecord("source_a", key_a, entity_key, household_key))

                key_b = f"B{index:06d}"
                val_b = label
                if rng.random() < spec.typo_rate:
                    val_b = _mutate_typo(val_b, rng)
                if rng.random() < spec.token_transposition_rate:
                    val_b = _mutate_token_transposition(val_b, rng)
                if extension is not None:
                    if rng.random() < extension.unicode_transliteration_rate:
                        val_b = _mutate_unicode_transliteration(val_b)
                    if rng.random() < extension.punctuation_change_rate:
                        val_b = _mutate_punctuation(val_b)

                dt_b = date.fromisoformat(date_text)
                if rng.random() < spec.date_shift_rate:
                    dt_b += timedelta(days=rng.choice((-3, -2, -1, 1, 2, 3)))
                if rng.random() < spec.date_ambiguity_rate and dt_b.day <= 12:
                    dt_b = date(dt_b.year, dt_b.day, dt_b.month)
                dt_str = dt_b.isoformat()

                miss_b = spec.missingness_rate
                if spec.informative_missingness and group in ("G00", "G01"):
                    miss_b = min(1.0, miss_b * 2.0 + 0.2)
                if rng.random() < miss_b:
                    val_b = None  # type: ignore[assignment]
                if rng.random() < miss_b / 2:
                    dt_str = None  # type: ignore[assignment]

                datasets["source_b"].append(
                    SyntheticRecord(
                        record_key=key_b,
                        label_value=val_b,
                        date_value=dt_str,
                        group_value=group,
                    )
                )
                truth.append(SyntheticTruthRecord("source_b", key_b, entity_key, household_key))

            # Extra duplicates if configured
            dup_count = int(spec.duplicate_density * len(bases))
            for d_idx in range(dup_count):
                entity_key, household_key, label, date_text, group = bases[d_idx]
                key_dup = f"AD{d_idx:05d}"
                datasets["source_a"].append(
                    SyntheticRecord(
                        record_key=key_dup,
                        label_value=_mutate_typo(label, rng),
                        date_value=date_text,
                        group_value=group,
                    )
                )
                truth.append(SyntheticTruthRecord("source_a", key_dup, entity_key, household_key))

        elif spec.linkage_mode == "dedupe_only":
            datasets["source_a"] = []
            for index, (entity_key, household_key, label, date_text, group) in enumerate(bases):
                key = f"A{index:06d}"
                datasets["source_a"].append(
                    SyntheticRecord(
                        record_key=key,
                        label_value=label,
                        date_value=date_text,
                        group_value=group,
                    )
                )
                truth.append(SyntheticTruthRecord("source_a", key, entity_key, household_key))

            dup_count = max(4, int(spec.duplicate_density * len(bases)))
            for d_idx in range(dup_count):
                entity_key, household_key, label, date_text, group = bases[d_idx % len(bases)]
                key_dup = f"AD{d_idx:05d}"
                val_dup = _mutate_typo(label, rng) if rng.random() < 0.5 else label
                datasets["source_a"].append(
                    SyntheticRecord(
                        record_key=key_dup,
                        label_value=val_dup,
                        date_value=date_text,
                        group_value=group,
                    )
                )
                truth.append(SyntheticTruthRecord("source_a", key_dup, entity_key, household_key))

        else:  # multi_source
            for src_name in ("source_a", "source_b", "source_c"):
                datasets[src_name] = []

            for index, (entity_key, household_key, label, date_text, group) in enumerate(bases):
                for src_idx, src_name in enumerate(("source_a", "source_b", "source_c")):
                    prefix = ["A", "B", "C"][src_idx]
                    key = f"{prefix}{index:06d}"
                    val = label
                    if src_idx > 0 and rng.random() < 0.2:
                        val = _mutate_typo(val, rng)
                    datasets[src_name].append(
                        SyntheticRecord(
                            record_key=key,
                            label_value=val,
                            date_value=date_text,
                            group_value=group,
                        )
                    )
                    truth.append(SyntheticTruthRecord(src_name, key, entity_key, household_key))

        for name in datasets:
            datasets[name].sort(key=lambda item: item.record_key)
        truth.sort(key=lambda item: (item.dataset_id, item.record_key))

        return BenchmarkScenarioBundle(
            family_id=spec.family_id,
            instance_id=spec.instance_id,
            seed=seed,
            family_manifest=family_manifest,
            instance_manifest=instance_manifest,
            task_profile=profile,
            datasets={name: tuple(records) for name, records in datasets.items()},
            truth=tuple(truth),
            latent_parameters={
                **spec.model_dump(mode="json"),
                **(
                    {"mechanic_extension": extension.model_dump(mode="json")}
                    if extension is not None
                    else {}
                ),
            },
        )


__all__ = [
    "BenchmarkScenarioBundle",
    "BenchmarkScenarioGenerator",
    "ScenarioLatentSpec",
    "ScenarioMechanicExtension",
]
