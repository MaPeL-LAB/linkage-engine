#!/usr/bin/env python3
"""Apply the bounded plural configuration patch before one-use CI verification."""

from __future__ import annotations

from pathlib import Path


def replace_slice(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


models_path = Path("src/mapel_linkage/configuration/models.py")
text = models_path.read_text(encoding="utf-8")

new_models = '''class StackingModelConfig(ConfigNode):
    enabled: StrictBool = False
    implementation: Literal["stacking_logistic"]
    model_id: Identifier
    base_model_ids: Annotated[tuple[Identifier, ...], Field(min_length=2, max_length=16)]
    require_verified_labels: Literal[True] = True
    meta_training_source: Literal["out_of_fold_training_predictions"] = (
        "out_of_fold_training_predictions"
    )
    maximum_training_pairs: Annotated[PositiveInt, Field(le=10_000_000)] = 1_000_000
    deterministic_mode: Literal[True] = True

    @model_validator(mode="after")
    def validate_base_model_ids(self) -> Self:
        if len(self.base_model_ids) != len(set(self.base_model_ids)):
            raise ValueError("Stacking base-model IDs must be unique.")
        if self.model_id in self.base_model_ids:
            raise ValueError("A stacking model cannot include itself as a base model.")
        return self


class ModelPortfolioConfig(ConfigNode):
    portfolio_schema_version: Literal["1"] = "1"
    portfolio_id: Identifier
    pair_model_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]
    ranking_model_ids: Annotated[tuple[Identifier, ...], Field(max_length=8)] = ()
    mandatory_baseline_id: Identifier
    maximum_challengers: Annotated[StrictInt, Field(ge=0, le=8)] = 3
    allow_shadow_scoring: Literal[True] = True
    test_partition_may_select_portfolio: Literal[False] = False

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if len(self.pair_model_ids) != len(set(self.pair_model_ids)):
            raise ValueError("Portfolio pair-model IDs must be unique.")
        if len(self.ranking_model_ids) != len(set(self.ranking_model_ids)):
            raise ValueError("Portfolio ranking-model IDs must be unique.")
        if self.mandatory_baseline_id not in self.pair_model_ids:
            raise ValueError("The mandatory baseline must be selected by the portfolio.")
        if self.maximum_challengers > len(self.pair_model_ids) - 1:
            raise ValueError("Portfolio challenger budget exceeds selected challengers.")
        return self


class ModelsConfig(ConfigNode):
    fellegi_sunter: FellegiSunterModelConfig
    boosted_tree: BoostedTreeModelConfig | None = None
    ranking: RankingModelConfig | None = None
    neural: NeuralModelConfig | None = None
    boosted_trees: Annotated[tuple[BoostedTreeModelConfig, ...], Field(max_length=16)] = ()
    ranking_models: Annotated[tuple[RankingModelConfig, ...], Field(max_length=8)] = ()
    neural_models: Annotated[tuple[NeuralModelConfig, ...], Field(max_length=8)] = ()
    ensembles: Annotated[tuple[StackingModelConfig, ...], Field(max_length=8)] = ()
    portfolio: ModelPortfolioConfig | None = None

    @model_validator(mode="after")
    def validate_portfolio_contract(self) -> Self:
        model_ids = self.all_model_ids()
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Model IDs must be unique across the complete portfolio.")
        base_pair_models = {
            self.fellegi_sunter.model_id: self.fellegi_sunter.enabled,
            **{model.model_id: model.enabled for model in self.all_boosted_trees()},
            **{model.model_id: model.enabled for model in self.all_neural_models()},
        }
        for ensemble in self.ensembles:
            if not set(ensemble.base_model_ids).issubset(base_pair_models):
                raise ValueError("Stacking base models must exist in the same project.")
            if ensemble.enabled and not all(
                base_pair_models[model_id] for model_id in ensemble.base_model_ids
            ):
                raise ValueError("Enabled stacking requires enabled base models.")
        if self.portfolio is None:
            return self
        if self.portfolio.mandatory_baseline_id != self.fellegi_sunter.model_id:
            raise ValueError("The portfolio baseline must be the Fellegi-Sunter model.")
        enabled_pair_ids = {
            model_id for model_id, enabled in base_pair_models.items() if enabled
        }
        enabled_pair_ids.update(ensemble.model_id for ensemble in self.ensembles if ensemble.enabled)
        enabled_ranker_ids = {
            model.model_id for model in self.all_ranking_models() if model.enabled
        }
        if not set(self.portfolio.pair_model_ids).issubset(enabled_pair_ids):
            raise ValueError("Portfolio pair models must be declared and enabled.")
        if not set(self.portfolio.ranking_model_ids).issubset(enabled_ranker_ids):
            raise ValueError("Portfolio rankers must be declared and enabled.")
        return self

    def all_boosted_trees(self) -> tuple[BoostedTreeModelConfig, ...]:
        models = list(self.boosted_trees)
        if self.boosted_tree is not None:
            models.insert(0, self.boosted_tree)
        return tuple(models)

    def all_ranking_models(self) -> tuple[RankingModelConfig, ...]:
        models = list(self.ranking_models)
        if self.ranking is not None:
            models.insert(0, self.ranking)
        return tuple(models)

    def all_neural_models(self) -> tuple[NeuralModelConfig, ...]:
        models = list(self.neural_models)
        if self.neural is not None:
            models.insert(0, self.neural)
        return tuple(models)

    def all_model_ids(self) -> tuple[str, ...]:
        identifiers = [self.fellegi_sunter.model_id]
        identifiers.extend(model.model_id for model in self.all_boosted_trees())
        identifiers.extend(model.model_id for model in self.all_ranking_models())
        identifiers.extend(model.model_id for model in self.all_neural_models())
        identifiers.extend(model.model_id for model in self.ensembles)
        return tuple(identifiers)
'''
text = replace_slice(
    text,
    "class ModelsConfig(ConfigNode):\n",
    "class CalibrationConfig(ConfigNode):\n",
    new_models + "\n\nclass CalibrationConfig(ConfigNode):\n",
)

unique_start = text.index("        model_ids = [self.models.fellegi_sunter.model_id]\n")
unique_end_marker = '        groups["model"] = model_ids\n'
unique_end = text.index(unique_end_marker, unique_start) + len(unique_end_marker)
text = (
    text[:unique_start]
    + '        groups["model"] = list(self.models.all_model_ids())\n'
    + text[unique_end:]
)

budget_start = text.index("        boosted = self.models.boosted_tree\n")
budget_end = text.index("        self._validate_outputs(set(variable_by_id))\n", budget_start)
new_budget = '''        if any(
            model.enabled
            and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.all_boosted_trees()
        ):
            raise ValueError("Boosted-tree training cannot exceed the runtime pair budget.")
        if any(
            model.enabled
            and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.all_ranking_models()
        ):
            raise ValueError("Ranking training cannot exceed the runtime pair budget.")
        if any(
            model.enabled
            and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.ensembles
        ):
            raise ValueError("Ensemble training cannot exceed the runtime pair budget.")
'''
text = text[:budget_start] + new_budget + text[budget_end:]

validate_start = text.index("    def _validate_models(self) -> None:\n")
validate_end = text.index(
    "    def _validate_outputs(self, variable_ids: set[str]) -> None:\n",
    validate_start,
)
new_validate = '''    def _validate_models(self) -> None:
        eligible_truth = self.labels is not None and self.labels.source.kind in {
            "synthetic_truth",
            "verified_human_adjudication",
            "verified_gold_standard",
        }
        supervised = [
            *self.models.all_boosted_trees(),
            *self.models.all_ranking_models(),
            *self.models.all_neural_models(),
            *self.models.ensembles,
        ]
        if any(
            model.enabled and model.require_verified_labels and not eligible_truth
            for model in supervised
        ):
            raise ValueError("Enabled supervised models require eligible verified labels.")
        score_model_ids: set[str] = set()
        if self.models.fellegi_sunter.enabled:
            score_model_ids.add(self.models.fellegi_sunter.model_id)
        score_model_ids.update(
            model.model_id for model in self.models.all_boosted_trees() if model.enabled
        )
        score_model_ids.update(
            model.model_id for model in self.models.all_neural_models() if model.enabled
        )
        score_model_ids.update(
            model.model_id for model in self.models.ensembles if model.enabled
        )
        if self.models.portfolio is not None:
            score_model_ids.intersection_update(self.models.portfolio.pair_model_ids)
        if not score_model_ids:
            raise ValueError("At least one pair-scoring model must be enabled.")
        if self.calibration.source_model == "selected_champion" and len(score_model_ids) < 2:
            raise ValueError("Champion selection requires at least two enabled pair models.")
        if (
            self.calibration.source_model != "selected_champion"
            and self.calibration.source_model not in score_model_ids
        ):
            raise ValueError(
                "Calibration source_model must identify an enabled pair model or selected_champion."
            )

'''
text = text[:validate_start] + new_validate + text[validate_end:]
models_path.write_text(text, encoding="utf-8")

init_path = Path("src/mapel_linkage/pipeline/__init__.py")
init_text = init_path.read_text(encoding="utf-8")
import_marker = "from mapel_linkage.pipeline.synthetic_vertical_slice import SyntheticVerticalSliceRunner\n"
artifact_import = '''from mapel_linkage.pipeline.stage_artifacts import (
    OutOfFoldPredictionManifest,
    StageArtifactLedger,
    StageArtifactRef,
)
'''
if artifact_import not in init_text:
    init_text = init_text.replace(import_marker, artifact_import + import_marker)
all_marker = '    "ModelPortfolioDeclaration",\n'
additions = '''    "OutOfFoldPredictionManifest",
    "StageArtifactLedger",
    "StageArtifactRef",
'''
if '    "StageArtifactRef",\n' not in init_text:
    init_text = init_text.replace(all_marker, all_marker + additions)
init_path.write_text(init_text, encoding="utf-8")

for filename in (
    "src/mapel_linkage/__init__.py",
    "pyproject.toml",
    "tests/test_package_metadata.py",
):
    path = Path(filename)
    value = path.read_text(encoding="utf-8").replace("0.2.0.dev1", "0.2.0.dev2")
    path.write_text(value, encoding="utf-8")

readme = Path("README.md")
readme_text = readme.read_text(encoding="utf-8")
marker = "## Command line\n"
addition = '''## Plural model configuration and stage provenance

Project configuration can declare multiple boosted, ranking, neural, and stacking candidates while retaining the existing singular fields for compatibility. Immutable stage-artifact and out-of-fold manifests provide digest-linked provenance without exposing rows, identifiers, candidate pairs, or local paths. See [`docs/implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md`](docs/implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md).

'''
if addition not in readme_text:
    readme_text = readme_text.replace(marker, addition + marker)
readme.write_text(readme_text, encoding="utf-8")

changelog = Path("CHANGELOG.md")
changelog_text = changelog.read_text(encoding="utf-8")
marker = "## [Unreleased]\n"
addition = '''
### Added — plural model configuration and stage provenance

- Backward-compatible plural boosted, ranking, neural, and stacking model declarations with a versioned bounded portfolio selection.
- Validation of model IDs, stacking base-model availability, enabled portfolio members, challenger limits, and supervised-label eligibility.
- Immutable stage-artifact references and ordered lineage ledgers with restricted row-level enforcement.
- Aggregate-only out-of-fold prediction manifests that prohibit test, calibration, and decision partition use.
- Synthetic-only configuration, lineage, privacy, and provenance tests.

'''
if addition.strip() not in changelog_text:
    changelog_text = changelog_text.replace(marker, marker + addition)
changelog.write_text(changelog_text, encoding="utf-8")

docs_index = Path("docs/README.md")
docs_text = docs_index.read_text(encoding="utf-8")
marker = "- [`implementation/MODEL_PORTFOLIO_AND_RECIPE_FOUNDATION.md`](implementation/MODEL_PORTFOLIO_AND_RECIPE_FOUNDATION.md)\n"
addition = marker + "- [`implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md`](implementation/PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md)\n"
if "PLURAL_CONFIGURATION_AND_STAGE_ARTIFACTS.md" not in docs_text:
    docs_text = docs_text.replace(marker, addition)
docs_index.write_text(docs_text, encoding="utf-8")
