"""Strict configuration schema for Linkage Engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

type Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]
type ColumnName = Annotated[StrictStr, Field(min_length=1, max_length=256)]
type PathText = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
type Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
type Fraction = Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]

type LinkageMode = Literal[
    "link_only",
    "dedupe_only",
    "link_and_dedupe",
    "multi_source",
]
type AssignmentConstraint = Literal[
    "one_to_one",
    "many_to_one",
    "one_to_many",
    "unconstrained",
]
type DataType = Literal[
    "string",
    "date",
    "integer",
    "float",
    "boolean",
    "categorical",
]


def _freeze_string_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


def _serialize_string_mapping(value: Mapping[str, str]) -> dict[str, str]:
    return dict(value)


type FrozenColumnMap = Annotated[
    Mapping[Identifier, ColumnName],
    AfterValidator(_freeze_string_mapping),
    PlainSerializer(_serialize_string_mapping, return_type=dict[str, str]),
]


class ConfigNode(BaseModel):
    """Base class for all immutable configuration nodes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class StripTransform(ConfigNode):
    kind: Literal["strip"]


class CasefoldTransform(ConfigNode):
    kind: Literal["casefold"]


class UnicodeNormalizeTransform(ConfigNode):
    kind: Literal["unicode_normalize"]
    form: Literal["NFC", "NFKC"] = "NFKC"


class CollapseWhitespaceTransform(ConfigNode):
    kind: Literal["collapse_whitespace"]


class ParseDateTransform(ConfigNode):
    kind: Literal["parse_date"]
    formats: Annotated[tuple[StrictStr, ...], Field(min_length=1, max_length=16)]


class NumericCastTransform(ConfigNode):
    kind: Literal["numeric_cast"]
    target: Literal["integer", "float"]


type TransformSpec = Annotated[
    StripTransform
    | CasefoldTransform
    | UnicodeNormalizeTransform
    | CollapseWhitespaceTransform
    | ParseDateTransform
    | NumericCastTransform,
    Field(discriminator="kind"),
]


class ExactPredicate(ConfigNode):
    kind: Literal["exact"]
    variable: Identifier


class PrefixEqualPredicate(ConfigNode):
    kind: Literal["prefix_equal"]
    variable: Identifier
    length: Annotated[PositiveInt, Field(le=128)]


class DateWindowPredicate(ConfigNode):
    kind: Literal["date_window"]
    variable: Identifier
    maximum_days: Annotated[PositiveInt, Field(le=36525)]


class AllPredicate(ConfigNode):
    kind: Literal["all"]
    terms: Annotated[tuple[BlockPredicate, ...], Field(min_length=1, max_length=32)]


class AnyPredicate(ConfigNode):
    kind: Literal["any"]
    terms: Annotated[tuple[BlockPredicate, ...], Field(min_length=1, max_length=32)]


type BlockPredicate = Annotated[
    ExactPredicate | PrefixEqualPredicate | DateWindowPredicate | AllPredicate | AnyPredicate,
    Field(discriminator="kind"),
]


class ExactComparison(ConfigNode):
    kind: Literal["exact"]


class JaroWinklerComparison(ConfigNode):
    kind: Literal["jaro_winkler"]


class LevenshteinComparison(ConfigNode):
    kind: Literal["levenshtein"]
    maximum_distance: Annotated[StrictInt, Field(ge=0, le=1024)] | None = None


class DamerauLevenshteinComparison(ConfigNode):
    kind: Literal["damerau_levenshtein"]
    maximum_distance: Annotated[StrictInt, Field(ge=0, le=1024)] | None = None


class QGramComparison(ConfigNode):
    kind: Literal["qgram"]
    q: Annotated[PositiveInt, Field(ge=1, le=8)] = 2


class DateDifferenceComparison(ConfigNode):
    kind: Literal["date_difference"]
    unit: Literal["day", "month", "year"] = "day"


class NumericDifferenceComparison(ConfigNode):
    kind: Literal["numeric_difference"]
    scale: Annotated[StrictFloat, Field(gt=0.0)] = 1.0


class CategoricalComparison(ConfigNode):
    kind: Literal["categorical"]


type ComparisonFunction = Annotated[
    ExactComparison
    | JaroWinklerComparison
    | LevenshteinComparison
    | DamerauLevenshteinComparison
    | QGramComparison
    | DateDifferenceComparison
    | NumericDifferenceComparison
    | CategoricalComparison,
    Field(discriminator="kind"),
]


class MissingLevel(ConfigNode):
    kind: Literal["missing"]


class ExactLevel(ConfigNode):
    kind: Literal["exact"]


class ThresholdLevel(ConfigNode):
    kind: Literal["threshold"]
    minimum: Probability


class MaximumDifferenceLevel(ConfigNode):
    kind: Literal["maximum_difference"]
    value: Annotated[StrictFloat | StrictInt, Field(ge=0)]


class ElseLevel(ConfigNode):
    kind: Literal["else"]


type ComparisonLevel = Annotated[
    MissingLevel | ExactLevel | ThresholdLevel | MaximumDifferenceLevel | ElseLevel,
    Field(discriminator="kind"),
]


class ProjectConfig(ConfigNode):
    project_id: Identifier
    entity_type: Identifier
    linkage_mode: LinkageMode
    assignment_constraint: AssignmentConstraint
    random_seed: Annotated[StrictInt, Field(ge=0, le=4294967295)]


class RuntimeConfig(ConfigNode):
    backend: Literal["duckdb"] = "duckdb"
    maximum_candidate_pairs: Annotated[PositiveInt, Field(le=10_000_000_000)]
    deterministic_mode: Literal[True] = True


class PrivacyConfig(ConfigNode):
    allowed_input_roots: Annotated[tuple[PathText, ...], Field(min_length=1, max_length=16)]
    allowed_output_roots: Annotated[tuple[PathText, ...], Field(min_length=1, max_length=16)]
    allow_remote_uris: Literal[False] = False
    allow_network_access: Literal[False] = False
    log_policy: Literal["aggregate_only"] = "aggregate_only"
    include_tracebacks: Literal[False] = False


class DatasetConfig(ConfigNode):
    id: Identifier
    role: Literal["source", "target", "reference", "auxiliary"]
    path: PathText
    format: Literal["parquet", "csv", "tsv", "jsonl", "duckdb"]
    record_id_column: ColumnName

    @model_validator(mode="after")
    def reject_reserved_columns(self) -> Self:
        if self.record_id_column.startswith("__ml_"):
            raise ValueError("Reserved internal column names cannot be source columns.")
        return self


class MissingnessConfig(ConfigNode):
    blank_is_missing: StrictBool = True
    comparison_policy: Literal["explicit_missing_level", "ignore_comparison"] = (
        "explicit_missing_level"
    )


class VariableConfig(ConfigNode):
    id: Identifier
    data_type: DataType
    source_columns: Annotated[FrozenColumnMap, Field(min_length=1)]
    normalisation: tuple[TransformSpec, ...] = ()
    missingness: MissingnessConfig = Field(default_factory=MissingnessConfig)
    restricted_output: StrictBool = False

    @model_validator(mode="after")
    def reject_reserved_columns(self) -> Self:
        if any(column.startswith("__ml_") for column in self.source_columns.values()):
            raise ValueError("Reserved internal column names cannot be source columns.")
        return self


class DeterministicAnchorConfig(ConfigNode):
    id: Identifier
    predicate: BlockPredicate
    require_unique_left: StrictBool = True
    require_unique_right: StrictBool = True
    action: Literal["evidence_only"] = "evidence_only"
    allow_as_training_truth: Literal[False] = False


class BlockingRuleConfig(ConfigNode):
    id: Identifier
    predicate: BlockPredicate


class BlockingConfig(ConfigNode):
    rules: Annotated[tuple[BlockingRuleConfig, ...], Field(min_length=1, max_length=128)]


class ComparisonConfig(ConfigNode):
    id: Identifier
    variable: Identifier
    function: ComparisonFunction
    levels: Annotated[tuple[ComparisonLevel, ...], Field(min_length=2, max_length=32)]

    @model_validator(mode="after")
    def validate_levels(self) -> Self:
        kinds = [level.kind for level in self.levels]
        if kinds[-1] != "else":
            raise ValueError("The final comparison level must be else.")
        if len(kinds) != len(set(kinds)) and any(
            kind in {"missing", "exact", "else"} and kinds.count(kind) > 1 for kind in kinds
        ):
            raise ValueError("Missing, exact, and else levels may occur at most once.")
        thresholds = [level.minimum for level in self.levels if isinstance(level, ThresholdLevel)]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("Threshold comparison levels must be unique.")
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError("Threshold comparison levels must be ordered from high to low.")
        differences = [
            float(level.value) for level in self.levels if isinstance(level, MaximumDifferenceLevel)
        ]
        if len(differences) != len(set(differences)):
            raise ValueError("Maximum-difference levels must be unique.")
        if differences != sorted(differences):
            raise ValueError("Maximum-difference levels must be ordered from low to high.")
        return self


class SyntheticTruthSource(ConfigNode):
    kind: Literal["synthetic_truth"]
    entity_group_columns: Annotated[FrozenColumnMap, Field(min_length=1)]
    household_group_columns: FrozenColumnMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_reserved_columns(self) -> Self:
        columns = (*self.entity_group_columns.values(), *self.household_group_columns.values())
        if any(column.startswith("__ml_") for column in columns):
            raise ValueError("Reserved internal column names cannot be truth columns.")
        return self


class VerifiedAdjudicationSource(ConfigNode):
    kind: Literal["verified_human_adjudication"]
    path: PathText
    protocol_version: Identifier


class VerifiedGoldStandardSource(ConfigNode):
    kind: Literal["verified_gold_standard"]
    path: PathText
    protocol_version: Identifier


class UnverifiedReferenceSource(ConfigNode):
    kind: Literal["unverified_reference"]
    path: PathText


type LabelSource = Annotated[
    SyntheticTruthSource
    | VerifiedAdjudicationSource
    | VerifiedGoldStandardSource
    | UnverifiedReferenceSource,
    Field(discriminator="kind"),
]


class LabelsConfig(ConfigNode):
    source: LabelSource
    permit_weak_labels_for_training: Literal[False] = False
    permit_unverified_crosswalk: Literal[False] = False


class FellegiSunterModelConfig(ConfigNode):
    enabled: StrictBool = True
    implementation: Literal["splink_duckdb"]
    model_id: Identifier
    probability_two_random_records_match: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)] = 0.0001
    u_max_pairs: Annotated[PositiveInt, Field(le=10_000_000_000)] = 1_000_000
    em_max_iterations: Annotated[PositiveInt, Field(le=1000)] = 25
    em_convergence: Annotated[StrictFloat, Field(gt=0.0, le=0.1)] = 0.0001
    probability_smoothing: Annotated[StrictFloat, Field(gt=0.0, le=100.0)] = 0.5
    estimate_u_by_random_sampling: Literal[True] = True
    estimate_m_by_em: Literal[True] = True
    term_frequency_adjustments: Literal[False] = False


class BoostedTreeModelConfig(ConfigNode):
    enabled: StrictBool = True
    implementation: Literal["xgboost_classifier", "lightgbm_classifier"]
    model_id: Identifier
    require_verified_labels: Literal[True] = True
    n_estimators: Annotated[PositiveInt, Field(le=5000)] = 300
    max_depth: Annotated[PositiveInt, Field(le=32)] = 5
    learning_rate: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 0.05
    subsample: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 1.0
    column_sample: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 1.0
    maximum_training_pairs: Annotated[PositiveInt, Field(le=10_000_000)] = 1_000_000
    hard_negative_fraction: Probability = 0.75
    n_jobs: Literal[1] = 1
    deterministic_mode: Literal[True] = True


class RankingModelConfig(ConfigNode):
    enabled: StrictBool = True
    implementation: Literal["xgboost_ranker", "lightgbm_ranker"]
    model_id: Identifier
    query_side: Literal["source", "target"]
    top_k: Annotated[PositiveInt, Field(le=1000)]
    require_verified_labels: Literal[True] = True
    n_estimators: Annotated[PositiveInt, Field(le=5000)] = 200
    max_depth: Annotated[PositiveInt, Field(le=32)] = 4
    learning_rate: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 0.05
    maximum_training_pairs: Annotated[PositiveInt, Field(le=10_000_000)] = 1_000_000
    n_jobs: Literal[1] = 1
    deterministic_mode: Literal[True] = True


class NeuralModelConfig(ConfigNode):
    enabled: StrictBool = False
    implementation: Literal["pytorch_pair_mlp"]
    model_id: Identifier = "neural_pair_matcher"
    require_verified_labels: Literal[True] = True


class StackingModelConfig(ConfigNode):
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
        enabled_pair_ids = {model_id for model_id, enabled in base_pair_models.items() if enabled}
        enabled_pair_ids.update(
            ensemble.model_id for ensemble in self.ensembles if ensemble.enabled
        )
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


class CalibrationConfig(ConfigNode):
    method: Literal["sigmoid", "isotonic", "beta"]
    source_model: Identifier
    partition: Literal["calibration"] = "calibration"
    require_independent_partition: Literal[True] = True


class ModelSelectionConfig(ConfigNode):
    mode: Literal["champion_challenger"] = "champion_challenger"
    selection_partition: Literal["validation"] = "validation"
    primary_metric: Literal["average_precision", "brier_score"] = "average_precision"
    test_partition_may_select_model: Literal[False] = False


class NoMatchAssignmentConfig(ConfigNode):
    enabled: Literal[True] = True
    utility: StrictFloat = 0.0


class AssignmentConfig(ConfigNode):
    solver: Literal["ortools_min_cost_flow", "unconstrained"]
    constraint: AssignmentConstraint
    no_match: NoMatchAssignmentConfig
    deterministic_tie_breaking: Literal[True] = True


class ConfirmedDecisionConfig(ConfigNode):
    minimum_probability: Probability
    minimum_probability_margin: Probability
    require_assignment: StrictBool = True
    require_valid_calibration: Literal[True] = True


class ReviewDecisionConfig(ConfigNode):
    minimum_probability: Probability


class NoMatchDecisionConfig(ConfigNode):
    maximum_top_probability: Probability
    require_complete_candidate_search: Literal[True] = True


class UnresolvedDecisionConfig(ConfigNode):
    fallback: Literal[True] = True


class DecisionPolicyConfig(ConfigNode):
    confirmed: ConfirmedDecisionConfig
    review_required: ReviewDecisionConfig
    no_match: NoMatchDecisionConfig
    unresolved: UnresolvedDecisionConfig

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if not (
            self.no_match.maximum_top_probability
            < self.review_required.minimum_probability
            <= self.confirmed.minimum_probability
        ):
            raise ValueError("Decision probability regions must not overlap.")
        return self


class SplitConfig(ConfigNode):
    method: Literal["entity_household_connected_components"]
    training_fraction: Fraction
    validation_fraction: Fraction
    calibration_fraction: Fraction
    decision_fraction: Fraction
    test_fraction: Fraction

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        total = sum(
            (
                self.training_fraction,
                self.validation_fraction,
                self.calibration_fraction,
                self.decision_fraction,
                self.test_fraction,
            )
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Validation split fractions must sum to one.")
        return self


class HardNegativeSamplingConfig(ConfigNode):
    enabled: StrictBool = True
    verified_nonmatches_only: Literal[True] = True


class ValidationConfig(ConfigNode):
    split: SplitConfig
    hard_negative_sampling: HardNegativeSamplingConfig
    candidate_recall_k: Annotated[tuple[PositiveInt, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_recall_k(self) -> Self:
        if tuple(sorted(set(self.candidate_recall_k))) != self.candidate_recall_k:
            raise ValueError("candidate_recall_k must be unique and ascending.")
        return self


type OutputField = Literal[
    "relationship_id",
    "source_dataset_id",
    "target_dataset_id",
    "source_record_ref",
    "target_record_ref",
    "relationship_status",
    "model_family",
    "model_version",
    "calibrated_probability",
    "candidate_rank",
    "probability_margin",
    "decision_rule_id",
    "assignment_method",
    "assignment_constraint",
    "anchor_rule_ids",
    "candidate_rule_ids",
    "review_reason_codes",
    "run_id",
    "configuration_digest",
    "feature_schema_digest",
    "non_sensitive_provenance",
    "created_at",
]


class OutputConfig(ConfigNode):
    restricted_directory: PathText
    permitted_fields: tuple[OutputField, ...] = ()
    permitted_variable_values: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def reject_duplicates(self) -> Self:
        if len(self.permitted_fields) != len(set(self.permitted_fields)):
            raise ValueError("Output fields must be unique.")
        if len(self.permitted_variable_values) != len(set(self.permitted_variable_values)):
            raise ValueError("Permitted output variables must be unique.")
        return self


class LinkageConfig(ConfigNode):
    schema_version: Literal["0.1"]
    project: ProjectConfig
    runtime: RuntimeConfig
    privacy: PrivacyConfig
    datasets: Annotated[tuple[DatasetConfig, ...], Field(min_length=1, max_length=64)]
    variables: Annotated[tuple[VariableConfig, ...], Field(min_length=1, max_length=512)]
    deterministic_anchors: tuple[DeterministicAnchorConfig, ...] = ()
    blocking: BlockingConfig
    comparisons: Annotated[tuple[ComparisonConfig, ...], Field(min_length=1, max_length=512)]
    labels: LabelsConfig | None = None
    models: ModelsConfig
    calibration: CalibrationConfig
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    assignment: AssignmentConfig
    decision_policy: DecisionPolicyConfig
    validation: ValidationConfig
    outputs: OutputConfig

    @model_validator(mode="after")
    def validate_project_consistency(self) -> Self:
        self._validate_unique_ids()
        dataset_ids = {dataset.id for dataset in self.datasets}
        variable_by_id = {variable.id: variable for variable in self.variables}
        self._validate_linkage_mode(dataset_ids)
        self._validate_variables(dataset_ids)
        self._validate_predicates(variable_by_id)
        self._validate_comparisons(variable_by_id)
        self._validate_labels(dataset_ids)
        self._validate_models()
        if self.models.fellegi_sunter.u_max_pairs > self.runtime.maximum_candidate_pairs:
            raise ValueError(
                "Fellegi-Sunter random-pair sampling cannot exceed the runtime pair budget."
            )
        if any(
            model.enabled and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.all_boosted_trees()
        ):
            raise ValueError("Boosted-tree training cannot exceed the runtime pair budget.")
        if any(
            model.enabled and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.all_ranking_models()
        ):
            raise ValueError("Ranking training cannot exceed the runtime pair budget.")
        if any(
            model.enabled and model.maximum_training_pairs > self.runtime.maximum_candidate_pairs
            for model in self.models.ensembles
        ):
            raise ValueError("Ensemble training cannot exceed the runtime pair budget.")
        self._validate_outputs(set(variable_by_id))
        if self.assignment.constraint != self.project.assignment_constraint:
            raise ValueError("Project and assignment constraints must agree.")
        if (
            self.assignment.constraint == "unconstrained"
            and self.assignment.solver != "unconstrained"
        ):
            raise ValueError("Unconstrained assignment requires the unconstrained solver.")
        if (
            self.assignment.constraint != "unconstrained"
            and self.assignment.solver == "unconstrained"
        ):
            raise ValueError("Constrained assignment requires a constrained solver.")
        if (
            self.assignment.constraint != "unconstrained"
            and not self.decision_policy.confirmed.require_assignment
        ):
            raise ValueError("Confirmed constrained links must require assignment.")
        return self

    def _validate_unique_ids(self) -> None:
        groups = {
            "dataset": [item.id for item in self.datasets],
            "variable": [item.id for item in self.variables],
            "anchor": [item.id for item in self.deterministic_anchors],
            "blocking rule": [item.id for item in self.blocking.rules],
            "comparison": [item.id for item in self.comparisons],
        }
        groups["model"] = list(self.models.all_model_ids())
        for label, values in groups.items():
            duplicate = next((value for value, count in Counter(values).items() if count > 1), None)
            if duplicate is not None:
                raise ValueError(f"{label.title()} IDs must be unique.")

    def _validate_linkage_mode(self, dataset_ids: set[str]) -> None:
        mode = self.project.linkage_mode
        if mode == "dedupe_only" and len(dataset_ids) != 1:
            raise ValueError("dedupe_only requires exactly one dataset.")
        if mode == "link_only" and len(dataset_ids) < 2:
            raise ValueError("link_only requires at least two datasets.")
        if mode == "multi_source" and len(dataset_ids) < 3:
            raise ValueError("multi_source requires at least three datasets.")
        roles = Counter(dataset.role for dataset in self.datasets)
        if mode == "link_only" and (roles["source"] < 1 or roles["target"] < 1):
            raise ValueError("link_only requires source and target dataset roles.")

    def _validate_variables(self, dataset_ids: set[str]) -> None:
        for variable in self.variables:
            if not set(variable.source_columns).issubset(dataset_ids):
                raise ValueError("A variable maps an unknown dataset.")
            for transform in variable.normalisation:
                allowed = {
                    "strip": {"string", "categorical"},
                    "casefold": {"string", "categorical"},
                    "unicode_normalize": {"string", "categorical"},
                    "collapse_whitespace": {"string", "categorical"},
                    "parse_date": {"date"},
                    "numeric_cast": {"integer", "float"},
                }[transform.kind]
                if variable.data_type not in allowed:
                    raise ValueError("A transform is incompatible with its variable type.")
                if (
                    isinstance(transform, NumericCastTransform)
                    and transform.target != variable.data_type
                ):
                    raise ValueError("numeric_cast target must match the variable data type.")

    def _validate_predicates(self, variable_by_id: dict[str, VariableConfig]) -> None:
        predicates = [anchor.predicate for anchor in self.deterministic_anchors]
        predicates.extend(rule.predicate for rule in self.blocking.rules)
        for predicate in predicates:
            for term in walk_predicate(predicate):
                variable = variable_by_id.get(term.variable)
                if variable is None:
                    raise ValueError("A predicate references an unknown variable.")
                if isinstance(term, PrefixEqualPredicate) and variable.data_type not in {
                    "string",
                    "categorical",
                }:
                    raise ValueError("prefix_equal requires a string-like variable.")
                if isinstance(term, DateWindowPredicate) and variable.data_type != "date":
                    raise ValueError("date_window requires a date variable.")

    def _validate_comparisons(self, variable_by_id: dict[str, VariableConfig]) -> None:
        similarity_functions = {
            "jaro_winkler",
            "levenshtein",
            "damerau_levenshtein",
            "qgram",
        }
        difference_functions = {"date_difference", "numeric_difference"}
        for comparison in self.comparisons:
            variable = variable_by_id.get(comparison.variable)
            if variable is None:
                raise ValueError("A comparison references an unknown variable.")
            function = comparison.function.kind
            if function in similarity_functions and variable.data_type not in {
                "string",
                "categorical",
            }:
                raise ValueError("A string comparison requires a string-like variable.")
            if function == "date_difference" and variable.data_type != "date":
                raise ValueError("date_difference requires a date variable.")
            if function == "numeric_difference" and variable.data_type not in {
                "integer",
                "float",
            }:
                raise ValueError("numeric_difference requires a numeric variable.")

            level_kinds = {level.kind for level in comparison.levels}
            if variable.missingness.comparison_policy == "explicit_missing_level":
                if "missing" not in level_kinds:
                    raise ValueError("Explicit missingness requires a missing comparison level.")
            elif "missing" in level_kinds:
                raise ValueError("Ignored missingness must not define a missing comparison level.")

            allowed_levels = {"missing", "exact", "else"}
            if function in similarity_functions:
                allowed_levels.add("threshold")
                if "threshold" not in level_kinds:
                    raise ValueError("A similarity comparison requires a threshold level.")
            elif function in difference_functions:
                allowed_levels.add("maximum_difference")
                if "maximum_difference" not in level_kinds:
                    raise ValueError("A difference comparison requires a maximum-difference level.")
            if not level_kinds.issubset(allowed_levels):
                raise ValueError("Comparison levels are incompatible with the comparison function.")

    def _validate_labels(self, dataset_ids: set[str]) -> None:
        if self.labels is None:
            return
        source = self.labels.source
        if isinstance(source, SyntheticTruthSource):
            entity_keys = set(source.entity_group_columns)
            household_keys = set(source.household_group_columns)
            if entity_keys != dataset_ids:
                raise ValueError("Synthetic entity truth must map every configured dataset.")
            if household_keys and household_keys != dataset_ids:
                raise ValueError("Synthetic household truth must map every configured dataset.")

    def _validate_models(self) -> None:
        eligible_truth = self.labels is not None and self.labels.source.kind in {
            "synthetic_truth",
            "verified_human_adjudication",
            "verified_gold_standard",
        }
        supervised_requires_verified_labels = (
            any(
                model.enabled and model.require_verified_labels
                for model in self.models.all_boosted_trees()
            )
            or any(
                model.enabled and model.require_verified_labels
                for model in self.models.all_ranking_models()
            )
            or any(
                model.enabled and model.require_verified_labels
                for model in self.models.all_neural_models()
            )
            or any(
                model.enabled and model.require_verified_labels for model in self.models.ensembles
            )
        )
        if supervised_requires_verified_labels and not eligible_truth:
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
        score_model_ids.update(model.model_id for model in self.models.ensembles if model.enabled)
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

    def _validate_outputs(self, variable_ids: set[str]) -> None:
        if not set(self.outputs.permitted_variable_values).issubset(variable_ids):
            raise ValueError("Outputs permit an unknown variable value.")
        variable_by_id = {variable.id: variable for variable in self.variables}
        if any(
            not variable_by_id[variable_id].restricted_output
            for variable_id in self.outputs.permitted_variable_values
        ):
            raise ValueError("Output variable values require restricted_output permission.")


def walk_predicate(
    predicate: BlockPredicate,
) -> tuple[
    ExactPredicate | PrefixEqualPredicate | DateWindowPredicate,
    ...,
]:
    """Return leaf predicates without exposing backend expressions."""

    if isinstance(predicate, (ExactPredicate, PrefixEqualPredicate, DateWindowPredicate)):
        return (predicate,)
    leaves: list[ExactPredicate | PrefixEqualPredicate | DateWindowPredicate] = []
    for term in predicate.terms:
        leaves.extend(walk_predicate(term))
    return tuple(leaves)


AllPredicate.model_rebuild(_types_namespace={"BlockPredicate": BlockPredicate})
AnyPredicate.model_rebuild(_types_namespace={"BlockPredicate": BlockPredicate})
