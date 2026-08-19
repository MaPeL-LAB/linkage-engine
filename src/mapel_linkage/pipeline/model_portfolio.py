"""Typed model-portfolio contracts for bounded champion/challenger development."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from mapel_linkage.configuration.models import LinkageConfig

Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]

PairModelFamily = Literal[
    "fellegi_sunter",
    "xgboost",
    "lightgbm",
    "pytorch",
    "stacking",
]
PairModelImplementation = Literal[
    "mapel_reference_fellegi_sunter",
    "xgboost_classifier",
    "lightgbm_classifier",
    "pytorch_pair_mlp",
    "stacking_logistic",
]
PairModelRole = Literal["baseline", "challenger", "ensemble", "shadow"]
RankerFamily = Literal["xgboost", "lightgbm"]
RankerImplementation = Literal["xgboost_ranker", "lightgbm_ranker"]
ArtifactFormat = Literal[
    "package_json",
    "xgboost_json",
    "lightgbm_text",
    "pytorch_state_dict",
]


class PortfolioNode(BaseModel):
    """Strict immutable portfolio node; configuration remains non-executable data."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class PairModelCandidateDeclaration(PortfolioNode):
    """One eligible pair-scoring candidate in a development tournament."""

    model_id: Identifier
    family: PairModelFamily
    implementation: PairModelImplementation
    role: PairModelRole
    enabled: StrictBool = True
    require_verified_labels: StrictBool
    artifact_format: ArtifactFormat
    base_model_ids: Annotated[tuple[Identifier, ...], Field(max_length=16)] = ()
    decision_authority: Literal["evidence_only"] = "evidence_only"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_family_contract(self) -> Self:
        expected = {
            "mapel_reference_fellegi_sunter": "fellegi_sunter",
            "xgboost_classifier": "xgboost",
            "lightgbm_classifier": "lightgbm",
            "pytorch_pair_mlp": "pytorch",
            "stacking_logistic": "stacking",
        }[self.implementation]
        if self.family != expected:
            raise ValueError("Pair-model family and implementation must agree.")
        if self.family == "fellegi_sunter" and self.require_verified_labels:
            raise ValueError("The reference Fellegi-Sunter baseline must not require labels.")
        if self.family != "stacking" and self.base_model_ids:
            raise ValueError("Only stacking candidates may declare base models.")
        if self.family == "stacking" and len(set(self.base_model_ids)) < 2:
            raise ValueError("A stacking candidate requires at least two distinct base models.")
        return self


class RankingCandidateDeclaration(PortfolioNode):
    """One candidate ranker; ranking never receives relationship authority."""

    model_id: Identifier
    family: RankerFamily
    implementation: RankerImplementation
    enabled: StrictBool = True
    query_side: Literal["source", "target"]
    top_k: Annotated[StrictInt, Field(gt=0, le=1000)]
    require_verified_labels: Literal[True] = True
    artifact_format: Literal["xgboost_json", "lightgbm_text"]
    ranking_authority: Literal["ordering_only"] = "ordering_only"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_family_contract(self) -> Self:
        expected = {
            "xgboost_ranker": "xgboost",
            "lightgbm_ranker": "lightgbm",
        }[self.implementation]
        if self.family != expected:
            raise ValueError("Ranking family and implementation must agree.")
        return self


class ModelPortfolioDeclaration(PortfolioNode):
    """Privacy-safe shortlist for a protected model-development tournament."""

    portfolio_id: Identifier
    pair_candidates: Annotated[
        tuple[PairModelCandidateDeclaration, ...], Field(min_length=1, max_length=16)
    ]
    ranking_candidates: Annotated[tuple[RankingCandidateDeclaration, ...], Field(max_length=8)] = ()
    mandatory_baseline_id: Identifier
    maximum_challengers: Annotated[StrictInt, Field(ge=0, le=8)] = 3
    allow_shadow_scoring: Literal[True] = True
    test_partition_may_select_portfolio: Literal[False] = False
    recommendation_authority: Literal["none"] = "none"
    decision_authority: Literal["none"] = "none"
    merge_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_portfolio(self) -> Self:
        pair_ids = [candidate.model_id for candidate in self.pair_candidates]
        ranker_ids = [candidate.model_id for candidate in self.ranking_candidates]
        all_ids = pair_ids + ranker_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Model-portfolio identifiers must be unique.")
        baseline = next(
            (
                candidate
                for candidate in self.pair_candidates
                if candidate.model_id == self.mandatory_baseline_id
            ),
            None,
        )
        if baseline is None or not baseline.enabled or baseline.role != "baseline":
            raise ValueError("The mandatory baseline must identify one enabled baseline.")
        if sum(candidate.role == "baseline" for candidate in self.pair_candidates) != 1:
            raise ValueError("A model portfolio must contain exactly one baseline.")
        pair_id_set = set(pair_ids)
        for candidate in self.pair_candidates:
            if candidate.family != "stacking":
                continue
            if candidate.model_id in candidate.base_model_ids:
                raise ValueError("A stacking candidate cannot include itself as a base model.")
            if not set(candidate.base_model_ids).issubset(pair_id_set):
                raise ValueError("Stacking base models must exist in the same portfolio.")
        return self

    @property
    def portfolio_digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_summary(self) -> dict[str, int | str | bool]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_digest": self.portfolio_digest,
            "pair_candidate_count": len(self.pair_candidates),
            "enabled_pair_candidate_count": sum(item.enabled for item in self.pair_candidates),
            "ranking_candidate_count": len(self.ranking_candidates),
            "maximum_challengers": self.maximum_challengers,
            "allow_shadow_scoring": self.allow_shadow_scoring,
            "test_partition_may_select_portfolio": self.test_partition_may_select_portfolio,
            "recommendation_authority": self.recommendation_authority,
            "decision_authority": self.decision_authority,
            "merge_authority": self.merge_authority,
        }


def compile_model_portfolio(config: LinkageConfig) -> ModelPortfolioDeclaration:
    """Normalize the current configuration into a portfolio-ready contract."""

    pair_candidates: list[PairModelCandidateDeclaration] = []
    fs = config.models.fellegi_sunter
    pair_candidates.append(
        PairModelCandidateDeclaration(
            model_id=fs.model_id,
            family="fellegi_sunter",
            implementation="mapel_reference_fellegi_sunter",
            role="baseline",
            enabled=fs.enabled,
            require_verified_labels=False,
            artifact_format="package_json",
        )
    )
    boosted = config.models.boosted_tree
    if boosted is not None:
        family: Literal["xgboost", "lightgbm"] = (
            "xgboost" if boosted.implementation == "xgboost_classifier" else "lightgbm"
        )
        artifact_format: Literal["xgboost_json", "lightgbm_text"] = (
            "xgboost_json" if family == "xgboost" else "lightgbm_text"
        )
        pair_candidates.append(
            PairModelCandidateDeclaration(
                model_id=boosted.model_id,
                family=family,
                implementation=boosted.implementation,
                role="challenger",
                enabled=boosted.enabled,
                require_verified_labels=boosted.require_verified_labels,
                artifact_format=artifact_format,
            )
        )
    neural = config.models.neural
    if neural is not None:
        pair_candidates.append(
            PairModelCandidateDeclaration(
                model_id=neural.model_id,
                family="pytorch",
                implementation=neural.implementation,
                role="challenger",
                enabled=neural.enabled,
                require_verified_labels=neural.require_verified_labels,
                artifact_format="pytorch_state_dict",
            )
        )
    ranking_candidates: list[RankingCandidateDeclaration] = []
    ranking = config.models.ranking
    if ranking is not None:
        rank_family: Literal["xgboost", "lightgbm"] = (
            "xgboost" if ranking.implementation == "xgboost_ranker" else "lightgbm"
        )
        rank_format: Literal["xgboost_json", "lightgbm_text"] = (
            "xgboost_json" if rank_family == "xgboost" else "lightgbm_text"
        )
        ranking_candidates.append(
            RankingCandidateDeclaration(
                model_id=ranking.model_id,
                family=rank_family,
                implementation=ranking.implementation,
                enabled=ranking.enabled,
                query_side=ranking.query_side,
                top_k=ranking.top_k,
                artifact_format=rank_format,
            )
        )
    enabled_challengers = sum(
        candidate.enabled and candidate.role != "baseline" for candidate in pair_candidates
    )
    return ModelPortfolioDeclaration(
        portfolio_id="compiled_model_portfolio",
        pair_candidates=tuple(pair_candidates),
        ranking_candidates=tuple(ranking_candidates),
        mandatory_baseline_id=fs.model_id,
        maximum_challengers=min(3, enabled_challengers),
    )


__all__ = [
    "ModelPortfolioDeclaration",
    "PairModelCandidateDeclaration",
    "RankingCandidateDeclaration",
    "compile_model_portfolio",
]
