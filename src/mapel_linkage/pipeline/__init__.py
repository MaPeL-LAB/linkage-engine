"""Package-owned orchestration and immutable pipeline approval contracts."""

from mapel_linkage.pipeline.contracts import StageSummary, SyntheticVerticalSliceResult
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
    RankingCandidateDeclaration,
    compile_model_portfolio,
)
from mapel_linkage.pipeline.recipe_io import (
    deserialize_pipeline_recipe,
    pipeline_recipe_payload,
    serialize_pipeline_recipe,
)
from mapel_linkage.pipeline.recipes import (
    OperationalValidationStatus,
    PipelineRecipeArtifact,
    RecipeApprovalStatus,
    RecipeExecutionMode,
)
from mapel_linkage.pipeline.synthetic_vertical_slice import SyntheticVerticalSliceRunner

__all__ = [
    "ModelPortfolioDeclaration",
    "OperationalValidationStatus",
    "PairModelCandidateDeclaration",
    "PipelineRecipeArtifact",
    "RankingCandidateDeclaration",
    "RecipeApprovalStatus",
    "RecipeExecutionMode",
    "StageSummary",
    "SyntheticVerticalSliceResult",
    "SyntheticVerticalSliceRunner",
    "compile_model_portfolio",
    "deserialize_pipeline_recipe",
    "pipeline_recipe_payload",
    "serialize_pipeline_recipe",
]
