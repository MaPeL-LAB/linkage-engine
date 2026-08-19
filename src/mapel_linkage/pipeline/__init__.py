"""Package-owned orchestration and immutable pipeline approval contracts."""

from mapel_linkage.pipeline.contracts import StageSummary, SyntheticVerticalSliceResult
from mapel_linkage.pipeline.deduplication_runner import (
    DeduplicationWorkflowResult,
    DeduplicationWorkflowRunner,
    LinkAndDedupeSolver,
    SingleSourceDeduplicationSolver,
)
from mapel_linkage.pipeline.inference_runner import (
    ApprovedRecipeInferenceResult,
    ApprovedRecipeInferenceRunner,
    infer_with_approved_recipe,
)
from mapel_linkage.pipeline.model_portfolio import (
    ModelPortfolioDeclaration,
    PairModelCandidateDeclaration,
    RankingCandidateDeclaration,
    compile_model_portfolio,
)
from mapel_linkage.pipeline.multisource_runner import (
    MultiSourceWorkflowResult,
    MultiSourceWorkflowRunner,
)
from mapel_linkage.pipeline.portfolio_runner import (
    ModelPortfolioRunner,
    PortfolioTournamentResult,
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
from mapel_linkage.pipeline.stage_artifacts import (
    OutOfFoldPredictionManifest,
    StageArtifactLedger,
    StageArtifactRef,
)
from mapel_linkage.pipeline.synthetic_vertical_slice import SyntheticVerticalSliceRunner

__all__ = [
    "ApprovedRecipeInferenceResult",
    "ApprovedRecipeInferenceRunner",
    "DeduplicationWorkflowResult",
    "DeduplicationWorkflowRunner",
    "LinkAndDedupeSolver",
    "ModelPortfolioDeclaration",
    "ModelPortfolioRunner",
    "MultiSourceWorkflowResult",
    "MultiSourceWorkflowRunner",
    "OperationalValidationStatus",
    "OutOfFoldPredictionManifest",
    "PairModelCandidateDeclaration",
    "PipelineRecipeArtifact",
    "PortfolioTournamentResult",
    "RankingCandidateDeclaration",
    "RecipeApprovalStatus",
    "RecipeExecutionMode",
    "SingleSourceDeduplicationSolver",
    "StageArtifactLedger",
    "StageArtifactRef",
    "StageSummary",
    "SyntheticVerticalSliceResult",
    "SyntheticVerticalSliceRunner",
    "compile_model_portfolio",
    "deserialize_pipeline_recipe",
    "infer_with_approved_recipe",
    "pipeline_recipe_payload",
    "serialize_pipeline_recipe",
]
