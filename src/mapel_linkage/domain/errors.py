"""Value-safe runtime errors for the local linkage data plane."""

from __future__ import annotations


class LinkageRuntimeError(RuntimeError):
    """A stable, non-sensitive runtime error suitable for CLI translation."""

    __slots__ = ("code", "public_message")

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


class DataPlaneError(LinkageRuntimeError):
    """Raised when a local table operation fails without exposing row values."""


class CandidateGenerationError(LinkageRuntimeError):
    """Raised when safe candidate generation cannot complete."""


class CandidateBudgetExceeded(CandidateGenerationError):
    """Raised before materialisation when the configured pair budget is exceeded."""


class PreprocessingError(LinkageRuntimeError):
    """Raised when configured ingestion or normalisation cannot complete safely."""


class ComparisonFeatureError(LinkageRuntimeError):
    """Raised when configured comparison features cannot be built safely."""


class AnchorEvidenceError(LinkageRuntimeError):
    """Raised when deterministic anchor evidence cannot be evaluated safely."""


class AnchorBudgetExceeded(AnchorEvidenceError):
    """Raised before materialisation when the anchor pair budget is exceeded."""


class FellegiSunterError(LinkageRuntimeError):
    """Raised when Fellegi-Sunter evidence estimation cannot complete safely."""


class FellegiSunterBudgetExceeded(FellegiSunterError):
    """Raised before fitting when a configured pair budget is exceeded."""


class LabelProvenanceError(LinkageRuntimeError):
    """Raised when verified-label provenance or partition safety is invalid."""


class BoostedTreeError(LinkageRuntimeError):
    """Raised when boosted pair-model training or scoring cannot complete safely."""


class BoostedTreeBudgetExceeded(BoostedTreeError):
    """Raised before fitting when a boosted-model pair budget is exceeded."""


class EnsembleError(LinkageRuntimeError):
    """Raised when ensemble model training, scoring, or stacking fails."""


class NeuralModelError(LinkageRuntimeError):
    """Raised when neural pair-model training or scoring cannot complete safely."""


class ModelSelectionError(LinkageRuntimeError):
    """Raised when champion-challenger selection violates its trust boundary."""


class CalibrationError(LinkageRuntimeError):
    """Raised when probability calibration cannot complete safely."""


class CalibrationArtifactError(CalibrationError):
    """Raised when a calibrator artifact fails integrity or path checks."""


class RankingError(LinkageRuntimeError):
    """Raised when candidate-ranking training or scoring cannot complete safely."""


class AssignmentError(LinkageRuntimeError):
    """Raised when constrained assignment cannot complete safely."""


class ClusteringError(LinkageRuntimeError):
    """Raised when multi-source clustering or resolution fails safely."""


class DecisionPolicyError(LinkageRuntimeError):
    """Raised when relationship decision evidence violates its contract."""


class AdjudicationError(LinkageRuntimeError):
    """Raised when a restricted review artifact cannot be created safely."""


class ValidationReportError(LinkageRuntimeError):
    """Raised when aggregate validation reporting cannot complete safely."""


class PipelineError(LinkageRuntimeError):
    """Raised when a pipeline stage cannot complete safely."""


class AdvisorError(LinkageRuntimeError):
    """Raised by advisory-only eligibility and recommendation boundaries."""
