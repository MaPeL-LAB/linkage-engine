"""Privacy-safe linkage-task profiling contracts."""

from mapel_linkage.profiling.contracts import (
    CalibrationEvidenceStatus,
    CandidateBudgetStatus,
    CandidateGraphProfile,
    CandidateRecallStatus,
    CountBand,
    EvidenceProfile,
    LabelEvidenceClass,
    PreflightTaskProfile,
    ProfileScope,
    RateBand,
    VariableTypeCount,
)
from mapel_linkage.profiling.preflight import build_preflight_task_profile

__all__ = [
    "CalibrationEvidenceStatus",
    "CandidateBudgetStatus",
    "CandidateGraphProfile",
    "CandidateRecallStatus",
    "CountBand",
    "EvidenceProfile",
    "LabelEvidenceClass",
    "PreflightTaskProfile",
    "ProfileScope",
    "RateBand",
    "VariableTypeCount",
    "build_preflight_task_profile",
]
