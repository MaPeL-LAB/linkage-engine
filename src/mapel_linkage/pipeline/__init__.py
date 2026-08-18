"""Package-owned orchestration for the complete synthetic vertical slice."""

from mapel_linkage.pipeline.contracts import StageSummary, SyntheticVerticalSliceResult
from mapel_linkage.pipeline.synthetic_vertical_slice import SyntheticVerticalSliceRunner

__all__ = ["StageSummary", "SyntheticVerticalSliceResult", "SyntheticVerticalSliceRunner"]
