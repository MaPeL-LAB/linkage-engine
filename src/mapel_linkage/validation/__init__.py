"""Validation metrics and leakage-resistant evaluation contracts."""

from mapel_linkage.validation.assignment_metrics import (
    AssignmentValidationReport,
    evaluate_assignment,
)
from mapel_linkage.validation.candidate_metrics import (
    CandidateRetrievalReport,
    evaluate_candidate_retrieval,
)
from mapel_linkage.validation.decision_metrics import (
    DecisionValidationReport,
    evaluate_decisions,
)
from mapel_linkage.validation.pair_metrics import PairValidationReport, evaluate_binary_scores
from mapel_linkage.validation.ranking_metrics import RankingValidationReport, evaluate_ranking
from mapel_linkage.validation.reporting import write_aggregate_validation_report
from mapel_linkage.validation.splitting import (
    EntityHouseholdRecord,
    PartitionAssignment,
    build_verified_candidate_label_batches,
    split_entity_household_components,
)
from mapel_linkage.validation.stratified_metrics import (
    PairPerformanceStratum,
    StratifiedPairValidationReport,
    candidate_set_size_band,
    evaluate_stratified_pair_performance,
)
from mapel_linkage.validation.threshold_metrics import (
    DecisionThresholdReport,
    evaluate_configured_decision_thresholds,
)

__all__ = [
    "AssignmentValidationReport",
    "CandidateRetrievalReport",
    "DecisionThresholdReport",
    "DecisionValidationReport",
    "EntityHouseholdRecord",
    "PairPerformanceStratum",
    "PairValidationReport",
    "PartitionAssignment",
    "RankingValidationReport",
    "StratifiedPairValidationReport",
    "build_verified_candidate_label_batches",
    "candidate_set_size_band",
    "evaluate_assignment",
    "evaluate_binary_scores",
    "evaluate_candidate_retrieval",
    "evaluate_configured_decision_thresholds",
    "evaluate_decisions",
    "evaluate_ranking",
    "evaluate_stratified_pair_performance",
    "split_entity_household_components",
    "write_aggregate_validation_report",
]
