"""Approved recipe inference runner with immutable provenance and zero parameter drift."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mapel_linkage.assignment.contracts import (
    AssignmentEdgeBatch,
    AssignmentPlan,
    AssignmentResult,
    pair_digest,
)
from mapel_linkage.assignment.solvers import (
    ManyToOneAssignmentSolver,
    OneToManyAssignmentSolver,
    OrToolsOneToOneAssignmentSolver,
    ScipyOneToOneAssignmentSolver,
    UnconstrainedAssignmentSolver,
)
from mapel_linkage.calibration.calibrators import (
    BetaCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
)
from mapel_linkage.calibration.contracts import CalibratorArtifact
from mapel_linkage.configuration.models import (
    ConfirmedDecisionConfig,
    DecisionPolicyConfig,
    NoMatchDecisionConfig,
    ReviewDecisionConfig,
    UnresolvedDecisionConfig,
)
from mapel_linkage.decisions.policy import (
    DecisionEvidenceBuilder,
    RelationshipDecision,
    RelationshipDecisionPolicy,
)
from mapel_linkage.domain.errors import PipelineError
from mapel_linkage.governance.atomic import atomic_write_text
from mapel_linkage.governance.paths import PathPolicy
from mapel_linkage.models.boosted import (
    BoostedFeatureMatrix,
    LightGBMModelArtifact,
    LightGBMPairClassifier,
    XGBoostModelArtifact,
    XGBoostPairClassifier,
)
from mapel_linkage.models.ensembles import (
    StackingModelArtifact,
    StackingPairClassifier,
)
from mapel_linkage.models.neural import (
    PyTorchModelArtifact,
    PyTorchPairMatcher,
)
from mapel_linkage.pipeline.recipe_io import deserialize_pipeline_recipe
from mapel_linkage.pipeline.recipes import (
    PipelineRecipeArtifact,
    RecipeExecutionMode,
)


def _default_decision_policy() -> DecisionPolicyConfig:
    return DecisionPolicyConfig(
        confirmed=ConfirmedDecisionConfig(
            minimum_probability=0.85,
            minimum_probability_margin=0.10,
        ),
        review_required=ReviewDecisionConfig(minimum_probability=0.50),
        no_match=NoMatchDecisionConfig(maximum_top_probability=0.20),
        unresolved=UnresolvedDecisionConfig(),
    )


@dataclass(frozen=True, slots=True, repr=False)
class ApprovedRecipeInferenceResult:
    """Outcome of approved recipe inference execution on new unlabelled records."""

    recipe_id: str
    recipe_version: str
    recipe_digest: str
    execution_mode: RecipeExecutionMode
    pair_count: int
    relationship_status_counts: dict[str, int]
    decisions: tuple[RelationshipDecision, ...] = field(repr=False)
    assignment_result: AssignmentResult = field(repr=False)
    output_path: Path | None = field(default=None, repr=False)
    inference_digest: str = ""

    def __post_init__(self) -> None:
        if not self.inference_digest:
            payload = {
                "recipe_digest": self.recipe_digest,
                "execution_mode": self.execution_mode.value,
                "pair_count": self.pair_count,
                "status_counts": self.relationship_status_counts,
                "assignment_digest": self.assignment_result.assignment_digest,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "inference_digest", digest)

    def safe_summary(self) -> dict[str, Any]:
        """Return aggregate summary without row-level keys."""
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "recipe_digest": self.recipe_digest,
            "execution_mode": self.execution_mode.value,
            "pair_count": self.pair_count,
            "relationship_status_counts": dict(self.relationship_status_counts),
            "assignment_method": self.assignment_result.solver,
            "real_assignment_count": self.assignment_result.real_assignment_count,
            "no_match_count": self.assignment_result.no_match_count,
            "inference_digest": self.inference_digest,
            "output_written": self.output_path is not None,
        }


class ApprovedRecipeInferenceRunner:
    """Execute frozen approved recipes on new unlabelled pairs with zero parameter drift."""

    @classmethod
    def run_inference(
        cls,
        *,
        recipe: PipelineRecipeArtifact | str | Path,
        source_record_keys: tuple[str, ...],
        pair_references: tuple[tuple[str, str], ...],
        raw_scores: NDArray[np.float64] | Sequence[float] | None = None,
        feature_matrix: BoostedFeatureMatrix | None = None,
        champion_model_artifact: Any | None = None,
        calibrator_artifact: CalibratorArtifact,
        decision_policy: DecisionPolicyConfig | None = None,
        ranking_artifact: Any | None = None,
        execution_mode: RecipeExecutionMode = RecipeExecutionMode.INFERENCE,
        source_dataset_id: str = "source",
        target_dataset_id: str = "target",
        output_decisions_path: str | Path | None = None,
        policy: PathPolicy | None = None,
        scipy_reference: bool = False,
    ) -> ApprovedRecipeInferenceResult:
        """Run deterministic inference using frozen recipe artifacts."""
        # 1. Resolve and validate recipe artifact
        recipe_obj: PipelineRecipeArtifact
        if isinstance(recipe, (str, Path)):
            path_or_str = str(recipe)
            if Path(path_or_str).is_file():
                recipe_obj = deserialize_pipeline_recipe(
                    Path(path_or_str).read_text(encoding="utf-8")
                )
            else:
                recipe_obj = deserialize_pipeline_recipe(path_or_str)
        else:
            recipe_obj = recipe

        # 2. Check approval authority for requested execution mode
        recipe_obj.assert_usable_for(execution_mode)

        # 3. Validate calibrator artifact matches recipe digest
        if calibrator_artifact.calibrator_digest != recipe_obj.calibrator_digest:
            raise PipelineError(
                "ML-PIPE-050",
                "Calibrator artifact digest does not match the approved pipeline recipe.",
            )

        # 4. Generate or validate model scores
        scores_array: NDArray[np.float64]
        if raw_scores is not None:
            scores_array = np.asarray(raw_scores, dtype=np.float64)
        elif feature_matrix is not None and champion_model_artifact is not None:
            scores_array = cls._score_with_model(
                feature_matrix=feature_matrix,
                model_artifact=champion_model_artifact,
            )
        else:
            raise PipelineError(
                "ML-PIPE-051",
                "Inference requires raw scores or feature matrix with champion model artifact.",
            )

        if len(scores_array) != len(pair_references):
            raise PipelineError(
                "ML-PIPE-052",
                "Score count does not match pair references count.",
            )

        # 5. Calibrate scores deterministically using frozen calibrator
        if calibrator_artifact.method == "sigmoid":
            calibrated_probs = SigmoidCalibrator.apply(scores_array, calibrator_artifact)
        elif calibrator_artifact.method == "beta":
            calibrated_probs = BetaCalibrator.apply(scores_array, calibrator_artifact)
        elif calibrator_artifact.method == "isotonic":
            calibrated_probs = IsotonicCalibrator.apply(scores_array, calibrator_artifact)
        else:
            raise PipelineError(
                "ML-PIPE-053",
                f"Unsupported calibrator method {calibrator_artifact.method}.",
            )

        # 6. Build Candidate ranks per source record
        pair_digests = tuple(pair_digest(u, v) for u, v in pair_references)
        ranks_by_source: dict[str, list[tuple[float, str, int]]] = {
            src: [] for src in source_record_keys
        }
        for idx, (src, _tgt) in enumerate(pair_references):
            if src in ranks_by_source:
                ranks_by_source[src].append((float(calibrated_probs[idx]), pair_digests[idx], idx))

        candidate_ranks = np.zeros(len(pair_references), dtype=np.int64)
        for _src, items in ranks_by_source.items():
            sorted_items = sorted(items, key=lambda it: (-it[0], it[1]))
            for rank_num, (_p, _dig, original_idx) in enumerate(sorted_items, start=1):
                candidate_ranks[original_idx] = rank_num

        # 7. Form AssignmentEdgeBatch
        batch = AssignmentEdgeBatch(
            source_record_keys=source_record_keys,
            pair_references=pair_references,
            pair_digests=pair_digests,
            probabilities=calibrated_probs,
            candidate_ranks=candidate_ranks,
            source_model_id=recipe_obj.champion_model_id,
            source_model_version=recipe_obj.champion_model_version,
            calibrator_digest=calibrator_artifact.calibrator_digest,
            ranking_model_digest=recipe_obj.ranking_artifact_digest,
            candidate_search_complete=True,
            candidate_search_truncated=False,
        )

        # 8. Solve assignment matching recipe constraint
        constraint = recipe_obj.assignment_constraint
        solver_name = (
            "scipy_linear_sum_assignment" if scipy_reference else "ortools_linear_sum_assignment"
        )
        plan = AssignmentPlan(constraint=constraint, solver=solver_name)

        if constraint == "one_to_one":
            if scipy_reference:
                assignment_res = ScipyOneToOneAssignmentSolver.solve(batch, plan)
            else:
                assignment_res = OrToolsOneToOneAssignmentSolver.solve(batch, plan)
        elif constraint == "many_to_one":
            assignment_res = ManyToOneAssignmentSolver.solve(batch, plan)
        elif constraint == "one_to_many":
            assignment_res = OneToManyAssignmentSolver.solve(batch, plan)
        elif constraint == "unconstrained":
            assignment_res = UnconstrainedAssignmentSolver.solve(batch, plan)
        else:
            raise PipelineError(
                "ML-PIPE-054",
                f"Unsupported assignment constraint {constraint}.",
            )

        # 9. Build DecisionEvidence
        evidence_list = DecisionEvidenceBuilder.build(
            candidates=batch,
            assignment=assignment_res,
            source_dataset_id=source_dataset_id,
            target_dataset_id=target_dataset_id,
        )

        # 10. Classify relationships with RelationshipDecisionPolicy
        policy_config = decision_policy or _default_decision_policy()
        run_id = hashlib.sha256(
            f"{recipe_obj.recipe_digest}_{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:24]

        decisions = RelationshipDecisionPolicy.classify_all(
            evidence=evidence_list,
            policy=policy_config,
            model_family="champion_model",
            model_version=recipe_obj.champion_model_version,
            assignment_method=assignment_res.solver,
            assignment_constraint=constraint,
            run_id=run_id,
            configuration_digest=recipe_obj.configuration_digest,
            feature_schema_digest=recipe_obj.feature_schema_digest,
        )

        status_counts = dict(Counter(d.relationship_status for d in decisions))

        # 11. Export decisions if requested
        output_file: Path | None = None
        if output_decisions_path is not None:
            output_file = cls.export_decisions(
                decisions=decisions,
                output_path=output_decisions_path,
                policy=policy,
            )

        status_dict = {str(k): int(v) for k, v in status_counts.items()}
        return ApprovedRecipeInferenceResult(
            recipe_id=recipe_obj.recipe_id,
            recipe_version=recipe_obj.recipe_version,
            recipe_digest=recipe_obj.recipe_digest,
            execution_mode=execution_mode,
            pair_count=len(pair_references),
            relationship_status_counts=status_dict,
            decisions=decisions,
            assignment_result=assignment_res,
            output_path=output_file,
        )

    @staticmethod
    def _score_with_model(
        *,
        feature_matrix: BoostedFeatureMatrix,
        model_artifact: Any,
    ) -> NDArray[np.float64]:
        """Score candidate feature matrix using typed model artifact."""
        if isinstance(model_artifact, XGBoostModelArtifact):
            return XGBoostPairClassifier._predict(matrix=feature_matrix, model=model_artifact)
        elif isinstance(model_artifact, LightGBMModelArtifact):
            return LightGBMPairClassifier._predict(matrix=feature_matrix, model=model_artifact)
        elif isinstance(model_artifact, PyTorchModelArtifact):
            return PyTorchPairMatcher._predict(matrix=feature_matrix, model=model_artifact)
        elif isinstance(model_artifact, StackingModelArtifact):
            # For stacking, features must be base model predictions
            return StackingPairClassifier().predict(
                base_scores=feature_matrix.features, model=model_artifact
            )
        else:
            raise PipelineError(
                "ML-PIPE-055",
                f"Unsupported model artifact type: {type(model_artifact)}",
            )

    @staticmethod
    def export_decisions(
        *,
        decisions: Sequence[RelationshipDecision],
        output_path: str | Path,
        policy: PathPolicy | None = None,
    ) -> Path:
        """Write relationship decisions to CSV or JSON."""
        dest = policy.resolve_output(str(output_path)) if policy is not None else Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        rows = [d.safe_summary() for d in decisions]

        if dest.suffix.lower() == ".json":
            atomic_write_text(dest, json.dumps(rows, indent=2, sort_keys=True) + "\n")
        else:
            with dest.open("w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "relationship_id",
                    "relationship_status",
                    "model_family",
                    "model_version",
                    "calibrated_probability",
                    "candidate_rank",
                    "probability_margin",
                    "decision_rule_id",
                    "assignment_method",
                    "run_id",
                    "decision_authority",
                    "merge_authority",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return dest


def infer_with_approved_recipe(
    *,
    recipe: PipelineRecipeArtifact | str | Path,
    source_record_keys: tuple[str, ...],
    pair_references: tuple[tuple[str, str], ...],
    raw_scores: NDArray[np.float64] | Sequence[float] | None = None,
    feature_matrix: BoostedFeatureMatrix | None = None,
    champion_model_artifact: Any | None = None,
    calibrator_artifact: CalibratorArtifact,
    decision_policy: DecisionPolicyConfig | None = None,
    ranking_artifact: Any | None = None,
    execution_mode: RecipeExecutionMode = RecipeExecutionMode.INFERENCE,
    source_dataset_id: str = "source",
    target_dataset_id: str = "target",
    output_decisions_path: str | Path | None = None,
    policy: PathPolicy | None = None,
    scipy_reference: bool = False,
) -> ApprovedRecipeInferenceResult:
    """Convenience function to run inference with an approved pipeline recipe."""
    return ApprovedRecipeInferenceRunner.run_inference(
        recipe=recipe,
        source_record_keys=source_record_keys,
        pair_references=pair_references,
        raw_scores=raw_scores,
        feature_matrix=feature_matrix,
        champion_model_artifact=champion_model_artifact,
        calibrator_artifact=calibrator_artifact,
        decision_policy=decision_policy,
        ranking_artifact=ranking_artifact,
        execution_mode=execution_mode,
        source_dataset_id=source_dataset_id,
        target_dataset_id=target_dataset_id,
        output_decisions_path=output_decisions_path,
        policy=policy,
        scipy_reference=scipy_reference,
    )


__all__ = [
    "ApprovedRecipeInferenceResult",
    "ApprovedRecipeInferenceRunner",
    "infer_with_approved_recipe",
]
