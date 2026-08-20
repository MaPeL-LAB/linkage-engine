"""Approved recipe inference runner with immutable provenance and zero parameter drift."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
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
from mapel_linkage.pipeline.portfolio_runner import (
    ReferenceFeatureScoreArtifact,
    StackingInferenceArtifactBundle,
)
from mapel_linkage.pipeline.recipe_io import deserialize_pipeline_recipe
from mapel_linkage.pipeline.recipes import (
    PipelineRecipeArtifact,
    RecipeExecutionMode,
    SyntheticInferenceAttestation,
)
from mapel_linkage.synthetic import (
    SyntheticBundle,
    SyntheticGenerationConfig,
    generate_synthetic_bundle,
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_bundle_digest(bundle: SyntheticBundle) -> str:
    """Hash a generated bundle without returning or logging its record-level values."""
    return _canonical_digest(
        {
            "provenance": asdict(bundle.provenance),
            "source_a": [record.as_mapping() for record in bundle.source_a],
            "source_b": [record.as_mapping() for record in bundle.source_b],
            "truth": [record.as_mapping() for record in bundle.truth],
        }
    )


def _verified_package_bundle_digest(bundle: SyntheticBundle) -> str:
    """Regenerate a claimed bundle and reject any non-package or modified content."""
    try:
        if not isinstance(bundle, SyntheticBundle):
            raise TypeError
        provenance = bundle.provenance
        regenerated = generate_synthetic_bundle(
            SyntheticGenerationConfig(
                seed=provenance.seed,
                entity_count=provenance.entity_count,
                left_only_count=provenance.left_only_count,
                right_only_count=provenance.right_only_count,
                duplicate_count=provenance.duplicate_count,
                competing_candidate_count=provenance.competing_candidate_count,
                source_a_missing_rate=provenance.source_a_missing_rate,
                source_b_missing_rate=provenance.source_b_missing_rate,
                source_b_typo_rate=provenance.source_b_typo_rate,
                source_b_date_shift_rate=provenance.source_b_date_shift_rate,
            )
        )
        bundle_digest = _synthetic_bundle_digest(bundle)
        regenerated_digest = _synthetic_bundle_digest(regenerated)
    except (AttributeError, TypeError, ValueError):
        raise PipelineError(
            "ML-PIPE-059",
            "Synthetic inference requires an unmodified package-generated bundle.",
        ) from None
    if bundle_digest != regenerated_digest:
        raise PipelineError(
            "ML-PIPE-059",
            "Synthetic inference requires an unmodified package-generated bundle.",
        )
    return bundle_digest


def _selected_evidence_digest(
    *,
    pair_references: tuple[tuple[str, str], ...],
    raw_scores: NDArray[np.float64] | Sequence[float] | None,
    feature_matrix: BoostedFeatureMatrix | None,
) -> tuple[str, str]:
    """Bind the exact selected evidence path without retaining values in the contract."""
    if raw_scores is not None:
        try:
            values = np.asarray(raw_scores, dtype="<f8")
        except (TypeError, ValueError):
            raise PipelineError(
                "ML-PIPE-061",
                "Synthetic inference evidence is invalid.",
            ) from None
        if (
            values.ndim != 1
            or values.shape[0] != len(pair_references)
            or not np.all(np.isfinite(values))
        ):
            raise PipelineError("ML-PIPE-061", "Synthetic inference evidence is invalid.")
        digest = hashlib.sha256()
        digest.update(b"raw_scores\x00")
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        return "raw_scores", digest.hexdigest()

    if feature_matrix is None:
        raise PipelineError("ML-PIPE-061", "Synthetic inference evidence is invalid.")
    values = np.asarray(feature_matrix.features, dtype="<f8")
    if (
        feature_matrix.pair_references != pair_references
        or values.ndim != 2
        or values.shape[0] != len(pair_references)
        or not np.all(np.isfinite(values))
    ):
        raise PipelineError("ML-PIPE-061", "Synthetic inference evidence is invalid.")
    digest = hashlib.sha256()
    digest.update(b"feature_matrix\x00")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(feature_matrix.feature_schema_digest.encode("ascii"))
    digest.update("\x00".join(feature_matrix.feature_names).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return "feature_matrix", digest.hexdigest()


def _inference_input_digest(
    *,
    source_record_keys: tuple[str, ...],
    pair_references: tuple[tuple[str, str], ...],
    raw_scores: NDArray[np.float64] | Sequence[float] | None,
    feature_matrix: BoostedFeatureMatrix | None,
    source_dataset_id: str,
    target_dataset_id: str,
) -> str:
    evidence_kind, evidence_digest = _selected_evidence_digest(
        pair_references=pair_references,
        raw_scores=raw_scores,
        feature_matrix=feature_matrix,
    )
    return _canonical_digest(
        {
            "source_dataset_id": source_dataset_id,
            "target_dataset_id": target_dataset_id,
            "source_record_digests": [
                hashlib.sha256(f"source\x00{key}".encode()).hexdigest()
                for key in source_record_keys
            ],
            "pair_digests": [pair_digest(left, right) for left, right in pair_references],
            "evidence_kind": evidence_kind,
            "evidence_digest": evidence_digest,
        }
    )


def attest_generated_synthetic_inference(
    *,
    bundle: SyntheticBundle,
    source_record_keys: tuple[str, ...],
    pair_references: tuple[tuple[str, str], ...],
    raw_scores: NDArray[np.float64] | Sequence[float] | None = None,
    feature_matrix: BoostedFeatureMatrix | None = None,
    source_dataset_id: str = "source_a",
    target_dataset_id: str = "source_b",
) -> SyntheticInferenceAttestation:
    """Authorize one exact inference input from a verified package-generated bundle.

    The returned object is an aggregate, in-memory capability. It cannot establish operational
    validity and carries no recommendation, relationship, assignment, or merge authority.
    """
    bundle_digest = _verified_package_bundle_digest(bundle)
    datasets = {
        "source_a": frozenset(record.record_key for record in bundle.source_a),
        "source_b": frozenset(record.record_key for record in bundle.source_b),
    }
    if (
        source_dataset_id not in datasets
        or target_dataset_id not in datasets
        or source_dataset_id == target_dataset_id
        or len(source_record_keys) != len(set(source_record_keys))
    ):
        raise PipelineError(
            "ML-PIPE-060",
            "Synthetic inference references are outside the verified generated bundle.",
        )
    source_keys = datasets[source_dataset_id]
    target_keys = datasets[target_dataset_id]
    supplied_sources = frozenset(source_record_keys)
    if (
        not source_record_keys
        or not pair_references
        or not supplied_sources.issubset(source_keys)
        or any(
            left not in supplied_sources or right not in target_keys
            for left, right in pair_references
        )
    ):
        raise PipelineError(
            "ML-PIPE-060",
            "Synthetic inference references are outside the verified generated bundle.",
        )
    input_digest = _inference_input_digest(
        source_record_keys=source_record_keys,
        pair_references=pair_references,
        raw_scores=raw_scores,
        feature_matrix=feature_matrix,
        source_dataset_id=source_dataset_id,
        target_dataset_id=target_dataset_id,
    )
    return SyntheticInferenceAttestation._issue(
        synthetic_bundle_digest=bundle_digest,
        inference_input_digest=input_digest,
        source_record_count=len(source_record_keys),
        pair_count=len(pair_references),
    )


def _resolve_recipe(recipe: PipelineRecipeArtifact | str | Path) -> PipelineRecipeArtifact:
    """Resolve an artifact, JSON payload, or path without probing JSON as a path."""
    if isinstance(recipe, PipelineRecipeArtifact):
        return recipe
    if isinstance(recipe, Path):
        try:
            if not recipe.is_file():
                raise PipelineError("ML-PIPE-058", "The pipeline recipe path is not a file.")
            payload = recipe.read_text(encoding="utf-8")
        except OSError:
            raise PipelineError("ML-PIPE-058", "The pipeline recipe path is invalid.") from None
        return deserialize_pipeline_recipe(payload)

    if recipe.lstrip().startswith("{"):
        return deserialize_pipeline_recipe(recipe)
    try:
        recipe_path = Path(recipe)
        if recipe_path.is_file():
            return deserialize_pipeline_recipe(recipe_path.read_text(encoding="utf-8"))
    except OSError:
        raise PipelineError("ML-PIPE-058", "The pipeline recipe path is invalid.") from None
    return deserialize_pipeline_recipe(recipe)


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
    synthetic_attestation_digest: str | None = None
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
                "synthetic_attestation_digest": self.synthetic_attestation_digest,
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
            "synthetic_attestation_digest": self.synthetic_attestation_digest,
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
        synthetic_attestation: SyntheticInferenceAttestation | None = None,
        synthetic_bundle: SyntheticBundle | None = None,
        source_dataset_id: str = "source",
        target_dataset_id: str = "target",
        output_decisions_path: str | Path | None = None,
        policy: PathPolicy | None = None,
        scipy_reference: bool = False,
    ) -> ApprovedRecipeInferenceResult:
        """Run deterministic inference using frozen recipe artifacts."""
        # 1. Resolve and validate recipe artifact
        recipe_obj = _resolve_recipe(recipe)

        # 2. Check approval authority and the package-issued synthetic input capability
        if execution_mode is RecipeExecutionMode.SYNTHETIC_INFERENCE and (
            synthetic_attestation is None or synthetic_bundle is None
        ):
            raise PipelineError(
                "ML-PIPE-062",
                "Synthetic inference requires a package-issued input attestation.",
            )
        recipe_obj.assert_usable_for(
            execution_mode,
            synthetic_attestation=synthetic_attestation,
        )
        if execution_mode is RecipeExecutionMode.SYNTHETIC_INFERENCE:
            assert synthetic_attestation is not None
            assert synthetic_bundle is not None
            expected_attestation = attest_generated_synthetic_inference(
                bundle=synthetic_bundle,
                source_record_keys=source_record_keys,
                pair_references=pair_references,
                raw_scores=raw_scores,
                feature_matrix=feature_matrix,
                source_dataset_id=source_dataset_id,
                target_dataset_id=target_dataset_id,
            )
            if not hmac.compare_digest(
                synthetic_attestation.attestation_digest,
                expected_attestation.attestation_digest,
            ):
                raise PipelineError(
                    "ML-RECIPE-015",
                    "The synthetic inference attestation does not authorize this input.",
                )
            input_digest = _inference_input_digest(
                source_record_keys=source_record_keys,
                pair_references=pair_references,
                raw_scores=raw_scores,
                feature_matrix=feature_matrix,
                source_dataset_id=source_dataset_id,
                target_dataset_id=target_dataset_id,
            )
            synthetic_attestation.assert_authorizes(
                inference_input_digest=input_digest,
                source_record_count=len(source_record_keys),
                pair_count=len(pair_references),
            )
        elif synthetic_bundle is not None:
            raise PipelineError(
                "ML-RECIPE-016",
                "A synthetic inference bundle cannot authorize this execution mode.",
            )

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
            cls._assert_champion_artifact_matches_recipe(
                recipe=recipe_obj,
                feature_matrix=feature_matrix,
                model_artifact=champion_model_artifact,
            )
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
            synthetic_attestation_digest=(
                synthetic_attestation.attestation_digest
                if synthetic_attestation is not None
                else None
            ),
            output_path=output_file,
        )

    @staticmethod
    def _assert_champion_artifact_matches_recipe(
        *,
        recipe: PipelineRecipeArtifact,
        feature_matrix: BoostedFeatureMatrix,
        model_artifact: Any,
    ) -> None:
        """Bind executable fitted artifacts to the immutable recipe and feature schema."""
        if isinstance(model_artifact, StackingModelArtifact):
            raise PipelineError(
                "ML-PIPE-064",
                "Stacking inference requires the recipe-bound fitted artifact bundle.",
            )
        if not isinstance(
            model_artifact,
            (
                ReferenceFeatureScoreArtifact,
                StackingInferenceArtifactBundle,
                XGBoostModelArtifact,
                LightGBMModelArtifact,
                PyTorchModelArtifact,
            ),
        ):
            raise PipelineError(
                "ML-PIPE-064",
                "The fitted champion artifact does not match the pipeline recipe.",
            )
        artifact_schema = model_artifact.feature_schema_digest
        if (
            model_artifact.model_id != recipe.champion_model_id
            or model_artifact.model_version != recipe.champion_model_version
            or not hmac.compare_digest(
                model_artifact.model_digest,
                recipe.champion_artifact_digest,
            )
            or artifact_schema != recipe.feature_schema_digest
            or feature_matrix.feature_schema_digest != recipe.feature_schema_digest
            or model_artifact.configuration_digest != recipe.configuration_digest
            or model_artifact.decision_authority != "evidence_only"
        ):
            raise PipelineError(
                "ML-PIPE-064",
                "The fitted champion artifact does not match the pipeline recipe.",
            )

    @staticmethod
    def _score_with_model(
        *,
        feature_matrix: BoostedFeatureMatrix,
        model_artifact: Any,
    ) -> NDArray[np.float64]:
        """Score candidate feature matrix using typed model artifact."""
        if isinstance(model_artifact, StackingInferenceArtifactBundle):
            base_scores: dict[str, NDArray[np.float64]] = {}
            for base_artifact in model_artifact.base_artifacts:
                base_scores[base_artifact.model_id] = (
                    ApprovedRecipeInferenceRunner._score_base_artifact(
                        feature_matrix=feature_matrix,
                        model_artifact=base_artifact,
                    )
                )
            return StackingPairClassifier().predict(
                base_scores=base_scores,
                model=model_artifact.stacking_artifact,
            )
        if isinstance(model_artifact, StackingModelArtifact):
            raise PipelineError(
                "ML-PIPE-064",
                "Stacking inference requires the recipe-bound fitted artifact bundle.",
            )
        return ApprovedRecipeInferenceRunner._score_base_artifact(
            feature_matrix=feature_matrix,
            model_artifact=model_artifact,
        )

    @staticmethod
    def _score_base_artifact(
        *,
        feature_matrix: BoostedFeatureMatrix,
        model_artifact: Any,
    ) -> NDArray[np.float64]:
        """Score one recipe-verified base artifact without granting decision authority."""
        if isinstance(model_artifact, ReferenceFeatureScoreArtifact):
            if model_artifact.scoring_rule != "mean_feature_clip":
                raise PipelineError(
                    "ML-PIPE-065",
                    "A stacking base artifact cannot be replayed for inference.",
                )
            scores = np.clip(np.mean(feature_matrix.features, axis=1), 0.0, 1.0)
            scores.setflags(write=False)
            return scores
        if isinstance(model_artifact, XGBoostModelArtifact):
            return XGBoostPairClassifier._predict(matrix=feature_matrix, model=model_artifact)
        if isinstance(model_artifact, LightGBMModelArtifact):
            return LightGBMPairClassifier._predict(matrix=feature_matrix, model=model_artifact)
        if isinstance(model_artifact, PyTorchModelArtifact):
            return PyTorchPairMatcher._predict(matrix=feature_matrix, model=model_artifact)
        raise PipelineError(
            "ML-PIPE-055",
            "The fitted model artifact type is unsupported.",
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
    synthetic_attestation: SyntheticInferenceAttestation | None = None,
    synthetic_bundle: SyntheticBundle | None = None,
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
        synthetic_attestation=synthetic_attestation,
        synthetic_bundle=synthetic_bundle,
        source_dataset_id=source_dataset_id,
        target_dataset_id=target_dataset_id,
        output_decisions_path=output_decisions_path,
        policy=policy,
        scipy_reference=scipy_reference,
    )


__all__ = [
    "ApprovedRecipeInferenceResult",
    "ApprovedRecipeInferenceRunner",
    "SyntheticInferenceAttestation",
    "attest_generated_synthetic_inference",
    "infer_with_approved_recipe",
]
