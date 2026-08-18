"""Validation-only champion-challenger selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from mapel_linkage.calibration.calibrators import (
    BetaCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
)
from mapel_linkage.calibration.contracts import (
    CalibrationMethod,
    CalibratorArtifact,
    ChampionSelection,
    ModelEvaluationCandidate,
    PairScoreBatch,
    canonical_digest,
)
from mapel_linkage.configuration.models import ModelSelectionConfig
from mapel_linkage.domain.errors import CalibrationError, ModelSelectionError


class ChampionChallengerSelector:
    """Choose one evidence model using the protected validation partition only."""

    @staticmethod
    def select(
        candidates: Sequence[ModelEvaluationCandidate],
        config: ModelSelectionConfig,
    ) -> ChampionSelection:
        materialised = tuple(candidates)
        if len(materialised) < 2:
            raise ModelSelectionError(
                "ML-SELECT-008", "Champion selection requires at least two model candidates."
            )
        identities = {
            (item.model_family, item.model_id, item.model_version) for item in materialised
        }
        if len(identities) != len(materialised):
            raise ModelSelectionError(
                "ML-SELECT-009", "Duplicate champion candidates were rejected."
            )
        validation_authorities = {item.validation_label_authority_digest for item in materialised}
        partition_manifests = {item.partition_manifest_digest for item in materialised}
        pair_counts = {item.pair_count for item in materialised}
        if (
            len(validation_authorities) != 1
            or len(partition_manifests) != 1
            or len(pair_counts) != 1
        ):
            raise ModelSelectionError(
                "ML-SELECT-010",
                "Champion candidates do not share one validation evidence contract.",
            )

        if config.primary_metric == "average_precision":
            secondary: Literal["average_precision", "brier_score"] = "brier_score"
            ordered = sorted(
                materialised,
                key=lambda item: (
                    -item.average_precision,
                    item.brier_score,
                    item.model_family,
                    item.model_id,
                    item.model_version,
                ),
            )
        else:
            secondary = "average_precision"
            ordered = sorted(
                materialised,
                key=lambda item: (
                    item.brier_score,
                    -item.average_precision,
                    item.model_family,
                    item.model_id,
                    item.model_version,
                ),
            )
        selected = ordered[0]
        candidate_summaries = tuple(
            item.safe_summary()
            for item in sorted(
                materialised, key=lambda x: (x.model_family, x.model_id, x.model_version)
            )
        )
        payload = {
            "selected": selected.safe_summary(),
            "selected_training_label_authority_digest": selected.training_label_authority_digest,
            "validation_label_authority_digest": selected.validation_label_authority_digest,
            "partition_manifest_digest": selected.partition_manifest_digest,
            "primary_metric": config.primary_metric,
            "secondary_metric": secondary,
            "candidates": candidate_summaries,
            "test_partition_used": False,
            "calibration_partition_used": False,
        }
        return ChampionSelection(
            selected_model_family=selected.model_family,
            selected_model_id=selected.model_id,
            selected_model_version=selected.model_version,
            selected_evidence_digest=selected.evidence_digest,
            selected_feature_schema_digest=selected.feature_schema_digest,
            selected_training_label_authority_digest=selected.training_label_authority_digest,
            validation_label_authority_digest=selected.validation_label_authority_digest,
            partition_manifest_digest=selected.partition_manifest_digest,
            primary_metric=config.primary_metric,
            secondary_metric=secondary,
            selection_digest=canonical_digest(payload),
            candidate_summaries=candidate_summaries,
        )


class ChampionCalibratorSelector:
    """Select the champion probability calibrator on the protected calibration partition."""

    @staticmethod
    def select(
        batch: PairScoreBatch,
        selection: ChampionSelection,
        *,
        methods: Sequence[CalibrationMethod] = ("sigmoid", "isotonic", "beta"),
        primary_metric: Literal["brier_score", "expected_calibration_error"] = "brier_score",
    ) -> CalibratorArtifact:
        artifacts: list[CalibratorArtifact] = []
        for method in methods:
            if method == "sigmoid":
                artifacts.append(SigmoidCalibrator.fit(batch, selection))
            elif method == "isotonic":
                artifacts.append(IsotonicCalibrator.fit(batch, selection))
            elif method == "beta":
                artifacts.append(BetaCalibrator.fit(batch, selection))
            else:
                raise CalibrationError("ML-CAL-029", f"Unknown calibration method: {method}")
        if not artifacts:
            raise CalibrationError("ML-CAL-044", "No calibrator candidates were provided.")

        if primary_metric == "brier_score":
            ordered = sorted(
                artifacts,
                key=lambda a: (
                    a.diagnostics.brier_score,
                    a.diagnostics.expected_calibration_error,
                    a.method,
                ),
            )
        else:
            ordered = sorted(
                artifacts,
                key=lambda a: (
                    a.diagnostics.expected_calibration_error,
                    a.diagnostics.brier_score,
                    a.method,
                ),
            )
        return ordered[0]
