from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from mapel_linkage.configuration.models import LinkageConfig
from mapel_linkage.pipeline import compile_model_portfolio
from tests.helpers import EXAMPLE_CONFIG


def example_payload() -> dict[str, object]:
    loaded = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def test_plural_configuration_compiles_lightgbm_neural_stacking_and_rankers() -> None:
    raw = example_payload()
    models = raw["models"]
    assert isinstance(models, dict)
    fs_id = models["fellegi_sunter"]["model_id"]
    xgb_id = models["boosted_tree"]["model_id"]
    xgb_ranker_id = models["ranking"]["model_id"]
    models["boosted_trees"] = [
        {
            "enabled": True,
            "implementation": "lightgbm_classifier",
            "model_id": "lgb_candidate",
        }
    ]
    models["ranking_models"] = [
        {
            "enabled": True,
            "implementation": "lightgbm_ranker",
            "model_id": "lgb_ranker",
            "query_side": "source",
            "top_k": 5,
        }
    ]
    models["neural_models"] = [
        {
            "enabled": True,
            "implementation": "pytorch_pair_mlp",
            "model_id": "torch_candidate",
        }
    ]
    models["ensembles"] = [
        {
            "enabled": True,
            "implementation": "stacking_logistic",
            "model_id": "stacked_candidate",
            "base_model_ids": [xgb_id, "lgb_candidate"],
        }
    ]
    models["portfolio"] = {
        "portfolio_schema_version": "1",
        "portfolio_id": "full_synthetic_portfolio",
        "pair_model_ids": [
            fs_id,
            xgb_id,
            "lgb_candidate",
            "torch_candidate",
            "stacked_candidate",
        ],
        "ranking_model_ids": [xgb_ranker_id, "lgb_ranker"],
        "mandatory_baseline_id": fs_id,
        "maximum_challengers": 4,
    }

    config = LinkageConfig.model_validate(raw)
    compiled = compile_model_portfolio(config)

    assert compiled.portfolio_id == "full_synthetic_portfolio"
    assert [candidate.family for candidate in compiled.pair_candidates] == [
        "fellegi_sunter",
        "xgboost",
        "lightgbm",
        "pytorch",
        "stacking",
    ]
    assert [candidate.family for candidate in compiled.ranking_candidates] == [
        "xgboost",
        "lightgbm",
    ]
    assert compiled.test_partition_may_select_portfolio is False
    assert compiled.merge_authority == "none"


def test_plural_configuration_rejects_duplicate_ids() -> None:
    raw = example_payload()
    models = raw["models"]
    assert isinstance(models, dict)
    models["boosted_trees"] = [
        {
            "implementation": "lightgbm_classifier",
            "model_id": models["boosted_tree"]["model_id"],
        }
    ]
    with pytest.raises(ValidationError):
        LinkageConfig.model_validate(raw)


def test_stacking_rejects_unknown_or_disabled_base_model() -> None:
    raw = example_payload()
    models = raw["models"]
    assert isinstance(models, dict)
    models["ensembles"] = [
        {
            "enabled": True,
            "implementation": "stacking_logistic",
            "model_id": "stacked_candidate",
            "base_model_ids": [models["boosted_tree"]["model_id"], "missing_candidate"],
        }
    ]
    with pytest.raises(ValidationError):
        LinkageConfig.model_validate(raw)
