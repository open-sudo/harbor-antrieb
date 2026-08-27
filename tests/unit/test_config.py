from typing import Any

import pytest
from pydantic import ValidationError

from infraset.config import JudgeConfig, SemanticPlanConfig, InfraSetDefinition


def test_infraset_definition_requires_cluster() -> None:
    with pytest.raises(ValidationError, match="at least one image"):
        InfraSetDefinition(cluster=[])


def test_static_prepare_is_disabled_by_default() -> None:
    definition = InfraSetDefinition(cluster=["ubuntu24.04"])

    assert definition.initialize == []
    assert definition.prepare.enabled is False
    assert definition.max_clusters == 1
    assert definition.base_runbooks == ["antrieb/primer"]


def test_base_runbooks_reject_scenario_or_non_antrieb_documents() -> None:
    with pytest.raises(ValidationError, match="primers and image references"):
        InfraSetDefinition(
            cluster=["ubuntu24.04"],
            base_runbooks=["antrieb/primer", "antrieb/vyos-dnat-port-forward"],
        )

    with pytest.raises(ValidationError, match="must include antrieb/primer"):
        InfraSetDefinition(
            cluster=["ubuntu24.04"],
            base_runbooks=["antrieb/networking-primer"],
        )


def test_cluster_quota_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        InfraSetDefinition(cluster=["ubuntu24.04"], max_clusters=0)


def test_initialize_requires_unique_safe_names() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        InfraSetDefinition(cluster=["rhel9.8"], initialize=["rhsm", "rhsm"])

    with pytest.raises(ValidationError, match="invalid initializer names"):
        InfraSetDefinition(cluster=["rhel9.8"], initialize=["RHSM subscription"])


def test_judge_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="must total 1.0"):
        JudgeConfig.model_validate(
            {
                "dimensions": {
                    "functionality": {"weight": 0.5, "description": "works"},
                    "security": {"weight": 0.2, "description": "safe"},
                }
            }
        )


def test_judge_enables_bounded_parallel_node_collection_by_default() -> None:
    config = JudgeConfig.model_validate(
        {"dimensions": {"functionality": {"weight": 1.0, "description": "works"}}}
    )

    assert config.parallel_node_collection is True
    assert config.node_collector_timeout_sec == 300
    assert config.node_collector_command_budget == 8
    assert config.coordinator_command_budget == 64


def _semantic_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probes": [
            {
                "id": "service-probe",
                "level": 1,
                "targets": ["node1"],
                "procedure": "Collect live service evidence.",
            }
        ],
        "assertions": [
            {
                "id": "service-works",
                "probe": "service-probe",
                "dimension": "functionality",
                "pass_condition": "The service responds.",
                "fail_condition": "The service is confirmed unavailable.",
            }
        ],
    }


def test_semantic_plan_separates_probes_from_atomic_assertions() -> None:
    plan = SemanticPlanConfig.model_validate(_semantic_plan())

    assert plan.probes[0].id == "service-probe"
    assert plan.assertions[0].probe == "service-probe"


def test_semantic_plan_rejects_unknown_and_cyclic_prerequisites() -> None:
    unknown = _semantic_plan()
    unknown["assertions"][0]["requires"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown prerequisites"):
        SemanticPlanConfig.model_validate(unknown)

    cyclic = _semantic_plan()
    cyclic["assertions"].append(
        {
            "id": "dependent",
            "probe": "service-probe",
            "dimension": "functionality",
            "requires": ["service-works"],
            "pass_condition": "The dependent property holds.",
            "fail_condition": "The dependent property is confirmed absent.",
        }
    )
    cyclic["assertions"][0]["requires"] = ["dependent"]
    with pytest.raises(ValidationError, match="contain a cycle"):
        SemanticPlanConfig.model_validate(cyclic)


def test_semantic_plan_rejects_mutation_without_cleanup() -> None:
    value = _semantic_plan()
    value["probes"][0]["effect"] = "evaluator_owned_data"

    with pytest.raises(ValidationError, match="must define cleanup"):
        SemanticPlanConfig.model_validate(value)
