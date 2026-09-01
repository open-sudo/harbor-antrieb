import pytest
from pydantic import ValidationError

from harbor_antrieb.config import AntriebDefinition


def test_infraset_definition_requires_cluster() -> None:
    with pytest.raises(ValidationError, match="at least one image"):
        AntriebDefinition(cluster=[])


def test_static_prepare_is_disabled_by_default() -> None:
    definition = AntriebDefinition(cluster=["ubuntu24.04"])

    assert definition.initialize == []
    assert definition.prepare.enabled is False
    assert definition.max_clusters == 1
    assert definition.base_runbooks == ["antrieb/primer"]


def test_base_runbooks_reject_scenario_or_non_antrieb_documents() -> None:
    with pytest.raises(ValidationError, match="primers and image references"):
        AntriebDefinition(
            cluster=["ubuntu24.04"],
            base_runbooks=["antrieb/primer", "antrieb/vyos-dnat-port-forward"],
        )

    with pytest.raises(ValidationError, match="must include antrieb/primer"):
        AntriebDefinition(
            cluster=["ubuntu24.04"],
            base_runbooks=["antrieb/networking-primer"],
        )


def test_cluster_quota_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        AntriebDefinition(cluster=["ubuntu24.04"], max_clusters=0)


def test_initialize_requires_unique_safe_names() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        AntriebDefinition(cluster=["rhel9.8"], initialize=["rhsm", "rhsm"])

    with pytest.raises(ValidationError, match="invalid initializer names"):
        AntriebDefinition(cluster=["rhel9.8"], initialize=["RHSM subscription"])
