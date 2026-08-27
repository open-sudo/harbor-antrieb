from __future__ import annotations

from typing import Any


class ClusterExpiredError(RuntimeError):
    """The provider-managed cluster lease has expired."""


def is_cluster_expired(value: Any) -> bool:
    """Recognize Antrieb's stable cluster-expiration error in wrapped payloads."""
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return "cluster_expired" in normalized
    if isinstance(value, dict):
        return any(is_cluster_expired(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(is_cluster_expired(item) for item in value)
    return False
