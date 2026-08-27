from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseRunbook:
    """One task-declared, read-only Antrieb platform reference."""

    fq_name: str
    body: str


def platform_references(environment: Any) -> tuple[BaseRunbook, ...]:
    references = getattr(environment, "base_runbooks", ())
    if not isinstance(references, tuple) or not all(
        isinstance(reference, BaseRunbook) for reference in references
    ):
        raise TypeError(
            "InfraSet environment returned malformed base-runbook references"
        )
    return references


def render_platform_references(environment: Any) -> str:
    """Render transient base runbooks for an agent prompt without persisting them."""
    references = platform_references(environment)
    if not references:
        return "(No platform references were declared.)"
    return "\n\n".join(
        f"<platform-reference fq_name={reference.fq_name!r}>\n"
        f"{reference.body}\n"
        "</platform-reference>"
        for reference in references
    )
