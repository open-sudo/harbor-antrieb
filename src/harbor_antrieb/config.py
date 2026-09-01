from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PrepareConfig(BaseModel):
    enabled: bool = False
    mode: Literal["static", "ai"] = "static"
    setup: str = "prepare/setup.toml"
    baseline: str = "prepare/baseline.toml"
    agent: str | None = None
    model: str | None = None
    reasoning_effort: str = "medium"
    timeout_sec: int = Field(default=1200, gt=0)
    prompt: str = "prepare/prompt.md"


class AntriebDefinition(BaseModel):
    cluster: list[str | dict[str, Any]]
    max_clusters: int = Field(default=1, ge=1)
    initialize: list[str] = Field(default_factory=list, max_length=16)
    base_runbooks: list[str] = Field(
        default_factory=lambda: ["antrieb/primer"], max_length=8
    )
    networks: list[dict[str, Any]] | None = None
    nics: dict[str, list[dict[str, str]]] | None = None
    control_node: str = "node1"
    endpoint: str = "https://antrieb.sh/mcp"
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)

    @model_validator(mode="after")
    def validate_cluster(self) -> AntriebDefinition:
        if not self.cluster:
            raise ValueError("cluster must contain at least one image")
        if len(self.initialize) != len(set(self.initialize)):
            raise ValueError("initialize must not contain duplicates")
        invalid_initializers = [
            name
            for name in self.initialize
            if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None
        ]
        if invalid_initializers:
            raise ValueError(
                f"initialize contains invalid initializer names: {invalid_initializers}"
            )
        if "antrieb/primer" not in self.base_runbooks:
            raise ValueError("base_runbooks must include antrieb/primer")
        if len(self.base_runbooks) != len(set(self.base_runbooks)):
            raise ValueError("base_runbooks must not contain duplicates")
        invalid_runbooks = [
            name
            for name in self.base_runbooks
            if re.fullmatch(r"antrieb/(?:primer|[a-z0-9-]+-(?:primer|reference))", name)
            is None
        ]
        if invalid_runbooks:
            raise ValueError(
                "base_runbooks may contain only Antrieb primers and image "
                f"references, got {invalid_runbooks}"
            )
        return self
