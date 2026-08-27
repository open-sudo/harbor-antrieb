from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    def validate_cluster(self) -> "AntriebDefinition":
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


class DimensionConfig(BaseModel):
    weight: float = Field(gt=0)
    description: str


class JudgeConfig(BaseModel):
    agent: str = "claude-code"
    model: str | None = None
    reasoning_effort: str = "medium"
    timeout_sec: int = 900
    allow_ssh_probes: bool = False
    parallel_node_collection: bool = True
    node_collector_timeout_sec: int = Field(default=300, gt=0)
    node_collector_command_budget: int = Field(default=8, ge=1, le=32)
    coordinator_command_budget: int = Field(default=64, ge=4, le=128)
    dimensions: dict[str, DimensionConfig]

    @model_validator(mode="after")
    def validate_weights(self) -> "JudgeConfig":
        invalid_names = [
            name
            for name in self.dimensions
            if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) is None
        ]
        if invalid_names:
            raise ValueError(f"invalid judge dimension names: {invalid_names}")
        total = sum(dimension.weight for dimension in self.dimensions.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"judge dimension weights must total 1.0, got {total}")
        return self


class SemanticCommandConfig(BaseModel):
    """One deterministic task-authored command in a semantic probe."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node: str
    command: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_command(self) -> "SemanticCommandConfig":
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.id) is None:
            raise ValueError(f"invalid semantic command id: {self.id!r}")
        return self


class SemanticProbeConfig(BaseModel):
    """One bounded evidence-collection procedure."""

    model_config = ConfigDict(extra="forbid")

    id: str
    level: int = Field(default=1, ge=1, le=10)
    targets: list[str] = Field(min_length=1)
    effect: Literal[
        "read_only",
        "evaluator_owned_data",
        "controlled_failure",
        "reboot",
    ] = "read_only"
    max_exec_calls: int = Field(default=3, ge=1, le=16)
    procedure: str = Field(min_length=1)
    cleanup: str | None = None
    commands: list[SemanticCommandConfig] = Field(default_factory=list)
    cleanup_commands: list[SemanticCommandConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_probe(self) -> "SemanticProbeConfig":
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.id) is None:
            raise ValueError(f"invalid semantic probe id: {self.id!r}")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError(f"semantic probe {self.id!r} has duplicate targets")
        command_ids = [command.id for command in self.commands]
        cleanup_ids = [command.id for command in self.cleanup_commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError(f"semantic probe {self.id!r} has duplicate command ids")
        if len(cleanup_ids) != len(set(cleanup_ids)):
            raise ValueError(
                f"semantic probe {self.id!r} has duplicate cleanup command ids"
            )
        all_command_ids = command_ids + cleanup_ids
        if len(all_command_ids) != len(set(all_command_ids)):
            raise ValueError(
                f"semantic probe {self.id!r} has duplicate command ids across phases"
            )
        unknown_command_nodes = sorted(
            {
                command.node
                for command in [
                    *self.commands,
                    *self.cleanup_commands,
                ]
                if command.node not in self.targets
            }
        )
        if unknown_command_nodes:
            raise ValueError(
                f"semantic probe {self.id!r} commands reference non-target nodes: "
                f"{unknown_command_nodes}"
            )
        if self.effect == "controlled_failure" and self.level < 6:
            raise ValueError(
                f"controlled-failure probe {self.id!r} must be level 6 or higher"
            )
        if self.effect == "reboot" and self.level < 8:
            raise ValueError(f"reboot probe {self.id!r} must be level 8 or higher")
        if self.effect != "read_only" and not self.cleanup:
            raise ValueError(f"mutating semantic probe {self.id!r} must define cleanup")
        return self


class SemanticAssertionConfig(BaseModel):
    """One atomic scored claim supported by a probe."""

    model_config = ConfigDict(extra="forbid")

    id: str
    probe: str
    dimension: str
    points: float = Field(default=1.0, gt=0)
    critical: bool = False
    requires: list[str] = Field(default_factory=list)
    pass_condition: str = Field(min_length=1)
    fail_condition: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_assertion(self) -> "SemanticAssertionConfig":
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.id) is None:
            raise ValueError(f"invalid semantic assertion id: {self.id!r}")
        if len(self.requires) != len(set(self.requires)):
            raise ValueError(
                f"semantic assertion {self.id!r} has duplicate prerequisites"
            )
        if self.id in self.requires:
            raise ValueError(f"semantic assertion {self.id!r} cannot require itself")
        return self


class SemanticPlanConfig(BaseModel):
    """Task-owned probes and assertions; weights remain in judge.toml."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    probes: list[SemanticProbeConfig] = Field(min_length=1)
    assertions: list[SemanticAssertionConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "SemanticPlanConfig":
        probe_ids = [probe.id for probe in self.probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("semantic probe ids must be unique")
        assertion_ids = [assertion.id for assertion in self.assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("semantic assertion ids must be unique")
        probes = {probe.id: probe for probe in self.probes}
        assertions = {assertion.id: assertion for assertion in self.assertions}
        unknown_probes = sorted(
            {assertion.probe for assertion in self.assertions} - set(probes)
        )
        if unknown_probes:
            raise ValueError(
                f"semantic assertions reference unknown probes: {unknown_probes}"
            )
        unused_probes = sorted(
            set(probes) - {assertion.probe for assertion in self.assertions}
        )
        if unused_probes:
            raise ValueError(f"semantic probes have no assertions: {unused_probes}")
        unknown_requirements = sorted(
            {
                required
                for assertion in self.assertions
                for required in assertion.requires
            }
            - set(assertions)
        )
        if unknown_requirements:
            raise ValueError(
                "semantic assertions reference unknown prerequisites: "
                f"{unknown_requirements}"
            )
        for assertion in self.assertions:
            level = probes[assertion.probe].level
            later = [
                required
                for required in assertion.requires
                if probes[assertions[required].probe].level > level
            ]
            if later:
                raise ValueError(
                    f"semantic assertion {assertion.id!r} requires deeper "
                    f"assertions: {later}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(assertion_id: str) -> None:
            if assertion_id in visited:
                return
            if assertion_id in visiting:
                raise ValueError("semantic assertion prerequisites contain a cycle")
            visiting.add(assertion_id)
            for required in assertions[assertion_id].requires:
                visit(required)
            visiting.remove(assertion_id)
            visited.add(assertion_id)

        for assertion_id in assertions:
            visit(assertion_id)
        if not any(probe.level == 1 for probe in self.probes):
            raise ValueError("semantic plan must contain at least one level-1 probe")
        return self
