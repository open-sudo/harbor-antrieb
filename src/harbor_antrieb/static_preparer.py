from __future__ import annotations

import asyncio
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from harbor_antrieb.config import PrepareConfig
from harbor_antrieb.errors import ClusterExpiredError


class PrepareCommandConfig(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    stage: int = Field(default=0, ge=0)
    node: str
    command: str
    retries: int = Field(default=1, ge=1, le=120)
    retry_delay_sec: float = Field(default=1.0, ge=0, le=60)


class BaselineObservationConfig(PrepareCommandConfig):
    required: bool = True


class StaticSetupConfig(BaseModel):
    timeout_sec: int = Field(default=600, gt=0)
    steps: list[PrepareCommandConfig]

    @model_validator(mode="after")
    def validate_steps(self) -> "StaticSetupConfig":
        _validate_command_ids(self.steps, "setup steps")
        if not self.steps:
            raise ValueError("static setup requires at least one step")
        return self


class StaticBaselineConfig(BaseModel):
    timeout_sec: int = Field(default=300, gt=0)
    observations: list[BaselineObservationConfig]

    @model_validator(mode="after")
    def validate_observations(self) -> "StaticBaselineConfig":
        _validate_command_ids(self.observations, "baseline observations")
        if not self.observations:
            raise ValueError("static baseline requires at least one observation")
        return self


def _validate_command_ids(commands: Sequence[PrepareCommandConfig], label: str) -> None:
    duplicate_ids = sorted(
        command_id
        for command_id in {command.id for command in commands}
        if sum(command.id == command_id for command in commands) > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate {label}: {duplicate_ids}")


@dataclass
class _CommandOutcome:
    command: PrepareCommandConfig
    passed: bool
    return_code: int | None
    stdout: str
    stderr: str
    attempts: int
    failure: str | None = None


class _StaticPrepareRunner:
    def __init__(self, environment: Any, output_dir: Path) -> None:
        self.environment = environment
        self.output_dir = output_dir
        self._audit_lock = asyncio.Lock()

    async def _append_audit(self, audit_path: Path, entry: dict[str, Any]) -> None:
        async with self._audit_lock:
            with audit_path.open("a") as audit_file:
                audit_file.write(json.dumps(entry, separators=(",", ":")) + "\n")

    async def _run_command(
        self,
        command: PrepareCommandConfig,
        audit_path: Path,
    ) -> _CommandOutcome:
        last_return_code: int | None = None
        last_stdout = ""
        last_stderr = ""
        failure: str | None = None
        for attempt in range(1, command.retries + 1):
            await self._append_audit(
                audit_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "id": command.id,
                    "stage": command.stage,
                    "node": command.node,
                    "command": command.command,
                    "attempt": attempt,
                    "outcome": "requested",
                },
            )
            try:
                result = await self.environment.exec_on_node(
                    command.node, command.command
                )
                last_return_code = result.return_code
                last_stdout = result.stdout or ""
                last_stderr = result.stderr or ""
                passed = result.return_code == 0
                failure = None if passed else f"exit code {result.return_code}"
            except ClusterExpiredError:
                await self._append_audit(
                    audit_path,
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "id": command.id,
                        "stage": command.stage,
                        "node": command.node,
                        "attempt": attempt,
                        "outcome": "cluster_expired",
                    },
                )
                raise
            except Exception as exc:
                passed = False
                failure = f"{type(exc).__name__}: {exc}"
                last_return_code = None
                last_stdout = ""
                last_stderr = ""
            await self._append_audit(
                audit_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "id": command.id,
                    "stage": command.stage,
                    "node": command.node,
                    "attempt": attempt,
                    "return_code": last_return_code,
                    "passed": passed,
                    "outcome": "completed",
                },
            )
            if passed:
                return _CommandOutcome(
                    command=command,
                    passed=True,
                    return_code=last_return_code,
                    stdout=last_stdout,
                    stderr=last_stderr,
                    attempts=attempt,
                )
            if attempt < command.retries:
                await asyncio.sleep(command.retry_delay_sec)
        return _CommandOutcome(
            command=command,
            passed=False,
            return_code=last_return_code,
            stdout=last_stdout,
            stderr=last_stderr,
            attempts=command.retries,
            failure=failure,
        )

    async def run_stages(
        self,
        commands: Sequence[PrepareCommandConfig],
        *,
        timeout_sec: int,
        audit_name: str,
        stop_after_failed_stage: bool = False,
    ) -> list[_CommandOutcome]:
        audit_path = self.output_dir / audit_name
        audit_path.write_text("")
        outcomes: list[_CommandOutcome] = []
        async with asyncio.timeout(timeout_sec):
            for stage in sorted({command.stage for command in commands}):
                stage_outcomes = await asyncio.gather(
                    *(
                        self._run_command(command, audit_path)
                        for command in commands
                        if command.stage == stage
                    )
                )
                outcomes.extend(stage_outcomes)
                if stop_after_failed_stage and any(
                    not outcome.passed for outcome in stage_outcomes
                ):
                    break
        return outcomes


def _load_task_config(
    task_root: Path,
    relative_path: str,
    model: type[StaticSetupConfig] | type[StaticBaselineConfig],
) -> StaticSetupConfig | StaticBaselineConfig:
    config_path = (task_root / relative_path).resolve()
    if not config_path.is_relative_to(task_root):
        raise ValueError("prepare configuration path must remain inside the task")
    if not config_path.is_file():
        raise FileNotFoundError(f"Antrieb prepare config not found: {config_path}")
    return model.model_validate(tomllib.loads(config_path.read_text()))


def _validate_nodes(
    commands: Sequence[PrepareCommandConfig], nodes: tuple[str, ...]
) -> None:
    unknown = sorted({command.node for command in commands} - set(nodes))
    if unknown:
        raise ValueError(f"prepare commands reference unknown nodes: {unknown}")


def _outcome_record(outcome: _CommandOutcome) -> dict[str, Any]:
    return {
        "id": outcome.command.id,
        "stage": outcome.command.stage,
        "node": outcome.command.node,
        "command": outcome.command.command,
        "passed": outcome.passed,
        "return_code": outcome.return_code,
        "stdout": outcome.stdout,
        "stderr": outcome.stderr,
        "attempts": outcome.attempts,
        "failure": outcome.failure,
    }


def _raise_failed_setup(outcomes: list[_CommandOutcome]) -> None:
    failed = [outcome for outcome in outcomes if not outcome.passed]
    if failed:
        summary = ", ".join(
            f"{outcome.command.id} on {outcome.command.node} "
            f"({outcome.failure or 'failed'})"
            for outcome in failed
        )
        raise RuntimeError(f"Antrieb static setup failed: {summary}")


def _baseline_report(outcomes: list[_CommandOutcome]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    limitations: list[str] = []
    for outcome in outcomes:
        observation = outcome.command
        assert isinstance(observation, BaselineObservationConfig)
        record = _outcome_record(outcome)
        record["required"] = observation.required
        observations.append(record)
        node = nodes.setdefault(
            observation.node,
            {"name": observation.node, "facts": [], "unavailable": []},
        )
        if outcome.passed:
            rendered = outcome.stdout.strip()
            if outcome.stderr.strip():
                rendered = "\n".join(
                    part for part in (rendered, outcome.stderr.strip()) if part
                )
            node["facts"].append(
                {
                    "category": observation.id,
                    "observation": rendered or "command succeeded with no output",
                    "command": observation.command,
                }
            )
        else:
            unavailable = f"{observation.id}: {outcome.failure or 'observation failed'}"
            node["unavailable"].append(unavailable)
            limitations.append(f"{observation.node}: {unavailable}")
    return {
        "source": "static-prepare",
        "nodes": list(nodes.values()),
        "topology_observations": [],
        "limitations": limitations,
        "observations": observations,
    }


def _prepare_output_dir(environment: Any) -> Path:
    output_dir = environment.trial_paths.trial_dir / "prepare"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_static_setup(environment: Any, config: PrepareConfig) -> None:
    task_root = environment.environment_dir.parent.resolve()
    output_dir = _prepare_output_dir(environment)
    runner = _StaticPrepareRunner(environment, output_dir)

    setup = _load_task_config(task_root, config.setup, StaticSetupConfig)
    assert isinstance(setup, StaticSetupConfig)
    _validate_nodes(setup.steps, tuple(environment.nodes))
    setup_outcomes = await runner.run_stages(
        setup.steps,
        timeout_sec=setup.timeout_sec,
        audit_name="setup-commands.jsonl",
        stop_after_failed_stage=True,
    )
    (output_dir / "setup-report.json").write_text(
        json.dumps(
            {
                "source": "static-prepare",
                "steps": list(map(_outcome_record, setup_outcomes)),
            },
            indent=2,
        )
    )
    _raise_failed_setup(setup_outcomes)


async def capture_static_baseline(environment: Any, config: PrepareConfig) -> None:
    task_root = environment.environment_dir.parent.resolve()
    output_dir = _prepare_output_dir(environment)
    runner = _StaticPrepareRunner(environment, output_dir)
    baseline = _load_task_config(task_root, config.baseline, StaticBaselineConfig)
    assert isinstance(baseline, StaticBaselineConfig)
    _validate_nodes(baseline.observations, tuple(environment.nodes))
    baseline_outcomes = await runner.run_stages(
        baseline.observations,
        timeout_sec=baseline.timeout_sec,
        audit_name="baseline-commands.jsonl",
    )
    report = _baseline_report(baseline_outcomes)
    (output_dir / "baseline-report.json").write_text(json.dumps(report, indent=2))

    failed_required = [
        outcome
        for outcome in baseline_outcomes
        if not outcome.passed
        and isinstance(outcome.command, BaselineObservationConfig)
        and outcome.command.required
    ]
    if failed_required:
        summary = ", ".join(
            f"{outcome.command.id} on {outcome.command.node}"
            for outcome in failed_required
        )
        raise RuntimeError(f"Antrieb required baseline observations failed: {summary}")


async def run_static_prepare(environment: Any, config: PrepareConfig) -> None:
    await run_static_setup(environment, config)
    await capture_static_baseline(environment, config)
