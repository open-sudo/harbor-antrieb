from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, Literal, override

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from pydantic import BaseModel, ConfigDict

from infraset.agent_runner import run_structured_agent
from infraset.config import (
    JudgeConfig,
    SemanticAssertionConfig,
    SemanticCommandConfig,
    SemanticPlanConfig,
    SemanticProbeConfig,
)
from infraset.evaluation_policy import DISABLED_EVALUATION_DIMENSIONS
from infraset.runbooks import render_platform_references

__all__ = ["InfraSetVerifier"]


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    node: str
    command: str
    observation: str


class _ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    summary: str
    cleanup_completed: bool
    limitations: list[str]


class _AssertionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    status: Literal["pass", "fail", "indeterminate"]
    summary: str
    evidence: list[_Evidence]
    failures: list[str]
    limitations: list[str]


class _SemanticReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    probes: list[_ProbeResult]
    assertions: list[_AssertionResult]
    actions_performed: list[str]
    overall_summary: str


def _report_schema() -> dict[str, Any]:
    evidence = {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
            "command": {"type": "string"},
            "observation": {"type": "string"},
        },
        "required": ["node", "command", "observation"],
        "additionalProperties": False,
    }
    probe = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "summary": {"type": "string"},
            "cleanup_completed": {"type": "boolean"},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "summary", "cleanup_completed", "limitations"],
        "additionalProperties": False,
    }
    assertion = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pass", "fail", "indeterminate"],
            },
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": evidence},
            "failures": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "id",
            "status",
            "summary",
            "evidence",
            "failures",
            "limitations",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "probes": {"type": "array", "items": probe},
            "assertions": {"type": "array", "items": assertion},
            "actions_performed": {"type": "array", "items": {"type": "string"}},
            "overall_summary": {"type": "string"},
        },
        "required": [
            "probes",
            "assertions",
            "actions_performed",
            "overall_summary",
        ],
        "additionalProperties": False,
    }


class InfraSetVerifier(BaseVerifier):
    """Collect semantic evidence and score atomic authored assertions."""

    def __init__(
        self,
        *,
        plan: str = "verifier/checks.toml",
        rubric: str = "verifier/judge.toml",
        agent: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        level: int | str = 10,
        minimum_coverage: float | str = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.plan = plan
        self.rubric = rubric
        self.agent = agent
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        if isinstance(level, bool):
            raise ValueError("semantic verification level must be from 1 to 10")
        try:
            self.level = int(level)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "semantic verification level must be from 1 to 10"
            ) from exc
        if not 1 <= self.level <= 10:
            raise ValueError("semantic verification level must be from 1 to 10")
        if isinstance(minimum_coverage, bool):
            raise ValueError("minimum coverage must be a number from 0 to 1")
        try:
            self.minimum_coverage = float(minimum_coverage)
        except (TypeError, ValueError) as exc:
            raise ValueError("minimum coverage must be a number from 0 to 1") from exc
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum coverage must be a number from 0 to 1")

    def _task_path(self, relative: str, label: str) -> Path:
        task_root = self.task.paths.task_dir.resolve()
        path = (task_root / relative).resolve()
        if not path.is_relative_to(task_root):
            raise ValueError(f"{label} path must remain inside the task directory")
        if not path.is_file():
            raise FileNotFoundError(f"InfraSet {label} not found: {path}")
        return path

    def _load(self) -> tuple[SemanticPlanConfig, JudgeConfig]:
        plan = SemanticPlanConfig.model_validate(
            tomllib.loads(self._task_path(self.plan, "semantic plan").read_text())
        )
        rubric = JudgeConfig.model_validate(
            tomllib.loads(self._task_path(self.rubric, "judge rubric").read_text())
        )
        unknown = sorted(
            {assertion.dimension for assertion in plan.assertions}
            - set(rubric.dimensions)
        )
        if unknown:
            raise ValueError(
                f"semantic assertions reference unknown dimensions: {unknown}"
            )
        return plan, rubric

    def _environment_state(self) -> tuple[str, tuple[str, ...], str]:
        session_id = getattr(self.environment, "remote_session_id", None)
        nodes = tuple(getattr(self.environment, "nodes", ()))
        endpoint = getattr(self.environment, "endpoint", None)
        if not session_id or not nodes or not endpoint:
            raise TypeError(
                "InfraSetVerifier requires a running InfraSetEnvironment"
            )
        return str(session_id), nodes, str(endpoint)

    @staticmethod
    def _probe_text(
        index: int,
        probe: SemanticProbeConfig,
        assertions: list[SemanticAssertionConfig],
    ) -> str:
        cleanup = probe.cleanup or "No evaluator mutation is permitted."
        cleanup_budget = (
            f" {len(probe.targets)} additional exec call(s), one per target, are "
            "reserved only for cleanup."
            if probe.cleanup
            else ""
        )
        assertion_text = "\n".join(
            f"  Assertion {item.id}:\n"
            f"    Pass only when: {item.pass_condition}\n"
            f"    Fail only when: {item.fail_condition}\n"
            f"    Prerequisites: {', '.join(item.requires) or 'none'}"
            for item in assertions
        )
        command_text = "\n".join(
            f"  {command.id} on {command.node}: {command.command}"
            for command in probe.commands
        ) or "  none"
        cleanup_command_text = "\n".join(
            f"  {command.id} on {command.node}: {command.command}"
            for command in probe.cleanup_commands
        ) or "  none"
        return f"""Probe {index}: {probe.id}
Level: {probe.level}; targets: {", ".join(probe.targets)}
Effect: {probe.effect}; evidence command budget: {probe.max_exec_calls}.{cleanup_budget}
Procedure: {probe.procedure}
Cleanup: {cleanup}
Deterministic authored commands (run by Harbor before the AI probe):
{command_text}
Deterministic authored cleanup commands (run by Harbor after the AI probe):
{cleanup_command_text}
Atomic assertions supported by this probe:
{assertion_text}"""

    @staticmethod
    def _exec_call_budget(probes: list[SemanticProbeConfig]) -> int:
        return sum(
            probe.max_exec_calls + (len(probe.targets) if probe.cleanup else 0)
            for probe in probes
        )

    @staticmethod
    def _format_command_output(value: str | None, limit: int = 20_000) -> str:
        text = value or ""
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n[output truncated at {limit} characters]"

    async def _run_authored_commands(
        self,
        commands: list[SemanticCommandConfig],
        *,
        phase: str,
        audit_path: Path,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run deterministic task commands and append redacted job audit records."""
        results: list[dict[str, Any]] = []
        all_succeeded = True
        environment_exec = getattr(self.environment, "exec_on_node", None)
        if not callable(environment_exec):
            raise TypeError("InfraSetVerifier requires exec_on_node support")

        for command in commands:
            record: dict[str, Any] = {
                "phase": phase,
                "id": command.id,
                "node": command.node,
                "command": command.command,
            }
            try:
                result = await environment_exec(command.node, command.command)
                record.update(
                    {
                        "return_code": result.return_code,
                        "stdout": self._format_command_output(result.stdout),
                        "stderr": self._format_command_output(result.stderr),
                    }
                )
                succeeded = result.return_code == 0
            except Exception as exc:
                record.update(
                    {
                        "return_code": None,
                        "stdout": "",
                        "stderr": f"{type(exc).__name__}: {exc}",
                    }
                )
                succeeded = False
            record["succeeded"] = succeeded
            with audit_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            results.append(record)
            all_succeeded = all_succeeded and succeeded
        return results, all_succeeded

    def _prompt(
        self,
        probes: list[SemanticProbeConfig],
        assertions: list[SemanticAssertionConfig],
        nodes: tuple[str, ...],
        authored_evidence: str,
    ) -> str:
        by_probe = {
            probe.id: [item for item in assertions if item.probe == probe.id]
            for probe in probes
        }
        plan_text = "\n\n".join(
            self._probe_text(index, probe, by_probe[probe.id])
            for index, probe in enumerate(probes, 1)
        )
        return f"""You are executing an authored semantic verification plan for an
infrastructure benchmark. Execute every evidence probe below in order using only the
managed exec tool. A probe may efficiently collect evidence for several assertions,
but each assertion is an independent decision. Do not assign scores and do not let
one failed assertion determine another assertion from the same probe.

Security evaluation is disabled by harness policy. Do not add security probes or
spend commands evaluating security beyond the authored assertions below.

Use the least invasive commands that establish the stated outcomes. Address target
nodes directly through managed exec. Prefer node-local service clients, local names,
and the node's normal resolver over discovering or hardcoding service IP addresses.
Only inspect or compare IP addresses, interfaces, or CIDRs when the authored task
explicitly makes network topology or address behavior part of the requirement; in
that case, derive values from live configuration and never assume a literal subnet.
Keep implementation-only addresses out of assertion conclusions. Balance equivalent
work across targets. Accept equivalent secure packaging and configuration layouts.
Do not fix executor work. For identity-bound protocols such as Kerberos or TLS, preserve
the canonical service name established by DNS and live configuration; do not replace
it with a short managed-node label, and validate the intended service identity before
classifying an authentication failure. Only create evaluator-owned test data.
Perform a mutation, controlled failure, or reboot solely when its probe explicitly
authorizes that effect, and complete cleanup before moving on.

Managed exec begins as the managed login identity and provides non-interactive
privilege escalation. When evidence requires protected local administrative access,
use privilege-aware discovery: inspect root-owned configuration through sudo and run
service clients with the privileged identity's correct home (for example, sudo -H)
or an explicitly discovered protected client configuration. Never print credentials
or secret file contents. A path that the login identity cannot read is not evidence
that the path or authentication method is absent. Validate one protected access
method before applying it across equivalent nodes.

Never use SSH, SCP, or another nested remote shell between managed nodes. Managed
exec already addresses every node independently, and inter-node login is not part of
the harness contract. When a protected client credential exists on one managed node,
run the network client from that node and target the required remote service directly
instead of copying or fetching the credential through another node.

Return every selected probe and assertion exactly once, in the authored order and
with exact IDs. Classification for each assertion is strict:
- pass: concrete live evidence satisfies that assertion's pass condition;
- fail: concrete live evidence establishes that assertion's fail condition;
- indeterminate: access, prerequisites, tooling, timeout, lease, ambiguous output,
  command construction, cleanup, or another evaluator limitation prevents a
  defensible independent decision.

Missing evidence and evaluator command errors are indeterminate, not executor
failures. For pass or fail, include exact commands and observations. Put confirmed
executor defects in failures and evaluator limitations in limitations. When an
assertion's prerequisite did not pass, return the dependent assertion as
indeterminate. In each evidence object, node is the single managed exec location;
when evidence was collected separately on several nodes, emit one evidence object
per node rather than a comma list or source-to-destination label. Probe
cleanup_completed is true only after evaluator-created state is
removed and changed services or nodes are restored; use true for read-only probes.
Each mutating probe has one additional exec call per target reserved exclusively for
cleanup. Do not spend those reserved calls on evidence collection or diagnosis. Use
managed exec independently on the relevant targets and combine cleanup with its
local confirmation when practical.
Once every probe and cleanup operation is complete, close live evidence and compose
the final JSON report offline.

Verification level: {self.level}/10
Managed nodes: {", ".join(nodes)}

Provider-maintained platform references describe the control surface and current
appliance syntax. They are not evidence of executor success:
{render_platform_references(self.environment)}

Candidate task:
{self.task.instruction}

Authored evidence plan:
{plan_text}

Deterministic authored command results:
{authored_evidence or "No deterministic authored commands were defined."}
"""

    @staticmethod
    def _execution_nodes(label: str, known_nodes: set[str]) -> set[str]:
        """Extract managed exec locations from a compact evidence label."""
        source = re.split(r"\s*(?:→|->)\s*|\s+to\s+", label, maxsplit=1)[0].strip()
        listed = {
            item.strip()
            for item in re.split(r"\s*(?:,|/|&|\band\b)\s*", source)
            if item.strip()
        }
        if len(listed) > 1:
            return listed
        claimed = {
            node
            for node in known_nodes
            if re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(node)}(?![A-Za-z0-9_-])",
                source,
            )
        }
        return claimed or {source}

    @staticmethod
    def _make_indeterminate(
        result: _AssertionResult,
        limitation: str,
    ) -> None:
        result.status = "indeterminate"
        result.failures = []
        if limitation not in result.limitations:
            result.limitations.append(limitation)

    @classmethod
    def _validate_report(
        cls,
        value: dict[str, Any],
        probes: list[SemanticProbeConfig],
        assertions: list[SemanticAssertionConfig],
    ) -> None:
        report = _SemanticReport.model_validate(value)
        expected_probes = [probe.id for probe in probes]
        actual_probes = [result.id for result in report.probes]
        if actual_probes != expected_probes:
            raise ValueError(
                "semantic report probe order differs: "
                f"expected={expected_probes}, actual={actual_probes}"
            )
        expected_assertions = [assertion.id for assertion in assertions]
        actual_assertions = [result.id for result in report.assertions]
        if actual_assertions != expected_assertions:
            raise ValueError(
                "semantic report assertion order differs: "
                f"expected={expected_assertions}, actual={actual_assertions}"
            )

        probe_configs = {probe.id: probe for probe in probes}
        known_nodes = {node for probe in probes for node in probe.targets}
        probe_results = {result.id: result for result in report.probes}
        assertion_results = {result.id: result for result in report.assertions}
        for assertion in assertions:
            result = assertion_results[assertion.id]
            probe = probe_configs[assertion.probe]
            if result.status in {"pass", "fail"} and not result.evidence:
                raise ValueError(f"semantic assertion {assertion.id!r} lacks evidence")
            cited_nodes = {
                node
                for item in result.evidence
                for node in cls._execution_nodes(item.node, known_nodes)
            }
            foreign = sorted(cited_nodes - set(probe.targets))
            if foreign:
                raise ValueError(
                    f"semantic assertion {assertion.id!r} cites nodes outside "
                    f"probe {probe.id!r}: {foreign}"
                )
            if result.status == "fail" and not result.failures:
                raise ValueError(
                    f"failed semantic assertion {assertion.id!r} lacks failures"
                )
            if result.status == "indeterminate" and not result.limitations:
                raise ValueError(
                    f"indeterminate semantic assertion {assertion.id!r} lacks "
                    "limitations"
                )
            if (
                probe.effect != "read_only"
                and not probe_results[probe.id].cleanup_completed
            ):
                cls._make_indeterminate(
                    result,
                    f"evaluator cleanup for probe {probe.id!r} was not completed",
                )

        changed = True
        while changed:
            changed = False
            for assertion in assertions:
                result = assertion_results[assertion.id]
                unmet = [
                    required
                    for required in assertion.requires
                    if assertion_results[required].status != "pass"
                ]
                if unmet and result.status != "indeterminate":
                    cls._make_indeterminate(
                        result,
                        f"prerequisite assertions did not pass: {', '.join(unmet)}",
                    )
                    changed = True
        value.clear()
        value.update(report.model_dump())

    @staticmethod
    def _aggregate(
        assertions: list[SemanticAssertionConfig],
        report: dict[str, Any],
        rubric: JudgeConfig,
        level: int,
        minimum_coverage: float,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        results = {item["id"]: item for item in report["assertions"]}
        details: dict[str, dict[str, Any]] = {}
        total_selected = sum(assertion.points for assertion in assertions)
        total_evaluated = 0.0
        for name in rubric.dimensions:
            members = [
                assertion for assertion in assertions if assertion.dimension == name
            ]
            selected = sum(assertion.points for assertion in members)
            evaluated = sum(
                assertion.points
                for assertion in members
                if results[assertion.id]["status"] != "indeterminate"
            )
            passed = sum(
                assertion.points
                for assertion in members
                if results[assertion.id]["status"] == "pass"
            )
            total_evaluated += evaluated
            applicable = evaluated > 0
            score = 100 * passed / evaluated if applicable else 0.0
            details[name] = {
                "score": round(score, 2),
                "applicable": applicable,
                "critical_failure": any(
                    assertion.critical and results[assertion.id]["status"] == "fail"
                    for assertion in members
                ),
                "selected_points": selected,
                "evaluated_points": evaluated,
                "passed_points": passed,
                "coverage": round(evaluated / selected, 4) if selected else 0.0,
                "assertions": [
                    {
                        "id": assertion.id,
                        "probe": assertion.probe,
                        "points": assertion.points,
                        **results[assertion.id],
                    }
                    for assertion in members
                ],
            }
        applicable = [
            (name, dimension)
            for name, dimension in rubric.dimensions.items()
            if details[name]["applicable"]
        ]
        denominator = sum(dimension.weight for _, dimension in applicable)
        weighted = (
            sum(
                details[name]["score"] * dimension.weight
                for name, dimension in applicable
            )
            / denominator
            if denominator
            else 0.0
        )
        caps: list[dict[str, Any]] = []
        functionality = details.get("functionality")
        if (
            functionality
            and functionality["applicable"]
            and functionality["score"] < 40
        ):
            caps.append({"reason": "functionality below 40", "maximum": 40})
        if any(item["critical_failure"] for item in details.values()):
            caps.append({"reason": "critical confirmed failure", "maximum": 50})
        if caps:
            weighted = min(weighted, *(cap["maximum"] for cap in caps))

        provisional_reward = round(weighted / 100, 4)
        raw_coverage = total_evaluated / total_selected
        coverage = round(raw_coverage, 4)
        evaluation_complete = raw_coverage >= minimum_coverage or math.isclose(
            raw_coverage, minimum_coverage
        )
        report.update(
            {
                "evaluator": "semantic-ai",
                "verification_level": level,
                "minimum_coverage": minimum_coverage,
                "dimensions": details,
                "evaluated_dimensions": [name for name, _ in applicable],
                "unverified_dimensions": [
                    name
                    for name in rubric.dimensions
                    if not details[name]["applicable"]
                ],
                "evaluation_coverage": coverage,
                "full_coverage": math.isclose(total_evaluated, total_selected),
                "evaluation_complete": evaluation_complete,
                "publication_eligible": evaluation_complete,
                "aggregation": {
                    "weighted_score": round(weighted, 2),
                    "reward": provisional_reward if evaluation_complete else None,
                    "provisional_reward": provisional_reward,
                    "caps_applied": caps,
                },
            }
        )
        rewards: dict[str, float] = {
            "evaluation_coverage": coverage,
            "evaluation_complete": float(evaluation_complete),
            "publication_eligible": float(evaluation_complete),
        }
        rewards.update(
            {name: round(details[name]["score"] / 100, 4) for name, _ in applicable}
        )
        if evaluation_complete:
            rewards = {"reward": provisional_reward, **rewards}
        return report, rewards

    @override
    async def verify(self) -> VerifierResult:
        plan, rubric = self._load()
        probes = sorted(
            (probe for probe in plan.probes if probe.level <= self.level),
            key=lambda probe: probe.level,
        )
        assertions = [
            assertion
            for assertion in plan.assertions
            if assertion.probe in {probe.id for probe in probes}
            and assertion.dimension not in DISABLED_EVALUATION_DIMENSIONS
        ]
        selected_assertion_ids = {assertion.id for assertion in assertions}
        while True:
            independent = [
                assertion
                for assertion in assertions
                if set(assertion.requires) <= selected_assertion_ids
            ]
            independent_ids = {assertion.id for assertion in independent}
            if independent_ids == selected_assertion_ids:
                break
            assertions = independent
            selected_assertion_ids = independent_ids
        if not assertions:
            raise ValueError(
                f"semantic verification level {self.level} selects no assertions"
            )
        selected_probe_ids = {assertion.probe for assertion in assertions}
        probes = [probe for probe in probes if probe.id in selected_probe_ids]
        assert_active = getattr(self.environment, "assert_cluster_active", None)
        if callable(assert_active):
            assert_active()
        session_id, nodes, endpoint = self._environment_state()
        unknown = sorted(
            {node for probe in probes for node in probe.targets} - set(nodes)
        )
        if unknown:
            raise ValueError(f"semantic probes reference unknown nodes: {unknown}")
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self.trial_paths.verifier_dir / "evaluator-commands.jsonl"
        audit_path.write_text("")
        authored_results: dict[str, list[dict[str, Any]]] = {}
        authored_evidence: list[str] = []
        for probe in probes:
            results, _ = await self._run_authored_commands(
                probe.commands,
                phase="authored_evidence",
                audit_path=audit_path,
            )
            authored_results[probe.id] = results
            for result in results:
                authored_evidence.append(json.dumps({"probe": probe.id, **result}))

        cleanup_success: dict[str, bool] = {}
        try:
            report, raw = await run_structured_agent(
                agent_name=self.agent or rubric.agent,
                model=self.model or rubric.model,
                prompt=self._prompt(
                    probes,
                    assertions,
                    nodes,
                    "\n".join(authored_evidence),
                ),
                schema=_report_schema(),
                session_id=session_id,
                nodes=nodes,
                endpoint=endpoint,
                workspace=self.trial_paths.verifier_dir,
                timeout_sec=rubric.timeout_sec,
                audit_path=audit_path,
                reasoning_effort=self.reasoning_effort or rubric.reasoning_effort,
                service_tier=self.service_tier,
                lease_expires_at=getattr(self.environment, "cluster_expires_at", None),
                allow_offline_finalization=True,
                output_validator=lambda value: self._validate_report(
                    value, probes, assertions
                ),
                max_exec_calls=self._exec_call_budget(probes),
            )
        finally:
            for probe in probes:
                cleanup_results, succeeded = await self._run_authored_commands(
                    probe.cleanup_commands,
                    phase="authored_cleanup",
                    audit_path=audit_path,
                )
                cleanup_success[probe.id] = succeeded
                authored_results.setdefault(probe.id, []).extend(cleanup_results)

        for probe in probes:
            if cleanup_success.get(probe.id, True):
                continue
            for probe_result in report["probes"]:
                if probe_result["id"] == probe.id:
                    probe_result["cleanup_completed"] = False
                    probe_result["limitations"].append(
                        "A deterministic authored cleanup command failed."
                    )
                    break
        raw_path = self.trial_paths.verifier_dir / "semantic-evaluator-agent-raw.json"
        raw_path.write_text(raw)
        if not audit_path.read_text().strip():
            raise RuntimeError("Semantic evaluator completed without an exec call")
        report, rewards = self._aggregate(
            assertions,
            report,
            rubric,
            self.level,
            self.minimum_coverage,
        )
        selected_plan = {
            "verification_level": self.level,
            "minimum_coverage": self.minimum_coverage,
            "disabled_dimensions": sorted(DISABLED_EVALUATION_DIMENSIONS),
            "probes": [probe.model_dump() for probe in probes],
            "assertions": [assertion.model_dump() for assertion in assertions],
        }
        (self.trial_paths.verifier_dir / "semantic-plan-selection.json").write_text(
            json.dumps(selected_plan, indent=2)
        )
        rendered = json.dumps(report, indent=2)
        (self.trial_paths.verifier_dir / "evaluation-report.json").write_text(rendered)
        (self.trial_paths.verifier_dir / "reward-details.json").write_text(rendered)
        self.trial_paths.reward_json_path.write_text(json.dumps(rewards, indent=2))
        return VerifierResult(rewards=rewards)
