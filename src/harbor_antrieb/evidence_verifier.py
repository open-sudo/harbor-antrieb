from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, override

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from pydantic import BaseModel, ConfigDict, Field

from harbor_antrieb.agent_runner import run_structured_log_agent
from harbor_antrieb.collector import AntriebCollector
from harbor_antrieb.exec_bridge import redact_data, redact_text

_TRACE_COMMAND_LIMIT = 8 * 1024
_TRACE_OUTPUT_LIMIT = 4 * 1024
_PROMPT_TRACE_LIMIT = 180 * 1024


class RequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    requirement: str
    status: Literal[
        "satisfied", "partially_satisfied", "not_satisfied", "indeterminate"
    ]
    summary: str
    evidence_ids: list[str]


class HygieneAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    score: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence_ids: list[str] = Field(min_length=1)


class EvidenceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requirements: list[RequirementAssessment]
    confidence: float = Field(ge=0.0, le=1.0)
    operational_hygiene: HygieneAssessment
    overall_summary: str
    limitations: list[str]


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = max(1, limit // 2)
    return (
        value[:half]
        + f"\n...[TRUNCATED {len(value) - (half * 2)} CHARACTERS]...\n"
        + value[-half:]
    )


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_data(value), indent=2) + "\n")


def _terminal_trace(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    requested: dict[str, dict[str, Any]] = {}
    malformed = 0
    if path.is_file():
        for index, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(loaded, dict):
                malformed += 1
                continue
            command_id = loaded.get("command_id")
            if not isinstance(command_id, str) or not command_id:
                command_id = f"legacy-cmd-{index:04d}"
            loaded = {**loaded, "command_id": command_id}
            if loaded.get("outcome") == "requested":
                requested[command_id] = loaded
                continue
            records.append(loaded)

    terminal_ids = {str(record["command_id"]) for record in records}
    for command_id, request in requested.items():
        if command_id not in terminal_ids:
            records.append(
                {
                    **request,
                    "outcome": "unfinished",
                    "return_code": None,
                    "error": "No terminal command record was captured.",
                }
            )

    compact: list[dict[str, Any]] = []
    successful = failed = indeterminate = 0
    for record in records:
        outcome = str(record.get("outcome", "unknown"))
        return_code = record.get("return_code")
        if outcome == "completed" and return_code == 0:
            successful += 1
        elif outcome == "completed" and isinstance(return_code, int):
            failed += 1
        else:
            indeterminate += 1
        compact_record = {
            key: record[key]
            for key in (
                "command_id",
                "executor_attempt",
                "timestamp",
                "node",
                "outcome",
                "return_code",
                "duration_ms",
            )
            if key in record
        }
        for key, limit in (
            ("command", _TRACE_COMMAND_LIMIT),
            ("stdout", _TRACE_OUTPUT_LIMIT),
            ("stderr", _TRACE_OUTPUT_LIMIT),
            ("error", _TRACE_OUTPUT_LIMIT),
        ):
            value = record.get(key)
            if isinstance(value, str) and value:
                compact_record[key] = _bounded(redact_text(value), limit)
        compact.append(compact_record)
    return compact, {
        "total": len(compact),
        "successful": successful,
        "failed": failed,
        "indeterminate": indeterminate,
        "malformed_audit_records": malformed,
    }


def _selected_evidence(
    executor_report: dict[str, Any], valid_command_ids: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    limitations: list[str] = []
    raw_evidence = executor_report.get("evidence", [])
    if not isinstance(raw_evidence, list):
        return [], ["The executor evidence field was not an array."]
    for index, item in enumerate(raw_evidence, 1):
        if not isinstance(item, dict):
            limitations.append(f"Executor evidence item {index} was not an object.")
            continue
        command_ids = [
            value
            for value in item.get("command_ids", [])
            if isinstance(value, str)
        ]
        unknown = sorted(set(command_ids) - valid_command_ids)
        if unknown:
            limitations.append(
                f"Executor evidence item {index} referenced unknown command IDs: "
                + ", ".join(unknown)
            )
        selected.append(
            {
                "requirement": str(item.get("requirement", "")),
                "command_ids": [value for value in command_ids if value in valid_command_ids],
                "summary": str(item.get("summary", "")),
            }
        )
    return selected, limitations


def _global_evidence_ids(report: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for phase in ("before_prepare", "after_prepare", "after_executor"):
        phase_report = report.get(phase, {})
        if not isinstance(phase_report, dict):
            continue
        for node in phase_report.get("nodes", []):
            if not isinstance(node, dict):
                continue
            for observation in node.get("observations", []):
                if isinstance(observation, dict) and isinstance(
                    observation.get("id"), str
                ):
                    values.add(observation["id"])
    return values


def _topology(environment: Any) -> dict[str, Any]:
    definition = getattr(environment, "definition", None)
    if definition is not None and hasattr(definition, "model_dump"):
        authored = definition.model_dump(mode="json")
    else:
        authored = {}
    return redact_data(
        {
            "managed_nodes": list(getattr(environment, "nodes", ())),
            "node_images": getattr(environment, "node_images", {}),
            "authored_environment": authored,
        }
    )


def _prompt_trace(trace: list[dict[str, Any]]) -> str:
    rendered = json.dumps(trace, separators=(",", ":"))
    if len(rendered) <= _PROMPT_TRACE_LIMIT:
        return rendered
    reduced: list[dict[str, Any]] = []
    for record in trace:
        item = dict(record)
        for key in ("stdout", "stderr", "error"):
            if isinstance(item.get(key), str):
                item[key] = _bounded(item[key], 600)
        if isinstance(item.get("command"), str):
            item["command"] = _bounded(item["command"], 2_000)
        reduced.append(item)
    rendered = json.dumps(reduced, separators=(",", ":"))
    if len(rendered) <= _PROMPT_TRACE_LIMIT:
        return rendered
    metadata_only = [
        {
            key: value
            for key, value in item.items()
            if key not in {"stdout", "stderr", "error"}
        }
        for item in reduced
    ]
    return _bounded(json.dumps(metadata_only, separators=(",", ":")), _PROMPT_TRACE_LIMIT)


def _score_requirements(requirements: list[RequirementAssessment]) -> float:
    values = {
        "satisfied": 1.0,
        "partially_satisfied": 0.5,
        "not_satisfied": 0.0,
    }
    determinate = [values[item.status] for item in requirements if item.status in values]
    return round(sum(determinate) / len(determinate), 4) if determinate else 0.0


class EvidenceVerifier(BaseVerifier):
    """Score task outcomes from executor evidence and fixed global observations."""

    def __init__(
        self,
        *,
        agent: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        evaluator_timeout_sec: int | str = 600,
        minimum_coverage: float | str = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent or "codex"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.evaluator_timeout_sec = int(evaluator_timeout_sec)
        self.minimum_coverage = float(minimum_coverage)
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum coverage must be a number from 0 to 1")

    def _evaluation_prompt(
        self,
        *,
        executor_report: dict[str, Any],
        evidence_record: dict[str, Any],
        global_report: dict[str, Any],
        task_baseline: dict[str, Any],
    ) -> str:
        return f"""You are the independent outcome verifier for an infrastructure task.
You have no tools and cannot inspect or modify the live systems. Evaluate only the
captured evidence below.

Treat the public task instruction as the contract a human supervisor gave the
executor. Derive a concise list of its material outcomes; do not invent extra
requirements or require one particular implementation. For each material outcome,
classify the result as satisfied, partially_satisfied, not_satisfied, or
indeterminate. Cite only evidence IDs included below. A final executor claim is not
evidence by itself. A nonzero command exit is not automatically a task failure: it
may be expected discovery or a failed attempt later corrected. Prefer final,
directly observed state and behavior.

For requirements concerning availability, failover, restart, reboot, recovery,
resynchronization, or healing, configuration is supporting evidence only. Mark the
outcome satisfied only when captured commands directly show the healthy baseline,
behavior during a bounded disruption, restoration, and the recovered state. Without
that transition, score supporting configuration as at most partially_satisfied. Use
indeterminate when a safe transition could not be performed or observed, and
not_satisfied when a valid transition directly demonstrates broken behavior.

Lifecycle observations are fixed supplemental data, not task-specific probes. Use
the before-prepare and after-prepare snapshots to separate preparer-created state
from executor changes, then compare after-prepare with after-executor. Penalize
hygiene only for executor-caused
residue, abandoned files, conflicting or failed services, unsafe exposure, damaged
package state, or similarly unwanted changes. Do not penalize pre-existing state or
changes required by the task. Score operational hygiene from 0 to 1 and cite global
or executor evidence IDs.

Confidence is evidence quality, not task success. Use 1 only when all material
outcomes have direct, coherent evidence; lower it for indirect, contradictory, or
missing evidence. State limitations plainly.

Task instruction:
{self.task.instruction}

Managed topology:
{json.dumps(_topology(self.environment), separators=(",", ":"))}

Task preparation baseline, when authored:
{json.dumps(task_baseline, separators=(",", ":"))}

Executor final report and selected evidence:
{json.dumps(executor_report, separators=(",", ":"))}

Executor command statistics and evidence-selection limitations:
{json.dumps({key: evidence_record[key] for key in ("command_stats", "selected_evidence", "limitations")}, separators=(",", ":"))}

Complete executor command timeline (captured outputs may be bounded):
{_prompt_trace(evidence_record["commands"])}

Fixed lifecycle snapshots and baseline-relative differences:
{json.dumps(global_report, separators=(",", ":"))}
"""

    @override
    async def verify(self) -> VerifierResult:
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        collector = getattr(self.environment, "collector", None)
        load_bundle = getattr(collector, "load_bundle", None)
        if not callable(load_bundle):
            collector = AntriebCollector(self.trial_paths.trial_dir)
            load_bundle = collector.load_bundle
        bundle = load_bundle()
        baseline = bundle["after_prepare"]
        post = bundle["after_executor"]
        global_report = {
            "schema_version": 1,
            "collector_attempt": bundle["attempt"],
            "collector_manifest": bundle["manifest"],
            "before_prepare": bundle["before_prepare"],
            "after_prepare": baseline,
            "after_executor": post,
            "preparation_comparison": bundle["preparation_comparison"],
            "executor_comparison": bundle["executor_comparison"],
        }
        _write_json(
            self.trial_paths.verifier_dir / "global-observations.json", global_report
        )
        # Retain the old convenience path for consumers of recorded jobs. This is
        # a copy of collector data; the verifier no longer contacts live nodes.
        _write_json(self.trial_paths.verifier_dir / "environment-post.json", post)

        executor_report = _load_json(
            self.trial_paths.agent_dir / "agent-output.json",
            {
                "status": "blocked",
                "summary": "The executor final report was unavailable.",
                "actions_performed": [],
                "evidence": [],
            },
        )
        if not isinstance(executor_report, dict):
            executor_report = {
                "status": "blocked",
                "summary": "The executor final report was malformed.",
                "actions_performed": [],
                "evidence": [],
            }
        commands, command_stats = _terminal_trace(
            self.trial_paths.agent_dir / "executor-commands.jsonl"
        )
        command_ids = {
            str(record["command_id"])
            for record in commands
            if isinstance(record.get("command_id"), str)
        }
        selected, selection_limitations = _selected_evidence(
            executor_report, command_ids
        )
        evidence_record = {
            "schema_version": 1,
            "command_stats": command_stats,
            "selected_evidence": selected,
            "limitations": selection_limitations,
            "commands": commands,
        }
        _write_json(
            self.trial_paths.verifier_dir / "executor-evidence.json", evidence_record
        )

        task_baseline = _load_json(
            self.trial_paths.trial_dir / "prepare" / "baseline-report.json", {}
        )
        workspace = self.trial_paths.verifier_dir / "outcome-evaluator"
        raw_report, raw_output = await run_structured_log_agent(
            agent_name=self.agent,
            model=self.model,
            prompt=self._evaluation_prompt(
                executor_report=executor_report,
                evidence_record=evidence_record,
                global_report=global_report,
                task_baseline=task_baseline,
            ),
            schema=EvidenceEvaluation.model_json_schema(),
            workspace=workspace,
            timeout_sec=self.evaluator_timeout_sec,
            reasoning_effort=self.reasoning_effort,
            service_tier=self.service_tier,
            telemetry_path=self.trial_paths.verifier_dir / "llm-metrics.jsonl",
            telemetry_context={"role": "outcome-verifier"},
        )
        evaluation = EvidenceEvaluation.model_validate(raw_report)
        valid_evidence_ids = command_ids | _global_evidence_ids(global_report)
        validation_limitations = list(selection_limitations)
        normalized_requirements: list[RequirementAssessment] = []
        seen_requirement_ids: set[str] = set()
        for requirement in evaluation.requirements:
            if requirement.id in seen_requirement_ids:
                raise ValueError(
                    f"Outcome verifier returned duplicate requirement ID {requirement.id!r}"
                )
            seen_requirement_ids.add(requirement.id)
            unknown = sorted(set(requirement.evidence_ids) - valid_evidence_ids)
            known = [
                evidence_id
                for evidence_id in requirement.evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            if unknown:
                validation_limitations.append(
                    f"Requirement {requirement.id} cited unknown evidence IDs: "
                    + ", ".join(unknown)
                )
            if requirement.status != "indeterminate" and not known:
                requirement = requirement.model_copy(
                    update={
                        "status": "indeterminate",
                        "summary": (
                            requirement.summary
                            + " No valid captured evidence ID supported this conclusion."
                        ),
                        "evidence_ids": [],
                    }
                )
            else:
                requirement = requirement.model_copy(update={"evidence_ids": known})
            normalized_requirements.append(requirement)

        total_requirements = len(normalized_requirements)
        determinate_requirements = sum(
            item.status != "indeterminate" for item in normalized_requirements
        )
        coverage = (
            round(determinate_requirements / total_requirements, 4)
            if total_requirements
            else 0.0
        )
        confidence = round(min(evaluation.confidence, coverage), 4)
        evaluation_complete = coverage > self.minimum_coverage or math.isclose(
            coverage, self.minimum_coverage
        )
        reward = _score_requirements(normalized_requirements)
        hygiene_ids = [
            evidence_id
            for evidence_id in evaluation.operational_hygiene.evidence_ids
            if evidence_id in valid_evidence_ids
        ]
        unknown_hygiene_ids = sorted(
            set(evaluation.operational_hygiene.evidence_ids) - valid_evidence_ids
        )
        if unknown_hygiene_ids:
            validation_limitations.append(
                "Operational hygiene cited unknown evidence IDs: "
                + ", ".join(unknown_hygiene_ids)
            )

        report = {
            "schema_version": 1,
            "evaluator": "evidence",
            "requirements": [item.model_dump() for item in normalized_requirements],
            "overall_summary": evaluation.overall_summary,
            "limitations": [*evaluation.limitations, *validation_limitations],
            "command_stats": command_stats,
            "evaluation_coverage": coverage,
            "confidence": confidence,
            "evaluation_complete": evaluation_complete,
            "publication_eligible": evaluation_complete,
            "functionality": reward,
            "operational_hygiene": {
                **evaluation.operational_hygiene.model_dump(),
                "evidence_ids": hygiene_ids,
            },
            "reward": reward if evaluation_complete else None,
            "provisional_reward": reward,
        }
        _write_json(self.trial_paths.verifier_dir / "evaluation-report.json", report)
        _write_json(self.trial_paths.verifier_dir / "reward-details.json", report)
        (self.trial_paths.verifier_dir / "outcome-evaluator-raw.txt").write_text(
            redact_text(raw_output)
        )
        if not evaluation_complete:
            _write_json(
                self.trial_paths.verifier_dir / "verifier-review.json",
                {
                    "disposition": "discard_task_result_pending_review",
                    "reason": "Material task outcomes lack determinate captured evidence.",
                    "coverage": coverage,
                    "requirements": [
                        item.model_dump()
                        for item in normalized_requirements
                        if item.status == "indeterminate"
                    ],
                },
            )

        rewards: dict[str, float] = {
            "evaluation_coverage": coverage,
            "confidence": confidence,
            "evaluation_complete": float(evaluation_complete),
            "publication_eligible": float(evaluation_complete),
            "functionality": reward,
            "operational_hygiene": round(
                evaluation.operational_hygiene.score, 4
            ),
        }
        if evaluation_complete:
            rewards = {"reward": reward, **rewards}
        _write_json(self.trial_paths.reward_json_path, rewards)
        return VerifierResult(rewards=rewards)


AntriebVerifier = EvidenceVerifier

__all__ = ["AntriebVerifier", "EvidenceVerifier"]
