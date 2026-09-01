import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from harbor.models.trial.paths import TrialPaths

from harbor_antrieb.evidence_verifier import AntriebVerifier
from harbor_antrieb.observations import (
    GLOBAL_OBSERVATIONS,
    collect_global_observations,
    compare_global_observations,
)


class FakeEnvironment:
    def __init__(self) -> None:
        self.nodes = ("node1",)
        self.remote_session_id = "managed-session"
        self.node_images = {"node1": {"ani": "antrieb:ubuntu24.04:v1"}}
        self.definition = SimpleNamespace(
            model_dump=lambda **_: {
                "cluster": ["ubuntu24.04"],
                "control_node": "node1",
            }
        )
        self.commands: list[tuple[str, str, str | int | None]] = []

    def assert_cluster_active(self) -> None:
        raise AssertionError("offline verifier must not require a live cluster")

    async def exec_on_node(
        self, node: str, command: str, *, user: str | int | None = None
    ) -> SimpleNamespace:
        self.commands.append((node, command, user))
        if "find /tmp" in command:
            stdout = "/tmp/executor-note\n"
        elif "systemctl --failed" in command:
            stdout = ""
        elif "ss -lntupH" in command:
            stdout = "tcp LISTEN 0 128 0.0.0.0:443 0.0.0.0:*\n"
        elif "dpkg --audit" in command:
            stdout = "manager=dpkg\n"
        else:
            stdout = "node=node1\naddress=10.0.0.10\nUbuntu\n"
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


def _write_executor_artifacts(trial_paths: TrialPaths) -> None:
    trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
    (trial_paths.agent_dir / "agent-output.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "summary": "Configured the service.",
                "actions_performed": ["Configured node1"],
                "evidence": [
                    {
                        "requirement": "Serve HTTPS",
                        "command_ids": ["cmd-1"],
                        "summary": "curl returned the expected response.",
                    }
                ],
            }
        )
    )
    records = [
        {
            "command_id": "cmd-1",
            "node": "node1",
            "command": "curl -fsS https://$NODE_IP/",
            "outcome": "requested",
        },
        {
            "command_id": "cmd-1",
            "node": "node1",
            "command": "curl -fsS https://$NODE_IP/",
            "outcome": "completed",
            "return_code": 0,
            "stdout": "ready",
            "stderr": "",
            "duration_ms": 12,
        },
        {
            "command_id": "cmd-2",
            "node": "node1",
            "command": "systemctl is-active missing.service",
            "outcome": "completed",
            "return_code": 3,
            "stdout": "inactive",
            "stderr": "",
            "duration_ms": 8,
        },
    ]
    (trial_paths.agent_dir / "executor-commands.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def _write_collector_artifacts(trial_paths: TrialPaths) -> None:
    attempt_dir = trial_paths.trial_dir / "collector" / "attempts" / "01"
    snapshots = attempt_dir / "snapshots"
    snapshots.mkdir(parents=True)

    def snapshot(phase: str, stdout: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": phase,
            "captured_at": "2026-08-31T00:00:00Z",
            "nodes": [
                {
                    "name": "node1",
                    "observations": [
                        {
                            "id": f"global:{phase}:node1:temporary-files",
                            "description": "Bounded inventory of entries below /tmp.",
                            "status": "observed",
                            "return_code": 0,
                            "stdout": stdout,
                            "stderr": "",
                        }
                    ],
                }
            ],
            "limitations": [],
        }

    values = {
        "before-prepare.json": snapshot("before_prepare", ""),
        "after-prepare.json": snapshot("after_prepare", ""),
        "after-executor.json": snapshot(
            "after_executor", "/tmp/executor-note\n"
        ),
    }
    for name, value in values.items():
        (snapshots / name).write_text(json.dumps(value))
    (attempt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": 1,
                "cluster_number": 1,
                "prepare_enabled": True,
                "phases": {
                    phase: {
                        "status": "captured",
                        "path": (
                            f"collector/attempts/01/snapshots/"
                            f"{phase.replace('_', '-')}.json"
                        ),
                    }
                    for phase in (
                        "before_prepare",
                        "after_prepare",
                        "after_executor",
                    )
                },
            }
        )
    )


@pytest.mark.asyncio
async def test_evidence_verifier_scores_without_task_local_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    _write_executor_artifacts(trial_paths)
    _write_collector_artifacts(trial_paths)
    environment = FakeEnvironment()
    captured_prompt = ""

    async def fake_evaluator(**kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal captured_prompt
        captured_prompt = kwargs["prompt"]
        return {
            "requirements": [
                {
                    "id": "https-service",
                    "requirement": "Serve the requested HTTPS response.",
                    "status": "satisfied",
                    "summary": "The final curl succeeded.",
                    "evidence_ids": ["cmd-1"],
                }
            ],
            "confidence": 0.9,
            "operational_hygiene": {
                "score": 0.75,
                "summary": "One new temporary path warrants review.",
                "evidence_ids": [
                    "global:after_executor:node1:temporary-files"
                ],
            },
            "overall_summary": "The material task outcome was observed.",
            "limitations": [],
        }, "raw evaluator output"

    monkeypatch.setattr(
        "harbor_antrieb.evidence_verifier.run_structured_log_agent", fake_evaluator
    )
    verifier = AntriebVerifier(
        task=SimpleNamespace(instruction="Configure HTTPS on node1."),
        trial_paths=trial_paths,
        environment=environment,
        agent="codex",
        model="test-model",
    )

    result = await verifier.verify()

    assert result.rewards == {
        "reward": 1.0,
        "evaluation_coverage": 1.0,
        "confidence": 0.9,
        "evaluation_complete": 1.0,
        "publication_eligible": 1.0,
        "functionality": 1.0,
        "operational_hygiene": 0.75,
    }
    assert environment.commands == []
    assert "cmd-1" in captured_prompt
    assert "Complete executor command timeline" in captured_prompt
    assert "configuration is supporting evidence only" in captured_prompt
    evidence = json.loads(
        (trial_paths.verifier_dir / "executor-evidence.json").read_text()
    )
    assert evidence["command_stats"] == {
        "total": 2,
        "successful": 1,
        "failed": 1,
        "indeterminate": 0,
        "malformed_audit_records": 0,
    }
    assert not (trial_paths.verifier_dir / "verifier-review.json").exists()


@pytest.mark.asyncio
async def test_unknown_evidence_makes_requirement_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    _write_executor_artifacts(trial_paths)
    _write_collector_artifacts(trial_paths)

    async def fake_evaluator(**_: Any) -> tuple[dict[str, Any], str]:
        return {
            "requirements": [
                {
                    "id": "service",
                    "requirement": "Run the service.",
                    "status": "satisfied",
                    "summary": "Claimed as working.",
                    "evidence_ids": ["invented-command"],
                }
            ],
            "confidence": 1.0,
            "operational_hygiene": {
                "score": 1.0,
                "summary": "No issue found.",
                "evidence_ids": [
                    "global:after_executor:node1:temporary-files"
                ],
            },
            "overall_summary": "Evidence was claimed.",
            "limitations": [],
        }, "raw"

    monkeypatch.setattr(
        "harbor_antrieb.evidence_verifier.run_structured_log_agent", fake_evaluator
    )
    verifier = AntriebVerifier(
        task=SimpleNamespace(instruction="Run a service."),
        trial_paths=trial_paths,
        environment=FakeEnvironment(),
    )

    result = await verifier.verify()

    assert "reward" not in result.rewards
    assert result.rewards["evaluation_coverage"] == 0.0
    assert result.rewards["confidence"] == 0.0
    report = json.loads(
        (trial_paths.verifier_dir / "evaluation-report.json").read_text()
    )
    assert report["requirements"][0]["status"] == "indeterminate"
    assert (trial_paths.verifier_dir / "verifier-review.json").is_file()


@pytest.mark.asyncio
async def test_global_collector_records_unavailable_data_without_failing(
    tmp_path: Path,
) -> None:
    class BrokenEnvironment:
        nodes = ("node1",)
        node_images: ClassVar[dict[str, Any]] = {}

        async def exec_on_node(self, *_: Any, **__: Any) -> None:
            raise TimeoutError("provider did not answer")

    report = await collect_global_observations(
        BrokenEnvironment(),
        phase="post",
        output_path=tmp_path / "post.json",
        audit_path=tmp_path / "audit.jsonl",
    )

    observations = report["nodes"][0]["observations"]
    assert len(observations) == len(GLOBAL_OBSERVATIONS)
    assert {item["status"] for item in observations} == {"unavailable"}
    assert "provider did not answer" in observations[0]["error"]


def test_global_comparison_is_baseline_relative() -> None:
    baseline = {
        "captured_at": "before",
        "nodes": [
            {
                "name": "node1",
                "observations": [
                    {
                        "id": "global:baseline:node1:temporary-files",
                        "status": "observed",
                        "return_code": 0,
                        "stdout": "/tmp/preexisting\n",
                    }
                ],
            }
        ],
        "limitations": [],
    }
    post = {
        "captured_at": "after",
        "nodes": [
            {
                "name": "node1",
                "observations": [
                    {
                        "id": "global:post:node1:temporary-files",
                        "status": "observed",
                        "return_code": 0,
                        "stdout": "/tmp/preexisting\n/tmp/new\n",
                    }
                ],
            }
        ],
        "limitations": [],
    }

    comparison = compare_global_observations(baseline, post)

    assert comparison["changes"] == [
        {
            "node": "node1",
            "observation": "temporary-files",
            "baseline_status": "observed",
            "post_status": "observed",
            "baseline_return_code": 0,
            "post_return_code": 0,
            "added_lines": ["/tmp/new"],
            "removed_lines": [],
        }
    ]
