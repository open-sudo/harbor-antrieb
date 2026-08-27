import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.models.trial.paths import TrialPaths
from harbor_antrieb.runbooks import BaseRunbook
from harbor_antrieb.verifier import AntriebVerifier


def _write_task(task_dir: Path) -> None:
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "judge.toml").write_text(
        "timeout_sec = 60\n"
        "parallel_node_collection = false\n"
        "[dimensions.functionality]\n"
        "weight = 0.6\n"
        'description = "works"\n'
        "[dimensions.security]\n"
        "weight = 0.4\n"
        'description = "safe"\n'
    )
    (verifier_dir / "checks.toml").write_text(
        "schema_version = 1\n"
        "[[probes]]\n"
        'id = "service-probe"\n'
        "level = 1\n"
        'targets = ["node1"]\n'
        "max_exec_calls = 2\n"
        'procedure = "Probe the service and its access controls."\n'
        "[[probes]]\n"
        'id = "access-probe"\n'
        "level = 2\n"
        'targets = ["node2"]\n'
        "max_exec_calls = 3\n"
        'procedure = "Test a second access path."\n'
        "[[probes]]\n"
        'id = "reboot-probe"\n'
        "level = 8\n"
        'targets = ["node1"]\n'
        'effect = "reboot"\n'
        'procedure = "Reboot and probe."\n'
        'cleanup = "Wait for the node."\n'
        "[[assertions]]\n"
        'id = "service"\n'
        'probe = "service-probe"\n'
        'dimension = "functionality"\n'
        "points = 2\n"
        'pass_condition = "It works."\n'
        'fail_condition = "It is confirmed broken."\n'
        "[[assertions]]\n"
        'id = "primary-access"\n'
        'probe = "service-probe"\n'
        'dimension = "security"\n'
        'pass_condition = "Unauthorized access is denied."\n'
        'fail_condition = "Unauthorized access is confirmed allowed."\n'
        "[[assertions]]\n"
        'id = "secondary-access"\n'
        'probe = "access-probe"\n'
        'dimension = "security"\n'
        'requires = ["service"]\n'
        'pass_condition = "The second access path is denied."\n'
        'fail_condition = "The second access path is confirmed allowed."\n'
        "[[assertions]]\n"
        'id = "reboot"\n'
        'probe = "reboot-probe"\n'
        'dimension = "functionality"\n'
        'requires = ["service"]\n'
        'pass_condition = "It returns."\n'
        'fail_condition = "It does not return."\n'
    )


def _evidence(node: str, command: str = "probe") -> list[dict[str, str]]:
    return [{"node": node, "command": command, "observation": "observed"}]


@pytest.mark.asyncio
async def test_verifier_excludes_disabled_security_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_authored_exec(node: str, command: str) -> None:
        raise AssertionError(f"unexpected authored command on {node}: {command}")

    task_dir = tmp_path / "task"
    _write_task(task_dir)
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    task = SimpleNamespace(
        paths=SimpleNamespace(task_dir=task_dir),
        instruction="Build it",
    )
    environment = SimpleNamespace(
        remote_session_id="session-1",
        nodes=("node1", "node2"),
        endpoint="https://antrieb.sh/mcp",
        base_runbooks=(BaseRunbook("antrieb/primer", "Use managed exec."),),
        exec_on_node=unexpected_authored_exec,
    )
    runner_kwargs: dict[str, Any] = {}

    async def fake_runner(**kwargs: Any) -> tuple[dict[str, Any], str]:
        runner_kwargs.update(kwargs)
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"probe","outcome":"completed"}\n'
        )
        report = {
            "probes": [
                {
                    "id": "service-probe",
                    "summary": "collected",
                    "cleanup_completed": True,
                    "limitations": [],
                },
            ],
            "assertions": [
                {
                    "id": "service",
                    "status": "pass",
                    "summary": "working",
                    "evidence": _evidence("node1"),
                    "failures": [],
                    "limitations": [],
                },
            ],
            "actions_performed": [],
            "overall_summary": "functionality evaluated",
        }
        kwargs["output_validator"](report)
        return report, "raw"

    monkeypatch.setattr("harbor_antrieb.verifier.run_structured_agent", fake_runner)
    verifier = AntriebVerifier(
        task=task,
        trial_paths=trial_paths,
        environment=environment,
        agent="codex",
        model="test-model",
        reasoning_effort="low",
        service_tier="fast",
        level=2,
    )

    result = await verifier.verify()

    assert result.rewards == {
        "reward": 1.0,
        "evaluation_coverage": 1.0,
        "evaluation_complete": 1.0,
        "publication_eligible": 1.0,
        "functionality": 1.0,
    }
    assert runner_kwargs["max_exec_calls"] == 2
    assert runner_kwargs["nodes"] == ("node1", "node2")
    assert runner_kwargs["agent_name"] == "codex"
    assert runner_kwargs["model"] == "test-model"
    assert runner_kwargs["reasoning_effort"] == "low"
    assert runner_kwargs["service_tier"] == "fast"
    assert (
        "Missing evidence and evaluator command errors are indeterminate"
        in (runner_kwargs["prompt"])
    )
    assert "use privilege-aware discovery" in runner_kwargs["prompt"]
    assert "Never print credentials" in runner_kwargs["prompt"]
    assert (
        "Never use SSH, SCP, or another nested remote shell" in runner_kwargs["prompt"]
    )
    assert "canonical service name established" in runner_kwargs["prompt"]
    assert "validate the intended service identity" in runner_kwargs["prompt"]
    assert "Probe 1: service-probe" in runner_kwargs["prompt"]
    assert "Assertion primary-access" not in runner_kwargs["prompt"]
    assert "Probe 2: access-probe" not in runner_kwargs["prompt"]
    assert "Probe 3: reboot-probe" not in runner_kwargs["prompt"]
    report = json.loads(
        (trial_paths.verifier_dir / "evaluation-report.json").read_text()
    )
    assert report["evaluation_coverage"] == 1.0
    assert report["evaluation_complete"] is True
    assert report["publication_eligible"] is True
    assert report["aggregation"]["reward"] == 1.0
    assert report["dimensions"]["security"]["applicable"] is False
    assert report["dimensions"]["security"]["assertions"] == []
    selected = json.loads(
        (trial_paths.verifier_dir / "semantic-plan-selection.json").read_text()
    )
    assert selected["disabled_dimensions"] == ["security"]


def test_verifier_scores_assertions_from_one_probe_independently(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, rubric = verifier._load()
    assertions = [item for item in plan.assertions if item.probe == "service-probe"]
    report = {
        "assertions": [
            {
                "id": "service",
                "status": "pass",
                "summary": "working",
                "evidence": _evidence("node1"),
                "failures": [],
                "limitations": [],
            },
            {
                "id": "primary-access",
                "status": "fail",
                "summary": "exposed",
                "evidence": _evidence("node1"),
                "failures": ["access succeeded"],
                "limitations": [],
            },
        ]
    }

    report, rewards = verifier._aggregate(assertions, report, rubric, 1, 1.0)

    assert rewards == {
        "reward": 0.6,
        "evaluation_coverage": 1.0,
        "evaluation_complete": 1.0,
        "publication_eligible": 1.0,
        "functionality": 1.0,
        "security": 0.0,
    }
    assert report["dimensions"]["functionality"]["score"] == 100
    assert report["dimensions"]["security"]["score"] == 0


def test_verifier_reserves_one_cleanup_call_per_mutating_probe_target(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, _ = verifier._load()
    service_probe = next(item for item in plan.probes if item.id == "service-probe")
    reboot_probe = next(item for item in plan.probes if item.id == "reboot-probe")

    assert verifier._exec_call_budget([service_probe, reboot_probe]) == (
        service_probe.max_exec_calls
        + reboot_probe.max_exec_calls
        + len(reboot_probe.targets)
    )
    rendered = verifier._probe_text(1, reboot_probe, [])
    assert "evidence command budget" in rendered
    assert "one per target" in rendered


def test_verifier_propagates_prerequisite_indeterminate(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, _ = verifier._load()
    probes = [probe for probe in plan.probes if probe.level <= 2]
    assertions = [
        assertion
        for assertion in plan.assertions
        if assertion.probe in {probe.id for probe in probes}
    ]
    report = {
        "probes": [
            {
                "id": "service-probe",
                "summary": "unavailable",
                "cleanup_completed": True,
                "limitations": ["lease ended"],
            },
            {
                "id": "access-probe",
                "summary": "collected",
                "cleanup_completed": True,
                "limitations": [],
            },
        ],
        "assertions": [
            {
                "id": "service",
                "status": "indeterminate",
                "summary": "unavailable",
                "evidence": [],
                "failures": [],
                "limitations": ["lease ended"],
            },
            {
                "id": "primary-access",
                "status": "pass",
                "summary": "denied",
                "evidence": _evidence("node1"),
                "failures": [],
                "limitations": [],
            },
            {
                "id": "secondary-access",
                "status": "fail",
                "summary": "allowed",
                "evidence": _evidence("node2"),
                "failures": ["access allowed"],
                "limitations": [],
            },
        ],
        "actions_performed": [],
        "overall_summary": "partial",
    }

    verifier._validate_report(report, probes, assertions)

    dependent = next(
        item for item in report["assertions"] if item["id"] == "secondary-access"
    )
    assert dependent["status"] == "indeterminate"
    assert dependent["failures"] == []
    assert dependent["limitations"] == ["prerequisite assertions did not pass: service"]


def test_verifier_accepts_compact_multi_node_evidence_labels(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, _ = verifier._load()
    probe = next(item for item in plan.probes if item.id == "service-probe")
    probe = probe.model_copy(update={"targets": ["node1", "node2"]})
    assertions = [item for item in plan.assertions if item.probe == "service-probe"]
    report = {
        "probes": [
            {
                "id": "service-probe",
                "summary": "collected on both nodes",
                "cleanup_completed": True,
                "limitations": [],
            }
        ],
        "assertions": [
            {
                "id": "service",
                "status": "pass",
                "summary": "both nodes responded",
                "evidence": _evidence("node1,node2"),
                "failures": [],
                "limitations": [],
            },
            {
                "id": "primary-access",
                "status": "pass",
                "summary": "node2 tested node1",
                "evidence": _evidence("node2→node1"),
                "failures": [],
                "limitations": [],
            },
        ],
        "actions_performed": [],
        "overall_summary": "complete",
    }

    verifier._validate_report(report, [probe], assertions)

    assert [item["status"] for item in report["assertions"]] == ["pass", "pass"]


def test_verifier_rejects_compact_labels_with_foreign_exec_nodes(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, _ = verifier._load()
    probe = next(item for item in plan.probes if item.id == "service-probe")
    assertions = [item for item in plan.assertions if item.probe == "service-probe"]
    report = {
        "probes": [
            {
                "id": "service-probe",
                "summary": "collected",
                "cleanup_completed": True,
                "limitations": [],
            }
        ],
        "assertions": [
            {
                "id": "service",
                "status": "pass",
                "summary": "working",
                "evidence": _evidence("node1,node2"),
                "failures": [],
                "limitations": [],
            },
            {
                "id": "primary-access",
                "status": "pass",
                "summary": "denied",
                "evidence": _evidence("node1"),
                "failures": [],
                "limitations": [],
            },
        ],
        "actions_performed": [],
        "overall_summary": "complete",
    }

    with pytest.raises(ValueError, match="cites nodes outside probe"):
        verifier._validate_report(report, [probe], assertions)


def test_verifier_turns_incomplete_mutation_cleanup_indeterminate(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, _ = verifier._load()
    probes = [probe for probe in plan.probes if probe.id == "reboot-probe"]
    assertions = [
        assertion.model_copy(update={"requires": []})
        for assertion in plan.assertions
        if assertion.id == "reboot"
    ]
    report = {
        "probes": [
            {
                "id": "reboot-probe",
                "summary": "node returned but cleanup was not confirmed",
                "cleanup_completed": False,
                "limitations": ["cleanup timed out"],
            }
        ],
        "assertions": [
            {
                "id": "reboot",
                "status": "pass",
                "summary": "node returned",
                "evidence": _evidence("node1", "reboot"),
                "failures": [],
                "limitations": [],
            }
        ],
        "actions_performed": ["rebooted node1"],
        "overall_summary": "cleanup incomplete",
    }

    verifier._validate_report(report, probes, assertions)

    assert report["assertions"][0]["status"] == "indeterminate"
    assert report["assertions"][0]["failures"] == []
    assert report["assertions"][0]["limitations"] == [
        "evaluator cleanup for probe 'reboot-probe' was not completed"
    ]


def test_verifier_can_use_an_explicit_lower_coverage_threshold(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    _write_task(task_dir)
    verifier = AntriebVerifier(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(),
        environment=SimpleNamespace(),
    )
    plan, rubric = verifier._load()
    assertions = [item for item in plan.assertions if item.probe == "service-probe"]
    report = {
        "assertions": [
            {
                "id": "service",
                "status": "pass",
                "summary": "working",
                "evidence": _evidence("node1"),
                "failures": [],
                "limitations": [],
            },
            {
                "id": "primary-access",
                "status": "indeterminate",
                "summary": "unavailable",
                "evidence": [],
                "failures": [],
                "limitations": ["tool missing"],
            },
        ]
    }

    report, rewards = verifier._aggregate(assertions, report, rubric, 1, 0.5)

    assert report["evaluation_coverage"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["evaluation_complete"] is True
    assert report["publication_eligible"] is True
    assert rewards["reward"] == 1.0
