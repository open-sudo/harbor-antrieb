import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.environments.base import ExecResult
from harbor.models.trial.paths import TrialPaths
from infraset.ai_preparer import run_ai_prepare
from infraset.config import PrepareConfig
from infraset.runbooks import BaseRunbook


@pytest.mark.asyncio
async def test_ai_preparer_runs_agent_then_static_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = tmp_path / "task"
    environment_dir = task_dir / "environment"
    prepare_config_dir = task_dir / "prepare"
    environment_dir.mkdir(parents=True)
    prepare_config_dir.mkdir()
    (task_dir / "instruction.md").write_text("Repair the legacy application.")
    (prepare_config_dir / "prompt.md").write_text(
        "Create a legacy application data file on node1."
    )
    (prepare_config_dir / "baseline.toml").write_text(
        "[[observations]]\n"
        'id = "legacy-data"\n'
        'node = "node1"\n'
        'command = "sha256sum /srv/legacy/data"\n'
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    runner_kwargs: dict[str, Any] = {}

    async def fake_runner(**kwargs: Any) -> tuple[dict[str, Any], str]:
        runner_kwargs.update(kwargs)
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"seed","outcome":"completed"}\n'
        )
        return (
            {
                "status": "completed",
                "summary": "legacy state created",
                "actions_performed": ["created data"],
            },
            "raw-output",
        )

    monkeypatch.setattr("infraset.ai_preparer.run_structured_agent", fake_runner)

    class Environment:
        nodes = ("node1", "node2")
        remote_session_id = "managed-session"
        endpoint = "https://antrieb.sh/mcp"
        base_runbooks = (
            BaseRunbook("antrieb/primer", "Antrieb commands use POSIX sh."),
        )

        def __init__(self) -> None:
            self.environment_dir = environment_dir
            self.trial_paths = trial_paths

        async def exec_on_node(self, node: str, command: str) -> ExecResult:
            assert (node, command) == ("node1", "sha256sum /srv/legacy/data")
            return ExecResult(stdout="baseline-hash\n", stderr="", return_code=0)

    config = PrepareConfig(
        enabled=True,
        mode="ai",
        agent="codex",
        model="test-model",
    )

    await run_ai_prepare(Environment(), config)

    assert runner_kwargs["agent_name"] == "codex"
    assert runner_kwargs["model"] == "test-model"
    assert runner_kwargs["nodes"] == ("node1", "node2")
    assert "Create a legacy application" in runner_kwargs["prompt"]
    assert "Antrieb commands use POSIX sh." in runner_kwargs["prompt"]
    output_dir = trial_paths.trial_dir / "prepare"
    assert (
        json.loads((output_dir / "ai-preparer-output.json").read_text())["status"]
        == "completed"
    )
    baseline = json.loads((output_dir / "baseline-report.json").read_text())
    assert baseline["observations"][0]["stdout"] == "baseline-hash\n"


@pytest.mark.asyncio
async def test_ai_preparer_requires_explicit_agent(tmp_path: Path) -> None:
    environment = SimpleNamespace()

    with pytest.raises(ValueError, match="requires prepare.agent"):
        await run_ai_prepare(
            environment,
            PrepareConfig(enabled=True, mode="ai"),
        )
