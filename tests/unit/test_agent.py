import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from harbor.models.agent.context import AgentContext
from harbor.environments.base import BaseEnvironment
from harbor_antrieb.agent import AntriebHostAgent, _postmortem_audit_evidence
from harbor_antrieb.errors import ClusterExpiredError
from harbor_antrieb.runbooks import BaseRunbook


class FakeRetryEnvironment:
    def __init__(self, max_clusters: int) -> None:
        self.remote_session_id = "managed-session-1"
        self.nodes = ("node1", "node2", "node3")
        self.endpoint = "https://antrieb.sh/mcp"
        self.cluster_expires_at = None
        self.clusters_provisioned = 1
        self.definition = SimpleNamespace(max_clusters=max_clusters)
        self.recreate_calls = 0

    def assert_cluster_active(self) -> None:
        pass

    async def recreate(self) -> None:
        self.recreate_calls += 1
        self.clusters_provisioned += 1
        self.remote_session_id = f"managed-session-{self.clusters_provisioned}"


def test_host_agent_manages_logs_in_the_host_trial_directory() -> None:
    assert AntriebHostAgent.capabilities.host_managed_logs is True


def test_postmortem_audit_is_compact_and_prioritizes_failures(tmp_path: Path) -> None:
    audit_path = tmp_path / "executor-commands.jsonl"
    records: list[dict[str, Any]] = []
    for index in range(30):
        records.extend(
            [
                {
                    "node": "node1",
                    "command": f"command-{index}",
                    "outcome": "requested",
                },
                {
                    "node": "node1",
                    "command": f"command-{index}",
                    "outcome": "completed",
                    "return_code": 0,
                    "stdout": "x" * 10_000,
                },
            ]
        )
    records.append(
        {
            "node": "node2",
            "command": "pcs property set stonith-watchdog-timeout=20s",
            "outcome": "completed",
            "return_code": 1,
            "stderr": "SBD not active",
        }
    )
    records.extend(
        {
            "timestamp": f"2026-08-26T20:00:0{index}Z",
            "node": "node3",
            "command": "date",
            "outcome": "cluster_expired",
            "error": "lease expired",
            "duration_ms": index,
        }
        for index in range(5)
    )
    audit_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    evidence = _postmortem_audit_evidence(audit_path)

    assert len(evidence.encode()) <= 20 * 1024
    assert "stonith-watchdog-timeout" in evidence
    assert "SBD not active" in evidence
    assert '"requested": 30' in evidence
    assert '"outcome":"requested"' not in evidence
    assert evidence.count('"outcome":"cluster_expired"') == 1


@pytest.mark.asyncio
async def test_postmortem_failure_does_not_prevent_fresh_cluster_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def fake_executor(**kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal calls
        calls += 1
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"date","outcome":"cluster_expired"}\n'
        )
        if calls == 1:
            raise ClusterExpiredError("managed cluster lease expired")
        return (
            {
                "status": "completed",
                "summary": "configured on the fresh cluster",
                "actions_performed": ["completed configuration"],
            },
            "completed output",
        )

    async def failed_diagnostic(**kwargs: Any) -> tuple[dict[str, Any], str]:
        raise OSError(7, "Argument list too long")

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_executor)
    monkeypatch.setattr(
        "harbor_antrieb.agent.run_structured_log_agent", failed_diagnostic
    )
    environment = FakeRetryEnvironment(max_clusters=2)
    agent = AntriebHostAgent(logs_dir=tmp_path, agent_name="codex")

    await agent.run(
        "Configure a database cluster",
        cast(BaseEnvironment, environment),
        AgentContext(),
    )

    assert environment.recreate_calls == 1
    history = json.loads((tmp_path / "attempt-history.json").read_text())
    assert history["attempts"][0]["outcome"] == "cluster_expired"
    assert "Argument list too long" in history["attempts"][0]["diagnostic_error"]


@pytest.mark.parametrize(
    ("model_name", "expected_provider"),
    [
        ("gpt-5.6-sol", "openai"),
        ("claude-sonnet-4-6", "anthropic"),
        ("kimi/kimi-k2.5", "kimi"),
        ("onprem/local-model", "onprem"),
    ],
)
def test_host_agent_reports_provider_from_model(
    tmp_path: Path,
    model_name: str,
    expected_provider: str,
) -> None:
    agent = AntriebHostAgent(logs_dir=tmp_path, model_name=model_name)

    info = agent.to_agent_info()

    assert info.model_info is not None
    assert info.model_info.provider == expected_provider


@pytest.mark.asyncio
async def test_host_agent_uses_managed_session_without_environment_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_kwargs: dict[str, Any] = {}

    async def fake_runner(**kwargs: Any) -> tuple[dict[str, Any], str]:
        runner_kwargs.update(kwargs)
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"true","outcome":"completed"}\n'
        )
        return (
            {
                "status": "completed",
                "summary": "configured",
                "actions_performed": ["updated all nodes"],
            },
            "raw-agent-output",
        )

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_runner)
    environment = cast(
        BaseEnvironment,
        SimpleNamespace(
            remote_session_id="managed-session",
            nodes=("node1", "node2", "node3"),
            endpoint="https://antrieb.sh/mcp",
            base_runbooks=(
                BaseRunbook("antrieb/primer", "Antrieb commands use POSIX sh."),
            ),
        ),
    )
    agent = AntriebHostAgent(
        logs_dir=tmp_path,
        model_name="claude-sonnet-4-6",
        reasoning_effort="medium",
        service_tier=None,
    )
    context = AgentContext()

    await agent.setup(environment)
    await agent.run("Configure the cluster", environment, context)

    assert runner_kwargs["session_id"] == "managed-session"
    assert runner_kwargs["model"] == "claude-sonnet-4-6"
    assert runner_kwargs["reasoning_effort"] == "medium"
    assert runner_kwargs["service_tier"] is None
    assert "node1, node2, node3" in runner_kwargs["prompt"]
    assert "Do not install the agent" in runner_kwargs["prompt"]
    assert "Antrieb commands use POSIX sh." in runner_kwargs["prompt"]
    assert (tmp_path / "agent-raw-output.txt").read_text() == "raw-agent-output"
    assert context.metadata is not None
    assert context.metadata["infraset_executor"]["status"] == "completed"
    history = json.loads((tmp_path / "attempt-history.json").read_text())
    assert history["clusters_provisioned"] == 1
    assert history["attempts"][0]["outcome"] == "completed"


@pytest.mark.asyncio
async def test_host_agent_forwards_requested_service_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_kwargs: dict[str, Any] = {}

    async def fake_runner(**kwargs: Any) -> tuple[dict[str, Any], str]:
        runner_kwargs.update(kwargs)
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"true","outcome":"completed"}\n'
        )
        return (
            {
                "status": "completed",
                "summary": "configured",
                "actions_performed": ["updated node1"],
            },
            "raw-agent-output",
        )

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_runner)
    environment = cast(
        BaseEnvironment,
        SimpleNamespace(
            remote_session_id="managed-session",
            nodes=("node1",),
            endpoint="https://antrieb.sh/mcp",
            base_runbooks=(),
        ),
    )
    agent = AntriebHostAgent(
        logs_dir=tmp_path,
        model_name="gpt-5.6-sol",
        agent_name="codex",
        service_tier="fast",
    )

    await agent.run("Configure node1", environment, AgentContext())

    assert runner_kwargs["service_tier"] == "fast"


@pytest.mark.asyncio
async def test_blocked_attempt_is_diagnosed_and_retried_on_fresh_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor_prompts: list[str] = []
    diagnostic_prompts: list[str] = []

    async def fake_executor(**kwargs: Any) -> tuple[dict[str, Any], str]:
        executor_prompts.append(kwargs["prompt"])
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"false","outcome":"completed",'
            '"return_code":1,"stderr":"failed"}\n'
        )
        if len(executor_prompts) == 1:
            return (
                {
                    "status": "blocked",
                    "summary": "package installation failed",
                    "actions_performed": ["tried the default repository"],
                },
                "blocked output",
            )
        return (
            {
                "status": "completed",
                "summary": "configured after changing approach",
                "actions_performed": ["used a compatible repository"],
            },
            "completed output",
        )

    async def fake_diagnostic(**kwargs: Any) -> tuple[dict[str, Any], str]:
        diagnostic_prompts.append(kwargs["prompt"])
        return (
            {
                "summary": "the default repository did not support this image",
                "evidence": [
                    {"source": "command audit", "finding": "install exited 1"}
                ],
                "likely_causes": [
                    {"cause": "repository incompatibility", "confidence": "high"}
                ],
                "next_attempt_guidance": ["select an image-compatible repository"],
            },
            "diagnostic output",
        )

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_executor)
    monkeypatch.setattr(
        "harbor_antrieb.agent.run_structured_log_agent", fake_diagnostic
    )
    environment = FakeRetryEnvironment(max_clusters=2)
    agent = AntriebHostAgent(
        logs_dir=tmp_path,
        model_name="gpt-5.6-sol",
        agent_name="codex",
    )
    context = AgentContext()

    await agent.run(
        "Configure a database cluster",
        cast(BaseEnvironment, environment),
        context,
    )

    assert environment.recreate_calls == 1
    assert "managed-session-1" not in executor_prompts[1]
    assert "This is a fresh cluster" in executor_prompts[1]
    assert "repository incompatibility" in executor_prompts[1]
    assert "return_code" in diagnostic_prompts[0]
    history = json.loads((tmp_path / "attempt-history.json").read_text())
    assert history["clusters_provisioned"] == 2
    assert [item["outcome"] for item in history["attempts"]] == [
        "executor_blocked",
        "completed",
    ]
    assert (tmp_path / "attempts/01/postmortem/diagnosis.json").is_file()
    assert (tmp_path / "attempts/02/executor-output.json").is_file()


@pytest.mark.asyncio
async def test_expired_cluster_is_diagnosed_from_logs_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def fake_executor(**kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal calls
        calls += 1
        kwargs["audit_path"].write_text(
            '{"node":"node2","command":"apt-get install","outcome":"cluster_expired"}\n'
        )
        if calls == 1:
            raise ClusterExpiredError("managed cluster lease expired")
        return (
            {
                "status": "completed",
                "summary": "configured",
                "actions_performed": ["completed on the fresh cluster"],
            },
            "completed output",
        )

    async def fake_diagnostic(**kwargs: Any) -> tuple[dict[str, Any], str]:
        assert "cluster_expired" in kwargs["prompt"]
        assert "apt-get install" in kwargs["prompt"]
        return (
            {
                "summary": "installation consumed the lease",
                "evidence": [{"source": "command audit", "finding": "lease expired"}],
                "likely_causes": [
                    {"cause": "slow package installation", "confidence": "medium"}
                ],
                "next_attempt_guidance": ["check package availability first"],
            },
            "diagnostic output",
        )

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_executor)
    monkeypatch.setattr(
        "harbor_antrieb.agent.run_structured_log_agent", fake_diagnostic
    )
    environment = FakeRetryEnvironment(max_clusters=2)
    agent = AntriebHostAgent(logs_dir=tmp_path, agent_name="codex")

    await agent.run(
        "Configure a database cluster",
        cast(BaseEnvironment, environment),
        AgentContext(),
    )

    history = json.loads((tmp_path / "attempt-history.json").read_text())
    assert history["attempts"][0]["outcome"] == "cluster_expired"
    assert environment.recreate_calls == 1


@pytest.mark.asyncio
async def test_cluster_quota_exhaustion_keeps_the_final_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_executor(**kwargs: Any) -> tuple[dict[str, Any], str]:
        kwargs["audit_path"].write_text(
            '{"node":"node1","command":"false","outcome":"completed"}\n'
        )
        return (
            {
                "status": "blocked",
                "summary": "could not complete",
                "actions_performed": [],
            },
            "blocked output",
        )

    async def fake_diagnostic(**kwargs: Any) -> tuple[dict[str, Any], str]:
        return (
            {
                "summary": "insufficient evidence",
                "evidence": [],
                "likely_causes": [],
                "next_attempt_guidance": ["collect narrower diagnostics"],
            },
            "diagnostic output",
        )

    monkeypatch.setattr("harbor_antrieb.agent.run_structured_agent", fake_executor)
    monkeypatch.setattr(
        "harbor_antrieb.agent.run_structured_log_agent", fake_diagnostic
    )
    environment = FakeRetryEnvironment(max_clusters=1)
    agent = AntriebHostAgent(logs_dir=tmp_path, agent_name="codex")

    with pytest.raises(RuntimeError, match="exhausted its managed-cluster quota"):
        await agent.run(
            "Configure a database cluster",
            cast(BaseEnvironment, environment),
            AgentContext(),
        )

    assert environment.recreate_calls == 0
    history = json.loads((tmp_path / "attempt-history.json").read_text())
    assert history["attempts"][0]["diagnosis"]["summary"] == "insufficient evidence"
