import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from infraset.agent_runner import (
    _communicate_with_cluster_guard,
    agent_environment,
    append_audit_argument,
    append_reasoning_effort,
    append_service_tier,
    configure_claude_mcp,
    configure_codex_mcp,
    configure_log_only_agent,
    log_agent_environment,
    run_structured_agent,
)
from infraset.errors import ClusterExpiredError
from infraset.exec_bridge import ExecBridge
from rewardkit.models import MCPServerConfig


def _bridge_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "infraset.exec_bridge", *args]


def test_python_bridge_exposes_only_exec() -> None:
    messages = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ]
    )
    process = subprocess.run(
        _bridge_command("--session-id", "managed-session"),
        input=f"{messages}\n",
        text=True,
        capture_output=True,
        env={**os.environ, "ANTRIEB_TOKEN": "test-token"},
        check=True,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert [tool["name"] for tool in responses[1]["result"]["tools"]] == ["exec"]


def test_python_bridge_reads_provider_owned_token_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("test-token\n")
    message = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    env = dict(os.environ)
    env.pop("ANTRIEB_TOKEN", None)

    process = subprocess.run(
        _bridge_command(
            "--session-id",
            "managed-session",
            "--token-file",
            str(token_path),
        ),
        input=f"{message}\n",
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert json.loads(process.stdout)["result"]["serverInfo"]["name"] == "infraset"


def test_bridge_rejects_unknown_managed_node_without_remote_call() -> None:
    message = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "exec",
                "arguments": {"node": "node2", "command": "uname -a"},
            },
        }
    )
    process = subprocess.run(
        _bridge_command(
            "--session-id",
            "managed-session",
            "--node",
            "node1",
        ),
        input=f"{message}\n",
        text=True,
        capture_output=True,
        env={**os.environ, "ANTRIEB_TOKEN": "test-token"},
        check=True,
    )
    response = json.loads(process.stdout)
    assert "Unknown managed node: node2" in response["error"]["message"]


@pytest.mark.asyncio
async def test_bridge_binds_exec_to_managed_session_and_node(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def call_tool_raw(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            return {"content": [{"type": "text", "text": '{"exit_code":0}'}]}

        async def close(self) -> None:
            pass

    client = FakeClient()
    audit_path = tmp_path / "commands.jsonl"
    bridge = ExecBridge(
        client=client,
        session_id="managed-session",
        nodes=("node1",),
        audit_log=audit_path,
    )

    response = await bridge.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "exec",
                "arguments": {"node": "node1", "command": "uname -a"},
            },
        }
    )

    assert response is not None
    assert response["id"] == 7
    assert client.calls == [
        (
            "exec",
            {
                "session_id": "managed-session",
                "node": "node1",
                "command": "uname -a",
            },
        )
    ]
    outcomes = [
        json.loads(line)["outcome"] for line in audit_path.read_text().splitlines()
    ]
    assert outcomes == ["requested", "completed"]


@pytest.mark.asyncio
async def test_bridge_enforces_per_worker_exec_command_budget(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool_raw(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls += 1
            return {"content": [{"type": "text", "text": '{"exit_code":0}'}]}

        async def close(self) -> None:
            pass

    client = FakeClient()
    audit_path = tmp_path / "commands.jsonl"
    bridge = ExecBridge(
        client=client,
        session_id="managed-session",
        nodes=("node1",),
        audit_log=audit_path,
        max_exec_calls=1,
    )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "exec",
            "arguments": {"node": "node1", "command": "uname -a"},
        },
    }

    await bridge.handle(request)
    with pytest.raises(RuntimeError, match="command budget exhausted"):
        await bridge.handle(request)

    assert client.calls == 1
    outcomes = [
        json.loads(line)["outcome"] for line in audit_path.read_text().splitlines()
    ]
    assert outcomes == ["requested", "completed", "budget_exhausted"]


@pytest.mark.asyncio
async def test_verifier_bridge_closes_live_exec_after_evidence_completion(
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def call_tool_raw(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            return {"content": []}

        async def close(self) -> None:
            pass

    client = FakeClient()
    completion_path = tmp_path / "evidence-complete.json"
    bridge = ExecBridge(
        client=client,
        session_id="managed-session",
        nodes=("node1",),
        audit_log=None,
        evidence_complete_log=completion_path,
    )

    listed = await bridge.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    completed = await bridge.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "complete_evidence", "arguments": {}},
        }
    )

    assert listed is not None
    assert [tool["name"] for tool in listed["result"]["tools"]] == [
        "exec",
        "complete_evidence",
    ]
    assert completed is not None
    assert json.loads(completion_path.read_text())["status"] == "complete"
    with pytest.raises(RuntimeError, match="exec is permanently closed"):
        await bridge.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "exec",
                    "arguments": {"node": "node1", "command": "true"},
                },
            }
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_bridge_audits_results_without_persisting_credentials(
    tmp_path: Path,
) -> None:
    class SecretClient:
        async def call_tool_raw(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "exit_code": 7,
                                "stdout": (
                                    "token=ant_secretvalue1234567890\n"
                                    "ldap_default_authtok = directory-bind-value\n"
                                    "bindpw: another-directory-secret"
                                ),
                                "stderr": "IDENTIFIED BY 'database-password'",
                            }
                        ),
                    }
                ]
            }

        async def close(self) -> None:
            pass

    audit_path = tmp_path / "commands.jsonl"
    bridge = ExecBridge(
        client=SecretClient(),
        session_id="managed-session",
        nodes=("node1",),
        audit_log=audit_path,
    )
    response = await bridge.handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "exec",
                "arguments": {
                    "node": "node1",
                    "command": "configure password=database-password " + ("a" * 200),
                },
            },
        }
    )

    assert response is not None
    assert "database-password" in json.dumps(response)
    persisted = audit_path.read_text()
    assert "database-password" not in persisted
    assert "ant_secretvalue" not in persisted
    assert "directory-bind-value" not in persisted
    assert "another-directory-secret" not in persisted
    assert "a" * 200 not in persisted
    completed = json.loads(persisted.splitlines()[1])
    assert completed["return_code"] == 7
    assert completed["duration_ms"] >= 0
    assert completed["result_available"] is True
    assert "[REDACTED]" in completed["stdout"]


@pytest.mark.asyncio
async def test_bridge_records_cluster_expiration_as_fatal(tmp_path: Path) -> None:
    class ExpiredClient:
        async def call_tool_raw(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            raise ClusterExpiredError("managed cluster expired")

        async def close(self) -> None:
            pass

    audit_path = tmp_path / "commands.jsonl"
    fatal_path = tmp_path / "fatal.json"
    bridge = ExecBridge(
        client=ExpiredClient(),
        session_id="managed-session",
        nodes=("node1",),
        audit_log=audit_path,
        fatal_log=fatal_path,
    )

    with pytest.raises(ClusterExpiredError, match="managed cluster expired"):
        await bridge.handle(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "exec",
                    "arguments": {"node": "node1", "command": "true"},
                },
            }
        )

    outcomes = [
        json.loads(line)["outcome"] for line in audit_path.read_text().splitlines()
    ]
    assert outcomes == ["requested", "cluster_expired"]
    assert json.loads(fatal_path.read_text())["type"] == "ClusterExpiredError"


@pytest.mark.asyncio
async def test_agent_process_is_stopped_at_cluster_deadline(tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    with pytest.raises(ClusterExpiredError, match="lease expired"):
        await _communicate_with_cluster_guard(
            process,
            timeout_sec=10,
            fatal_path=tmp_path / "missing-fatal.json",
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
            agent_name="codex",
        )

    assert process.returncode is not None


@pytest.mark.asyncio
async def test_agent_can_finish_offline_after_closing_evidence(tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(0.4); print('finished offline')",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    completion_path = tmp_path / "evidence-complete.json"

    async def close_evidence() -> None:
        await asyncio.sleep(0.01)
        completion_path.write_text('{"status":"complete"}')

    closer = asyncio.create_task(close_evidence())
    stdout, stderr, expiration_message = await _communicate_with_cluster_guard(
        process,
        timeout_sec=2,
        fatal_path=tmp_path / "missing-fatal.json",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=0.25),
        agent_name="codex",
        evidence_complete_path=completion_path,
        allow_offline_finalization=True,
    )
    await closer

    assert stdout.strip() == b"finished offline"
    assert stderr == b""
    assert expiration_message is None


@pytest.mark.asyncio
async def test_agent_shutdown_timeout_preserves_output_after_evidence_completion(
    tmp_path: Path,
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        ('import time; print(\'{"report": "complete"}\', flush=True); time.sleep(30)'),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    completion_path = tmp_path / "evidence-complete.json"
    completion_path.write_text('{"status":"complete"}')

    stdout, stderr, expiration_message = await _communicate_with_cluster_guard(
        process,
        timeout_sec=1,
        fatal_path=tmp_path / "missing-fatal.json",
        lease_expires_at=None,
        agent_name="codex",
        evidence_complete_path=completion_path,
        allow_offline_finalization=True,
        offline_finalization_grace_sec=1,
    )

    assert json.loads(stdout) == {"report": "complete"}
    assert stderr == b""
    assert expiration_message is None
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_expiration_returns_output_emitted_before_forced_shutdown(
    tmp_path: Path,
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        ("import time; print('{\"dimensions\": {}}', flush=True); time.sleep(30)"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, _, expiration_message = await _communicate_with_cluster_guard(
        process,
        timeout_sec=2,
        fatal_path=tmp_path / "missing-fatal.json",
        lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
        agent_name="codex",
        allow_offline_finalization=True,
        offline_finalization_grace_sec=1,
    )

    assert json.loads(stdout) == {"dimensions": {}}
    assert expiration_message is not None
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_runner_recovers_valid_report_when_cluster_expires_during_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"dimensions": {"functionality": {"score": 100}}}
    earlier_report = {"dimensions": {"functionality": {"score": 10}}}

    class FakeBackend:
        def ensure_installed(self) -> None:
            pass

        def build_command(
            self,
            prompt: str,
            schema: dict[str, Any],
            *,
            allowed_tools: tuple[str, ...],
        ) -> list[str]:
            assert "complete_evidence" in prompt
            assert len(allowed_tools) == 2
            return ["fake-agent"]

        def model_args(self, model: str) -> list[str]:
            return ["--model", model]

        def parse_output(self, output: str) -> str:
            return output

        def cleanup(self) -> None:
            pass

    class FakeProcess:
        returncode = 1

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess()

    async def fake_guard(*args: Any, **kwargs: Any) -> tuple[bytes, bytes, str]:
        output = (
            "progress\n"
            f"{json.dumps(earlier_report)}\n"
            f"{json.dumps(report)}\n"
            '{"incomplete":'
        )
        return output.encode(), b"cluster expired", "lease expired"

    monkeypatch.setattr(
        "infraset.agent_runner.get_agent", lambda _judge, _workspace: FakeBackend()
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        "infraset.agent_runner._communicate_with_cluster_guard", fake_guard
    )
    workspace = tmp_path / "verifier"
    workspace.mkdir()
    audit_path = workspace / "audit.jsonl"
    audit_path.write_text('{"node":"node1","command":"true","outcome":"completed"}\n')

    def validate_report(value: dict[str, Any]) -> None:
        if "dimensions" not in value:
            raise ValueError("invalid report")

    parsed, _ = await run_structured_agent(
        agent_name="codex",
        model="test-model",
        prompt="Evaluate",
        schema={"type": "object"},
        session_id="session-1",
        nodes=("node1",),
        endpoint="https://provider.example/mcp",
        workspace=workspace,
        timeout_sec=30,
        audit_path=audit_path,
        allow_offline_finalization=True,
        output_validator=validate_report,
    )

    assert parsed == report
    recovery = json.loads((workspace / "offline-report-recovered.json").read_text())
    assert recovery["reason"] == "lease expired"


@pytest.mark.asyncio
async def test_runner_recovers_valid_report_after_expired_exec_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"dimensions": {"functionality": {"score": 100}}}

    class FakeBackend:
        def ensure_installed(self) -> None:
            pass

        def build_command(
            self,
            prompt: str,
            schema: dict[str, Any],
            *,
            allowed_tools: tuple[str, ...],
        ) -> list[str]:
            assert "do not call exec again" in prompt
            return ["fake-agent"]

        def model_args(self, model: str) -> list[str]:
            return ["--model", model]

        def parse_output(self, output: str) -> str:
            return output

        def cleanup(self) -> None:
            pass

    class FakeProcess:
        returncode = 1

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess()

    async def fake_guard(*args: Any, **kwargs: Any) -> tuple[bytes, bytes, str]:
        kwargs["evidence_complete_path"].write_text('{"status":"complete"}')
        return json.dumps(report).encode(), b"cluster expired", "lease expired"

    monkeypatch.setattr(
        "infraset.agent_runner.get_agent", lambda _judge, _workspace: FakeBackend()
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        "infraset.agent_runner._communicate_with_cluster_guard", fake_guard
    )
    workspace = tmp_path / "verifier"
    workspace.mkdir()
    audit_path = workspace / "audit.jsonl"
    audit_path.write_text(
        '{"node":"node1","command":"true","outcome":"cluster_expired"}\n'
    )

    def validate_report(value: dict[str, Any]) -> None:
        if "dimensions" not in value:
            raise ValueError("invalid report")

    parsed, _ = await run_structured_agent(
        agent_name="codex",
        model="test-model",
        prompt="Evaluate",
        schema={"type": "object"},
        session_id="session-1",
        nodes=("node1",),
        endpoint="https://provider.example/mcp",
        workspace=workspace,
        timeout_sec=30,
        audit_path=audit_path,
        allow_offline_finalization=True,
        output_validator=validate_report,
    )

    assert parsed == report
    recovery = json.loads((workspace / "offline-report-recovered.json").read_text())
    assert recovery == {
        "reason": "lease expired",
        "evidence_collection_closed": True,
        "expired_exec_observed": True,
        "recovered_at": recovery["recovered_at"],
    }


@pytest.mark.asyncio
async def test_runner_recovers_valid_report_when_cli_lingers_after_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"dimensions": {"functionality": {"score": 100}}}

    class FakeBackend:
        def ensure_installed(self) -> None:
            pass

        def build_command(
            self,
            prompt: str,
            schema: dict[str, Any],
            *,
            allowed_tools: tuple[str, ...],
        ) -> list[str]:
            return ["fake-agent"]

        def model_args(self, model: str) -> list[str]:
            return ["--model", model]

        def parse_output(self, output: str) -> str:
            return output

        def cleanup(self) -> None:
            pass

    class FakeProcess:
        returncode = -9

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess()

    async def fake_guard(*args: Any, **kwargs: Any) -> tuple[bytes, bytes, None]:
        kwargs["evidence_complete_path"].write_text('{"status":"complete"}')
        return json.dumps(report).encode(), b"", None

    monkeypatch.setattr(
        "infraset.agent_runner.get_agent", lambda _judge, _workspace: FakeBackend()
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        "infraset.agent_runner._communicate_with_cluster_guard", fake_guard
    )
    workspace = tmp_path / "verifier"
    workspace.mkdir()

    def validate_report(value: dict[str, Any]) -> None:
        if "dimensions" not in value:
            raise ValueError("invalid report")

    parsed, _ = await run_structured_agent(
        agent_name="codex",
        model="test-model",
        prompt="Evaluate",
        schema={"type": "object"},
        session_id="session-1",
        nodes=("node1",),
        endpoint="https://provider.example/mcp",
        workspace=workspace,
        timeout_sec=30,
        allow_offline_finalization=True,
        output_validator=validate_report,
    )

    assert parsed == report
    recovery = json.loads((workspace / "offline-report-recovered.json").read_text())
    assert "after closing live evidence" in recovery["reason"]


def test_claude_uses_invocation_scoped_mcp_config(tmp_path: Path) -> None:
    command = ["claude", "-p", "inspect"]
    server = MCPServerConfig(
        name="infraset_1234",
        transport="stdio",
        command="/venv/bin/python",
        args=("-m", "infraset.exec_bridge", "--session-id", "managed-session"),
        allowed_tools=("exec",),
    )

    config_path = configure_claude_mcp(command, server, tmp_path)

    assert command[-3:] == [
        "--mcp-config",
        str(config_path),
        "--strict-mcp-config",
    ]
    assert json.loads(config_path.read_text()) == {
        "mcpServers": {
            "infraset_1234": {
                "type": "stdio",
                "command": "/venv/bin/python",
                "args": [
                    "-m",
                    "infraset.exec_bridge",
                    "--session-id",
                    "managed-session",
                ],
            }
        }
    }


def test_claude_host_agent_uses_requested_reasoning_effort() -> None:
    command = ["claude", "-p", "inspect"]

    append_reasoning_effort("claude-code", command, "medium")

    assert command[-2:] == ["--effort", "medium"]


def test_codex_host_agent_uses_requested_service_tier() -> None:
    command = ["codex", "exec", "inspect"]

    append_service_tier("codex", command, "fast")

    assert command[-2:] == ["-c", 'service_tier="fast"']


def test_service_tier_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="only by Codex"):
        append_service_tier("claude-code", [], "fast")


def test_codex_uses_invocation_scoped_mcp_config() -> None:
    command = ["codex", "exec", "inspect"]
    server = MCPServerConfig(
        name="infraset_1234",
        transport="stdio",
        command="/venv/bin/python",
        args=("-m", "infraset.exec_bridge", "--session-id", "managed-session"),
        allowed_tools=("exec",),
    )

    configure_codex_mcp(command, server)

    assert "--ignore-user-config" not in command
    assert "--approve-for-me" not in command
    assert 'approval_policy="never"' in command
    assert 'mcp_servers.infraset_1234.default_tools_approval_mode="approve"' in command
    assert 'mcp_servers.infraset_1234.command="/venv/bin/python"' in command
    assert 'mcp_servers.infraset_1234.env_vars=["ANTRIEB_TOKEN"]' in command
    assert (
        'mcp_servers.infraset_1234.args=["-m",'
        '"infraset.exec_bridge","--session-id","managed-session"]' in command
    )


def test_audit_path_is_absolute(tmp_path: Path) -> None:
    bridge_args: list[str] = []
    audit_path = tmp_path / "verifier" / "commands.jsonl"

    append_audit_argument(bridge_args, audit_path)

    assert bridge_args == ["--audit-log", str(audit_path.resolve())]


def test_codex_log_evaluator_ignores_user_mcp_configuration(tmp_path: Path) -> None:
    command = ["codex", "exec", "diagnose"]

    config_path = configure_log_only_agent("codex", command, tmp_path)

    assert config_path is None
    assert "--ignore-user-config" in command
    assert 'approval_policy="never"' in command
    assert "--approve-for-me" not in command
    assert command[-3:] == ["--sandbox", "read-only", "--ephemeral"]


def test_claude_log_evaluator_has_no_tools_or_mcp_servers(tmp_path: Path) -> None:
    command = ["claude", "-p", "diagnose"]

    config_path = configure_log_only_agent("claude-code", command, tmp_path)

    assert config_path is not None
    assert json.loads(config_path.read_text()) == {"mcpServers": {}}
    assert "--strict-mcp-config" in command
    assert command[-3:] == ["--tools", "", "--no-session-persistence"]


def test_log_evaluator_does_not_inherit_cluster_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setenv("ANTRIEB_MCP_URL", "https://provider.example/mcp")
    monkeypatch.setenv("ANTRIEB_SESSION_ID", "managed-session")

    environment = log_agent_environment()

    assert "ANTRIEB_TOKEN" not in environment
    assert "ANTRIEB_MCP_URL" not in environment
    assert "ANTRIEB_SESSION_ID" not in environment


def test_agent_environment_removes_host_initialization_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFRASET_INITIALIZE_CREDENTIALS_FILE", "/run/secrets/init")
    monkeypatch.setenv("INFRASET_INITIALIZE_CREDENTIALS", "must-not-leak")
    monkeypatch.setenv("MODEL_PROVIDER_TOKEN", "model-token")

    environment = agent_environment()

    assert "INFRASET_INITIALIZE_CREDENTIALS_FILE" not in environment
    assert "INFRASET_INITIALIZE_CREDENTIALS" not in environment
    assert environment["MODEL_PROVIDER_TOKEN"] == "model-token"
