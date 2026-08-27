import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, override

import pytest

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from infraset.environment import InfraSetEnvironment
from infraset.errors import ClusterExpiredError


class FakeClient:
    calls: list[tuple[str, dict[str, Any]]]

    def __init__(self, endpoint: str, token: str) -> None:
        self.endpoint = endpoint
        self.token = token
        self.calls = []
        self.closed = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "search":
            fq_name = arguments["fq_name"]
            return {
                "runbook": {
                    "fq_name": fq_name,
                    "body": f"# Platform reference: {fq_name}",
                }
            }
        if name == "provision":
            return {
                "session_id": "cluster-123",
                "nodes": ["node1", "node2"],
                "ttl_seconds": 600,
                "expires_at": "2099-01-01T00:00:00Z",
            }
        if name == "exec":
            if "harbor_i=0" in arguments["command"]:
                return {
                    "stdout": "__INFRASET_DONE__:0",
                    "stderr": "",
                    "exit_code": 0,
                }
            return {"stdout": "", "stderr": "", "exit_code": 0}
        return {"deleted": True}

    async def call_tool_raw(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        parsed = await self.call_tool(name, arguments)
        return {"content": [{"type": "text", "text": json.dumps(parsed)}]}

    @staticmethod
    def parse_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
        del name
        return json.loads(result["content"][0]["text"])

    async def close(self) -> None:
        self.closed = True


def make_environment(
    tmp_path: Path,
    definition: str | None = None,
    **environment_kwargs: Any,
) -> InfraSetEnvironment:
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "infraset.toml").write_text(
        definition or ('cluster = ["ubuntu24.04 x2"]\ncontrol_node = "node1"\n')
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return InfraSetEnvironment(
        environment_dir=environment_dir,
        environment_name="example",
        session_id="trial-env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        logger=logging.getLogger("test"),
        **environment_kwargs,
    )


@pytest.mark.asyncio
async def test_environment_owns_cluster_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", FakeClient)
    environment = make_environment(tmp_path)

    await environment.start(force_build=False)

    client = environment._client
    assert isinstance(client, FakeClient)
    assert environment.remote_session_id == "cluster-123"
    assert environment.nodes == ("node1", "node2")
    assert environment.cluster_ttl_seconds == 600
    assert environment.cluster_expires_at == datetime(2099, 1, 1, tzinfo=UTC)
    assert client.calls[0] == (
        "search",
        {"type": "runbook", "fq_name": "antrieb/primer"},
    )
    assert client.calls[1] == (
        "provision",
        {"cluster": ["ubuntu24.04 x2"]},
    )
    assert all(
        "secret" not in arguments.get("command", "") for _, arguments in client.calls
    )

    await environment.stop(delete=False)

    assert client.calls[-1] == (
        "delete",
        {"type": "cluster", "name": "cluster-123"},
    )
    assert client.closed


@pytest.mark.asyncio
async def test_environment_recreates_and_repeats_harness_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecreateClient(FakeClient):
        instances: list["RecreateClient"] = []

        def __init__(self, endpoint: str, token: str) -> None:
            super().__init__(endpoint, token)
            self.number = len(self.instances) + 1
            self.instances.append(self)

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            result = await super().call_tool(name, arguments)
            if name == "provision":
                return {**result, "session_id": f"cluster-{self.number}"}
            return result

    initialize_runs = 0
    prepare_runs = 0
    lifecycle: list[str] = []

    async def fake_initialize() -> None:
        nonlocal initialize_runs
        initialize_runs += 1
        lifecycle.append("initialize")

    async def fake_prepare() -> None:
        nonlocal prepare_runs
        prepare_runs += 1
        lifecycle.append("prepare")

    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", RecreateClient)
    environment = make_environment(
        tmp_path,
        'cluster = ["ubuntu24.04 x2"]\nmax_clusters = 2\n',
    )
    monkeypatch.setattr(environment, "initialize", fake_initialize)
    monkeypatch.setattr(environment, "prepare", fake_prepare)

    await environment.start(force_build=False)
    await environment.recreate()

    assert environment.remote_session_id == "cluster-2"
    assert environment.clusters_provisioned == 2
    assert initialize_runs == 2
    assert prepare_runs == 2
    assert lifecycle == ["initialize", "prepare", "initialize", "prepare"]
    assert RecreateClient.instances[0].closed
    assert any(name == "delete" for name, _ in RecreateClient.instances[0].calls)
    assert [
        name for name, _ in RecreateClient.instances[0].calls if name == "search"
    ] == ["search"]
    assert not any(name == "search" for name, _ in RecreateClient.instances[1].calls)
    assert not any(
        name == "save"
        for client in RecreateClient.instances
        for name, _ in client.calls
    )
    with pytest.raises(RuntimeError, match="quota exhausted"):
        await environment.recreate()
    assert environment.remote_session_id == "cluster-2"

    await environment.stop(delete=True)


@pytest.mark.asyncio
async def test_initialize_registers_rhel_without_persisting_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RhsmClient(FakeClient):
        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            if name == "search":
                fq_name = arguments["fq_name"]
                return {
                    "runbook": {
                        "fq_name": fq_name,
                        "body": f"# Platform reference: {fq_name}",
                    }
                }
            if name == "provision":
                return {
                    "session_id": "cluster-rhel",
                    "nodes": ["node1", "node2"],
                    "node_images": [
                        {
                            "node": "node1",
                            "ani": "antrieb:rhel7.9:v1",
                            "image_interface": {
                                "version": 1,
                                "scope": "exact-ani",
                                "ani": "antrieb:rhel7.9:v1",
                                "execution": {
                                    "transport": "ssh",
                                    "user": "antrieb",
                                    "privilegeEscalation": "sudo-noninteractive",
                                },
                            },
                        },
                        {"node": "node2", "ani": "antrieb:rhel9.8:v1"},
                    ],
                }
            if name == "exec":
                return {"stdout": "", "stderr": "", "exit_code": 0}
            return {"deleted": True}

    credentials = tmp_path / "initialize.credentials"
    credentials.write_text("subscription-user\nsubscription-password\n")
    credentials.chmod(0o600)
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setenv("HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setattr("infraset.environment.AntriebClient", RhsmClient)
    environment = make_environment(
        tmp_path,
        'cluster = ["rhel9.8 x2"]\ninitialize = ["rhsm"]\n',
    )

    await environment.start(force_build=False)

    client = environment._client
    assert isinstance(client, RhsmClient)
    rhsm_calls = [
        arguments
        for name, arguments in client.calls
        if name == "exec" and "HARBOR_ANTRIEB_INITIALIZE_USERNAME" in arguments["command"]
    ]
    assert {call["node"] for call in rhsm_calls} == {"node1", "node2"}
    calls_by_node = {call["node"]: call for call in rhsm_calls}
    assert calls_by_node["node1"]["command"].startswith(
        'sudo -n env HARBOR_ANTRIEB_INITIALIZE_USERNAME="$HARBOR_ANTRIEB_INITIALIZE_USERNAME" '
        'HARBOR_ANTRIEB_INITIALIZE_PASSWORD="$HARBOR_ANTRIEB_INITIALIZE_PASSWORD" '
        "/bin/sh -c "
    )
    assert calls_by_node["node2"]["command"].startswith("set -eu\n")
    assert all("subscription-password" not in call["command"] for call in rhsm_calls)
    assert all(
        "getent hosts subscription.rhsm.redhat.com" in call["command"]
        for call in rhsm_calls
    )
    assert all("python3" not in call["command"] for call in rhsm_calls)
    assert all("--auto-attach" not in call["command"] for call in rhsm_calls)
    assert all(
        "install -d -m 0750 /var/lib/rhsm/cache" in call["command"]
        for call in rhsm_calls
    )
    assert all(
        call["secret_env"]
        == {
            "HARBOR_ANTRIEB_INITIALIZE_USERNAME": "subscription-user",
            "HARBOR_ANTRIEB_INITIALIZE_PASSWORD": "subscription-password",
        }
        for call in rhsm_calls
    )
    assert environment.initialization_results[0].name == "rhsm"
    initialize_report = (
        environment.trial_paths.trial_dir / "initialize-report.json"
    ).read_text()
    provision_report = (
        environment.trial_paths.trial_dir / "provision-response.json"
    ).read_text()
    assert "subscription-user" not in initialize_report + provision_report
    assert "subscription-password" not in initialize_report + provision_report

    await environment.stop(delete=True)


@pytest.mark.asyncio
async def test_failed_initialize_deletes_cluster_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedRhsmClient(FakeClient):
        latest: "FailedRhsmClient | None" = None

        def __init__(self, endpoint: str, token: str) -> None:
            super().__init__(endpoint, token)
            self.__class__.latest = self

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if name == "provision":
                self.calls.append((name, arguments))
                return {
                    "session_id": "cluster-rhel",
                    "nodes": ["node1"],
                    "node_images": [{"node": "node1", "ani": "antrieb:rhel9.8:v1"}],
                }
            if name == "exec" and "subscription-manager" in arguments["command"]:
                self.calls.append((name, arguments))
                return {
                    "stdout": ("HARBOR_ANTRIEB_INITIALIZE_STATUS=provider-unreachable\n"),
                    "stderr": "",
                    "exit_code": 7,
                }
            return await super().call_tool(name, arguments)

    credentials = tmp_path / "initialize.credentials"
    credentials.write_text("subscription-user\nsubscription-password\n")
    credentials.chmod(0o600)
    prepare_called = False

    async def fake_prepare() -> None:
        nonlocal prepare_called
        prepare_called = True

    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setenv("HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setattr("infraset.environment.AntriebClient", FailedRhsmClient)
    environment = make_environment(
        tmp_path,
        'cluster = ["rhel9.8"]\ninitialize = ["rhsm"]\n',
    )
    monkeypatch.setattr(environment, "prepare", fake_prepare)

    with pytest.raises(
        RuntimeError,
        match="rhsm initialization failed: node1=provider-unreachable",
    ):
        await environment.start(force_build=False)

    client = FailedRhsmClient.latest
    assert client is not None
    assert prepare_called is False
    assert client.calls[-1] == (
        "delete",
        {"type": "cluster", "name": "cluster-rhel"},
    )
    assert client.closed


@pytest.mark.asyncio
async def test_initializer_rejects_exposed_credentials_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RhsmClient(FakeClient):
        latest: "RhsmClient | None" = None

        def __init__(self, endpoint: str, token: str) -> None:
            super().__init__(endpoint, token)
            self.__class__.latest = self

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if name == "provision":
                self.calls.append((name, arguments))
                return {
                    "session_id": "cluster-rhel",
                    "nodes": ["node1"],
                    "node_images": [{"node": "node1", "ani": "antrieb:rhel9.8:v1"}],
                }
            return await super().call_tool(name, arguments)

    credentials = tmp_path / "initialize.credentials"
    credentials.write_text("subscription-user\nsubscription-password\n")
    credentials.chmod(0o644)
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setenv("HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setattr("infraset.environment.AntriebClient", RhsmClient)
    environment = make_environment(
        tmp_path,
        'cluster = ["rhel9.8"]\ninitialize = ["rhsm"]\n',
    )

    with pytest.raises(RuntimeError, match="must not be group/world accessible"):
        await environment.start(force_build=False)

    client = RhsmClient.latest
    assert client is not None
    assert not any(name == "exec" for name, _ in client.calls)
    assert client.calls[-1] == (
        "delete",
        {"type": "cluster", "name": "cluster-rhel"},
    )


@pytest.mark.asyncio
async def test_malformed_base_runbook_aborts_before_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MalformedRunbookClient(FakeClient):
        latest: "MalformedRunbookClient | None" = None

        def __init__(self, endpoint: str, token: str) -> None:
            super().__init__(endpoint, token)
            self.__class__.latest = self

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            if name == "search":
                self.calls.append((name, arguments))
                return {"runbook": {"fq_name": arguments["fq_name"], "body": ""}}
            return await super().call_tool(name, arguments)

    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", MalformedRunbookClient)
    environment = make_environment(tmp_path)

    with pytest.raises(RuntimeError, match="malformed base runbook"):
        await environment.start(force_build=False)

    client = MalformedRunbookClient.latest
    assert client is not None
    assert not any(name == "provision" for name, _ in client.calls)
    assert client.closed


@pytest.mark.asyncio
async def test_environment_rejects_exec_after_reported_expiration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", FakeClient)
    environment = make_environment(tmp_path)
    await environment.start(force_build=False)
    client = environment._client
    assert isinstance(client, FakeClient)
    call_count = len(client.calls)
    environment.cluster_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    with pytest.raises(ClusterExpiredError, match="lease expired"):
        await environment.exec_on_node("node1", "true")

    assert len(client.calls) == call_count
    await environment.stop(delete=True)
    assert client.closed
    assert not any(name == "delete" for name, _ in client.calls[call_count:])


def test_environment_applies_ai_prepare_cli_overrides(tmp_path: Path) -> None:
    environment = make_environment(
        tmp_path,
        (
            'cluster = ["ubuntu24.04 x2"]\n'
            "[prepare]\n"
            "enabled = true\n"
            'mode = "ai"\n'
            'agent = "claude-code"\n'
        ),
        prepare_agent="codex",
        prepare_model="local/kimi-k3",
        prepare_reasoning_effort="high",
        prepare_timeout_sec=777,
    )

    assert environment.definition is not None
    assert environment.definition.prepare.agent == "codex"
    assert environment.definition.prepare.model == "local/kimi-k3"
    assert environment.definition.prepare.reasoning_effort == "high"
    assert environment.definition.prepare.timeout_sec == 777


@pytest.mark.asyncio
async def test_static_prepare_runs_setup_then_captures_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PrepareClient(FakeClient):
        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            if name == "search":
                fq_name = arguments["fq_name"]
                return {
                    "runbook": {
                        "fq_name": fq_name,
                        "body": f"# Platform reference: {fq_name}",
                    }
                }
            if name == "provision":
                return {"session_id": "cluster-123", "nodes": ["node1", "node2"]}
            if name == "exec":
                command = arguments["command"]
                if command.startswith("observe-"):
                    return {
                        "stdout": f"baseline-{arguments['node']}\n",
                        "stderr": "",
                        "exit_code": 0,
                    }
                return {"stdout": "", "stderr": "", "exit_code": 0}
            return {"deleted": True}

    definition = (
        'cluster = ["ubuntu24.04 x2"]\n'
        'control_node = "node1"\n'
        "[prepare]\n"
        "enabled = true\n"
    )
    environment = make_environment(tmp_path, definition)
    prepare_dir = tmp_path / "task" / "prepare"
    prepare_dir.mkdir()
    (prepare_dir / "setup.toml").write_text(
        "timeout_sec = 30\n"
        "[[steps]]\n"
        'id = "seed-node1"\n'
        "stage = 10\n"
        'node = "node1"\n'
        'command = "seed-node1"\n'
        "[[steps]]\n"
        'id = "seed-node2"\n'
        "stage = 10\n"
        'node = "node2"\n'
        'command = "seed-node2"\n'
        "[[steps]]\n"
        'id = "finish-seed"\n'
        "stage = 20\n"
        'node = "node1"\n'
        'command = "finish-seed"\n'
    )
    (prepare_dir / "baseline.toml").write_text(
        "timeout_sec = 30\n"
        "[[observations]]\n"
        'id = "state-node1"\n'
        'node = "node1"\n'
        'command = "observe-node1"\n'
        "[[observations]]\n"
        'id = "state-node2"\n'
        'node = "node2"\n'
        'command = "observe-node2"\n'
    )
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", PrepareClient)

    await environment.start(force_build=False)

    client = environment._client
    assert isinstance(client, PrepareClient)
    managed_commands = [
        arguments["command"]
        for name, arguments in client.calls
        if name == "exec"
        and arguments["command"]
        in {
            "seed-node1",
            "seed-node2",
            "finish-seed",
            "observe-node1",
            "observe-node2",
        }
    ]
    assert set(managed_commands[:2]) == {"seed-node1", "seed-node2"}
    assert managed_commands[2] == "finish-seed"
    assert set(managed_commands[3:]) == {"observe-node1", "observe-node2"}

    output_dir = environment.trial_paths.trial_dir / "prepare"
    assert (output_dir / "setup-commands.jsonl").is_file()
    assert (output_dir / "baseline-commands.jsonl").is_file()
    baseline = json.loads((output_dir / "baseline-report.json").read_text())
    assert baseline["source"] == "static-prepare"
    assert [node["name"] for node in baseline["nodes"]] == ["node1", "node2"]
    assert baseline["nodes"][0]["facts"][0]["observation"] == "baseline-node1"

    await environment.stop(delete=True)


@pytest.mark.asyncio
async def test_failed_prepare_stops_later_stages_and_deletes_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedPrepareClient(FakeClient):
        latest: "FailedPrepareClient | None" = None

        def __init__(self, endpoint: str, token: str) -> None:
            super().__init__(endpoint, token)
            self.__class__.latest = self

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            if name == "search":
                fq_name = arguments["fq_name"]
                return {
                    "runbook": {
                        "fq_name": fq_name,
                        "body": f"# Platform reference: {fq_name}",
                    }
                }
            if name == "provision":
                return {"session_id": "cluster-123", "nodes": ["node1", "node2"]}
            if name == "exec":
                return {
                    "stdout": "",
                    "stderr": "setup failed" if arguments["command"] == "fail" else "",
                    "exit_code": 9 if arguments["command"] == "fail" else 0,
                }
            return {"deleted": True}

    environment = make_environment(
        tmp_path,
        (
            'cluster = ["ubuntu24.04 x2"]\n'
            'control_node = "node1"\n'
            "[prepare]\n"
            "enabled = true\n"
        ),
    )
    prepare_dir = tmp_path / "task" / "prepare"
    prepare_dir.mkdir()
    (prepare_dir / "setup.toml").write_text(
        "[[steps]]\n"
        'id = "failing-step"\n'
        "stage = 10\n"
        'node = "node1"\n'
        'command = "fail"\n'
        "[[steps]]\n"
        'id = "must-not-run"\n'
        "stage = 20\n"
        'node = "node2"\n'
        'command = "later"\n'
    )
    (prepare_dir / "baseline.toml").write_text(
        '[[observations]]\nid = "unused"\nnode = "node1"\ncommand = "unused"\n'
    )
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", FailedPrepareClient)

    with pytest.raises(RuntimeError, match="static setup failed"):
        await environment.start(force_build=False)

    client = FailedPrepareClient.latest
    assert client is not None
    commands = [
        arguments["command"] for name, arguments in client.calls if name == "exec"
    ]
    assert "fail" in commands
    assert "later" not in commands
    assert "unused" not in commands
    assert client.calls[-1] == (
        "delete",
        {"type": "cluster", "name": "cluster-123"},
    )
    assert client.closed


@pytest.mark.asyncio
async def test_exec_uses_bash_for_harbor_pipefail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", FakeClient)
    environment = make_environment(tmp_path)
    await environment.start(force_build=False)

    await environment.exec(
        "set -o pipefail; true",
        cwd="/tmp",
        env={"EXAMPLE": "value"},
        user="root",
    )

    client = environment._client
    assert isinstance(client, FakeClient)
    launch_calls = [
        arguments
        for name, arguments in client.calls
        if name == "exec" and "nohup setsid" in arguments["command"]
    ]
    assert launch_calls
    launch = launch_calls[-1]
    assert launch["session_id"] == "cluster-123"
    assert launch["node"] == "node1"
    assert (
        "env HOME=/root USER=root LOGNAME=root EXAMPLE=value bash -lc"
        in launch["command"]
    )
    assert "cd /tmp && set -o pipefail; true" in launch["command"]


@pytest.mark.asyncio
async def test_exec_detaches_and_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ResultClient(FakeClient):
        detached_result = False

        @override
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append((name, arguments))
            if name == "search":
                fq_name = arguments["fq_name"]
                return {
                    "runbook": {
                        "fq_name": fq_name,
                        "body": f"# Platform reference: {fq_name}",
                    }
                }
            if name == "provision":
                return {"session_id": "cluster-123", "nodes": ["node1", "node2"]}
            if name == "exec" and "harbor_i=0" in arguments["command"]:
                if not self.detached_result:
                    return {
                        "stdout": "__INFRASET_DONE__:0",
                        "stderr": "",
                        "exit_code": 0,
                    }
                return {
                    "stdout": "__INFRASET_DONE__:23\ncommand output\n",
                    "stderr": "command error\n",
                    "exit_code": 0,
                }
            return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setenv("ANTRIEB_TOKEN", "secret")
    monkeypatch.setattr("infraset.environment.AntriebClient", ResultClient)
    environment = make_environment(tmp_path)
    await environment.start(force_build=False)
    client = environment._client
    assert isinstance(client, ResultClient)
    client.detached_result = True

    result = await environment.exec("false", timeout_sec=90)

    assert result.return_code == 23
    assert result.stdout == "command output\n"
    assert result.stderr == "command error\n"
    launch = next(
        arguments["command"]
        for name, arguments in reversed(client.calls)
        if name == "exec" and "nohup setsid" in arguments["command"]
    )
    assert "timeout --signal=TERM --kill-after=5s 90s" in launch


@pytest.mark.asyncio
async def test_detached_exec_protocol_with_local_shell(tmp_path: Path) -> None:
    class LocalExecClient:
        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            assert name == "exec"
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                arguments["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "exit_code": process.returncode,
            }

        async def close(self) -> None:
            pass

    environment = make_environment(tmp_path)
    environment._client = LocalExecClient()  # ty: ignore[invalid-assignment]
    environment.remote_session_id = "local-shell"
    environment.nodes = ("node1",)
    environment._REMOTE_EXEC_DIR = str(tmp_path / "detached-exec")

    result = await environment.exec(
        "printf 'hello from stdout\\n'; printf 'hello from stderr\\n' >&2; exit 7",
        timeout_sec=5,
    )

    assert result.return_code == 7
    assert result.stdout == "hello from stdout\n"
    assert result.stderr is not None
    assert "hello from stderr\n" in result.stderr
