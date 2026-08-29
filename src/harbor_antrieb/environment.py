from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shlex
import tarfile
import tempfile
import tomllib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from harbor_antrieb.client import AntriebClient, AntriebMCPError
from harbor_antrieb.config import AntriebDefinition, PrepareConfig
from harbor_antrieb.errors import ClusterExpiredError
from harbor_antrieb.initializers import InitializationResult, run_initializers
from harbor_antrieb.runbooks import BaseRunbook


class AntriebEnvironment(BaseEnvironment):
    """A Harbor environment backed by one provider-managed Antrieb cluster."""

    _TRANSFER_CHUNK_SIZE = 48 * 1024
    _REMOTE_EXEC_DIR = "/tmp"
    _EXEC_DONE_PREFIX = "__INFRASET_DONE__:"
    _EXEC_RUNNING = "__INFRASET_RUNNING__"

    def __init__(
        self,
        *args: Any,
        prepare_mode: str | None = None,
        prepare_agent: str | None = None,
        prepare_model: str | None = None,
        prepare_reasoning_effort: str | None = None,
        prepare_timeout_sec: int | None = None,
        **kwargs: Any,
    ) -> None:
        environment_dir = Path(kwargs.get("environment_dir", args[0] if args else ""))
        definition_path = environment_dir / "harbor_antrieb.toml"
        self.definition = (
            AntriebDefinition.model_validate(tomllib.loads(definition_path.read_text()))
            if definition_path.is_file()
            else None
        )
        prepare_overrides = {
            key: value
            for key, value in {
                "mode": prepare_mode,
                "agent": prepare_agent,
                "model": prepare_model,
                "reasoning_effort": prepare_reasoning_effort,
                "timeout_sec": prepare_timeout_sec,
            }.items()
            if value is not None
        }
        if self.definition is not None and prepare_overrides:
            prepare = PrepareConfig.model_validate(
                {**self.definition.prepare.model_dump(), **prepare_overrides}
            )
            self.definition = self.definition.model_copy(update={"prepare": prepare})
        self.remote_session_id: str | None = None
        self.nodes: tuple[str, ...] = ()
        self.cluster_ttl_seconds: int | None = None
        self.cluster_expires_at: datetime | None = None
        self.clusters_provisioned = 0
        self.base_runbooks: tuple[BaseRunbook, ...] = ()
        self.node_images: dict[str, dict[str, Any]] = {}
        self.initialization_results: tuple[InitializationResult, ...] = ()
        self._client: AntriebClient | None = None
        super().__init__(*args, **kwargs)

    @classmethod
    @override
    def preflight(cls) -> None:
        if not os.environ.get("ANTRIEB_TOKEN"):
            raise SystemExit("Antrieb's Antrieb provider requires ANTRIEB_TOKEN")

    @staticmethod
    @override
    def type() -> str:
        return "harbor_antrieb"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    @override
    def _validate_definition(self) -> None:
        definition_path = self.environment_dir / "harbor_antrieb.toml"
        if self.definition is None:
            raise FileNotFoundError(
                f"Antrieb environment definition not found: {definition_path}"
            )
        conflicting = (
            "Dockerfile",
            "docker-compose.yaml",
            "docker-compose.yml",
        )
        if any((self.environment_dir / name).exists() for name in conflicting):
            raise ValueError(
                "Antrieb uses environment/infraset.toml, not Dockerfile or Compose"
            )

    def _require_started(self) -> tuple[AntriebClient, str]:
        if self._client is None or self.remote_session_id is None:
            raise RuntimeError("Antrieb environment has not been started")
        return self._client, self.remote_session_id

    def assert_cluster_active(self) -> None:
        """Fail locally once the provider-reported hard lease deadline passes."""
        if (
            self.cluster_expires_at is not None
            and datetime.now(UTC) >= self.cluster_expires_at
        ):
            raise ClusterExpiredError(
                "The Antrieb managed cluster lease expired at "
                f"{self.cluster_expires_at.isoformat()}"
            )

    @staticmethod
    def _parse_expiration(provisioned: dict[str, Any]) -> datetime | None:
        expires_at = provisioned.get("expires_at")
        if isinstance(expires_at, str):
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise RuntimeError(
                    f"Antrieb returned an invalid expires_at value: {expires_at!r}"
                ) from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        ttl_seconds = provisioned.get("ttl_seconds")
        if isinstance(ttl_seconds, (int, float)):
            return datetime.now(UTC) + timedelta(seconds=float(ttl_seconds))
        return None

    @property
    def endpoint(self) -> str:
        assert self.definition is not None
        return os.environ.get("ANTRIEB_MCP_URL", self.definition.endpoint)

    def _save_provision_response(self, response: dict[str, Any]) -> None:
        """Persist the raw MCP provision result for post-run diagnostics.

        This stores only the provider response. The request headers, including the
        Antrieb bearer token, are never included in this file.
        """
        path = self.trial_paths.trial_dir / "provision-response.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(response, indent=2) + "\n")
        temporary.replace(path)

    def _load_node_images(self, provisioned: dict[str, Any]) -> None:
        raw_node_images = provisioned.get("node_images")
        if raw_node_images is None:
            self.node_images = {}
            return
        if not isinstance(raw_node_images, list):
            raise RuntimeError("Antrieb returned malformed node_images")
        node_images: dict[str, dict[str, Any]] = {}
        for item in raw_node_images:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("node"), str)
                or not isinstance(item.get("ani"), str)
            ):
                raise RuntimeError("Antrieb returned malformed node_images")
            node = item["node"]
            if node in node_images:
                raise RuntimeError(f"Antrieb returned duplicate node image for {node}")
            node_images[node] = item
        if set(node_images) != set(self.nodes):
            raise RuntimeError(
                "Antrieb node_images do not match the provisioned node names"
            )
        self.node_images = node_images

    def _save_initialization_report(self) -> None:
        path = self.trial_paths.trial_dir / "initialize-report.json"
        temporary = path.with_suffix(".json.tmp")
        report = {
            "status": "completed",
            "initializers": [
                result.as_dict() for result in self.initialization_results
            ],
        }
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    @override
    async def start(self, force_build: bool) -> None:
        del force_build
        assert self.definition is not None
        if self.clusters_provisioned >= self.definition.max_clusters:
            raise RuntimeError(
                "Antrieb managed-cluster quota exhausted: "
                f"{self.clusters_provisioned}/{self.definition.max_clusters}"
            )
        token = os.environ.get("ANTRIEB_TOKEN")
        if not token:
            raise RuntimeError("ANTRIEB_TOKEN is not set")
        self._client = AntriebClient(self.endpoint, token)
        arguments: dict[str, Any] = {"cluster": self.definition.cluster}
        if self.definition.networks is not None:
            arguments["networks"] = self.definition.networks
        if self.definition.nics is not None:
            arguments["nics"] = self.definition.nics
        try:
            await self._load_base_runbooks()
            try:
                raw_provision_response = await self._client.call_tool_raw(
                    "provision", arguments
                )
            except AntriebMCPError as exc:
                self._save_provision_response(exc.payload)
                raise
            self._save_provision_response(raw_provision_response)
            provisioned = self._client.parse_tool_result(
                "provision", raw_provision_response
            )
            self.remote_session_id = str(provisioned["session_id"])
            self.nodes = tuple(str(node) for node in provisioned.get("nodes", []))
            self._load_node_images(provisioned)
            self.clusters_provisioned += 1
            ttl_seconds = provisioned.get("ttl_seconds")
            self.cluster_ttl_seconds = (
                int(ttl_seconds) if isinstance(ttl_seconds, (int, float)) else None
            )
            self.cluster_expires_at = self._parse_expiration(provisioned)
            if self.definition.control_node not in self.nodes:
                raise RuntimeError(
                    f"control_node {self.definition.control_node!r} is not in "
                    f"provisioned nodes {list(self.nodes)!r}"
                )
            await self.initialize()
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self.prepare()
        except BaseException:
            try:
                await self.stop(delete=True)
            except Exception:
                self.logger.exception("Failed to clean up Antrieb after start failure")
            raise

    async def _load_base_runbooks(self) -> None:
        """Fetch task-declared provider documentation once and retain it in memory."""
        if self.base_runbooks:
            return
        assert self.definition is not None
        if self._client is None:
            raise RuntimeError("Antrieb Antrieb client is unavailable")
        loaded: list[BaseRunbook] = []
        for fq_name in self.definition.base_runbooks:
            try:
                response = await self._client.call_tool(
                    "search", {"type": "runbook", "fq_name": fq_name}
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read required Antrieb base runbook {fq_name!r}"
                ) from exc
            runbook = response.get("runbook")
            if not isinstance(runbook, dict):
                raise RuntimeError(
                    f"Antrieb returned malformed base runbook {fq_name!r}"
                )
            returned_name = runbook.get("fq_name")
            body = runbook.get("body")
            if (
                returned_name != fq_name
                or not isinstance(body, str)
                or not body.strip()
            ):
                raise RuntimeError(
                    f"Antrieb returned malformed base runbook {fq_name!r}"
                )
            loaded.append(BaseRunbook(fq_name=fq_name, body=body))
        self.base_runbooks = tuple(loaded)

    async def recreate(self) -> None:
        """Replace the cluster and repeat initialization and preparation."""
        assert self.definition is not None
        if self.clusters_provisioned >= self.definition.max_clusters:
            raise RuntimeError(
                "Antrieb managed-cluster quota exhausted: "
                f"{self.clusters_provisioned}/{self.definition.max_clusters}"
            )
        await self.stop(delete=True)
        await self.start(force_build=False)

    @override
    async def stop(self, delete: bool) -> None:
        del delete
        client = self._client
        session_id = self.remote_session_id
        if client is None:
            return
        delete_error: Exception | None = None
        try:
            if session_id and (
                self.cluster_expires_at is None
                or datetime.now(UTC) < self.cluster_expires_at
            ):
                for attempt in range(3):
                    try:
                        await client.call_tool(
                            "delete", {"type": "cluster", "name": session_id}
                        )
                        delete_error = None
                        break
                    except ClusterExpiredError:
                        delete_error = None
                        break
                    except Exception as exc:
                        delete_error = exc
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
        finally:
            await client.close()
            self._client = None
            self.remote_session_id = None
            self.nodes = ()
            self.cluster_ttl_seconds = None
            self.cluster_expires_at = None
            self.node_images = {}
            self.initialization_results = ()
        if delete_error is not None:
            raise RuntimeError(
                f"Failed to delete managed Antrieb cluster {session_id} after 3 attempts"
            ) from delete_error

    async def exec_on_node(
        self,
        node: str,
        command: str,
        secret_env: dict[str, str] | None = None,
    ) -> ExecResult:
        exec_result = await self._exec_on_node_raw(node, command, secret_env)
        await self._emit_output(exec_result)
        return exec_result

    async def _exec_on_node_raw(
        self,
        node: str,
        command: str,
        secret_env: dict[str, str] | None = None,
    ) -> ExecResult:
        self.assert_cluster_active()
        client, session_id = self._require_started()
        if node not in self.nodes:
            raise ValueError(
                f"Unknown Antrieb node {node!r}; expected one of {self.nodes}"
            )
        arguments: dict[str, Any] = {
            "session_id": session_id,
            "node": node,
            "command": command,
        }
        if secret_env:
            arguments["secret_env"] = secret_env
        result = await client.call_tool("exec", arguments)
        exec_result = ExecResult(
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
            return_code=int(result.get("exit_code", result.get("return_code", 1))),
        )
        return exec_result

    async def _emit_output(self, exec_result: ExecResult) -> None:
        callback = self._output_callback()
        if callback:
            if exec_result.stdout:
                await callback(exec_result.stdout, "stdout")
            if exec_result.stderr:
                await callback(exec_result.stderr, "stderr")

    async def _exec_detached(
        self,
        node: str,
        command: str,
        timeout_sec: int | None,
    ) -> ExecResult:
        """Run a Harbor command without holding one Antrieb HTTP request open."""
        execution_id = uuid.uuid4().hex
        execution_dir = f"{self._REMOTE_EXEC_DIR}/.harbor-antrieb-{execution_id}"
        prefix = f"{execution_dir}/result"
        stdout_path = f"{prefix}.stdout"
        stderr_path = f"{prefix}.stderr"
        status_path = f"{prefix}.status"
        status_tmp_path = f"{status_path}.tmp"
        pid_path = f"{prefix}.pid"

        executed_command = command
        if timeout_sec is not None:
            executed_command = (
                f"timeout --signal=TERM --kill-after=5s {max(1, timeout_sec)}s "
                f"{command}"
            )
        runner = (
            f"{executed_command}\n"
            "harbor_rc=$?\n"
            f"printf '%s\\n' \"$harbor_rc\" > {shlex.quote(status_tmp_path)} && "
            f"mv {shlex.quote(status_tmp_path)} {shlex.quote(status_path)}"
        )
        launcher = (
            f"nohup setsid bash -c {shlex.quote(runner)} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)} "
            f"</dev/null & printf '%s\\n' \"$!\" > {shlex.quote(pid_path)}"
        )
        launch_command = (
            f"umask 077 && mkdir -p {shlex.quote(self._REMOTE_EXEC_DIR)} && "
            f"mkdir -m 700 {shlex.quote(execution_dir)} && "
            f": > {shlex.quote(stdout_path)} && "
            f": > {shlex.quote(stderr_path)} && "
            f"rm -f {shlex.quote(status_path)} {shlex.quote(status_tmp_path)} "
            f"{shlex.quote(pid_path)} && bash -c {shlex.quote(launcher)}"
        )
        launch_result = await self._exec_on_node_raw(node, launch_command)
        if launch_result.return_code != 0:
            try:
                await self._exec_on_node_raw(
                    node,
                    f"rm -f {shlex.quote(stdout_path)} {shlex.quote(stderr_path)} "
                    f"{shlex.quote(status_path)} {shlex.quote(status_tmp_path)} "
                    f"{shlex.quote(pid_path)}; "
                    f"rmdir {shlex.quote(execution_dir)} 2>/dev/null || true",
                )
            except BaseException:
                pass
            await self._emit_output(launch_result)
            return launch_result

        poll_command = (
            'harbor_i=0; while [ "$harbor_i" -lt 20 ]; do '
            f"if [ -f {shlex.quote(status_path)} ]; then "
            f"printf '%s' {shlex.quote(self._EXEC_DONE_PREFIX)}; "
            f"cat {shlex.quote(status_path)}; "
            f"cat {shlex.quote(stdout_path)}; "
            f"cat {shlex.quote(stderr_path)} >&2; "
            f"rm -f {shlex.quote(stdout_path)} {shlex.quote(stderr_path)} "
            f"{shlex.quote(status_path)} {shlex.quote(pid_path)}; "
            f"rmdir {shlex.quote(execution_dir)} 2>/dev/null || true; exit 0; fi; "
            "harbor_i=$((harbor_i + 1)); sleep 0.5; done; "
            f"printf %s {shlex.quote(self._EXEC_RUNNING)}"
        )
        try:
            while True:
                poll_result = await self._exec_on_node_raw(node, poll_command)
                if poll_result.return_code != 0:
                    await self._emit_output(poll_result)
                    return poll_result
                stdout = poll_result.stdout or ""
                if stdout.startswith(self._EXEC_DONE_PREFIX):
                    completion = stdout.removeprefix(self._EXEC_DONE_PREFIX)
                    status_line, separator, command_stdout = completion.partition("\n")
                    if not separator:
                        command_stdout = ""
                    try:
                        return_code = int(status_line)
                    except ValueError as exc:
                        raise RuntimeError(
                            "Antrieb detached exec returned a malformed exit code"
                        ) from exc
                    exec_result = ExecResult(
                        stdout=command_stdout,
                        stderr=poll_result.stderr,
                        return_code=return_code,
                    )
                    await self._emit_output(exec_result)
                    return exec_result
                if stdout.strip() != self._EXEC_RUNNING:
                    raise RuntimeError(
                        "Antrieb detached exec returned an unexpected poll response"
                    )
        except BaseException:
            abort_command = (
                f"if [ -s {shlex.quote(pid_path)} ]; then "
                f"kill -- -$(cat {shlex.quote(pid_path)}) 2>/dev/null || true; fi; "
                f"rm -f {shlex.quote(stdout_path)} {shlex.quote(stderr_path)} "
                f"{shlex.quote(status_path)} {shlex.quote(status_tmp_path)} "
                f"{shlex.quote(pid_path)}; "
                f"rmdir {shlex.quote(execution_dir)} 2>/dev/null || true"
            )
            try:
                await asyncio.shield(self._exec_on_node_raw(node, abort_command))
            except BaseException:
                pass
            raise

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        assert self.definition is not None
        wrapped = command
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            wrapped = f"cd {shlex.quote(effective_cwd)} && {wrapped}"
        wrapped = f"bash -lc {shlex.quote(wrapped)}"
        effective_user = self._resolve_user(user)
        merged_env: dict[str, str] = {}
        if effective_user in (None, "root", 0, "0"):
            merged_env.update(HOME="/root", USER="root", LOGNAME="root")
        merged_env.update(self._merge_env(env) or {})
        if merged_env:
            assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in merged_env.items()
            )
            wrapped = f"env {assignments} {wrapped}"
        if effective_user in ("root", 0, "0"):
            wrapped = (
                'if [ "$(id -u)" -eq 0 ]; then '
                f"{wrapped}; "
                "elif command -v sudo >/dev/null 2>&1; then "
                f"sudo -n -- {wrapped}; "
                "else printf '%s\\n' "
                "'Harbor requested root execution, but sudo is unavailable' >&2; "
                "exit 126; fi"
            )
        elif effective_user is not None:
            wrapped = (
                f"su -s /bin/bash {shlex.quote(str(effective_user))} -c "
                f"{shlex.quote(wrapped)}"
            )
        return await self._exec_detached(
            self.definition.control_node, wrapped, timeout_sec
        )

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        data = Path(source_path).read_bytes()
        encoded = base64.b64encode(data).decode()
        temporary = f"/tmp/.harbor-upload-{hashlib.sha256(data).hexdigest()[:16]}"
        await self.exec(f": > {shlex.quote(temporary)}", user="root")
        for offset in range(0, len(encoded), self._TRANSFER_CHUNK_SIZE):
            chunk = encoded[offset : offset + self._TRANSFER_CHUNK_SIZE]
            result = await self.exec(
                f"printf %s {shlex.quote(chunk)} | base64 -d >> {shlex.quote(temporary)}",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "Antrieb upload failed")
        result = await self.exec(
            f"mkdir -p {shlex.quote(str(Path(target_path).parent))} && "
            f"mv {shlex.quote(temporary)} {shlex.quote(target_path)}",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "Antrieb upload finalization failed")

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        size_result = await self.exec(
            f"wc -c < {shlex.quote(source_path)}", user="root"
        )
        if size_result.return_code != 0:
            raise FileNotFoundError(size_result.stderr or source_path)
        size = int((size_result.stdout or "0").strip())
        data = bytearray()
        for offset in range(0, size, self._TRANSFER_CHUNK_SIZE):
            result = await self.exec(
                f"dd if={shlex.quote(source_path)} bs=1 skip={offset} "
                f"count={self._TRANSFER_CHUNK_SIZE} 2>/dev/null | base64 | tr -d '\\n'",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "Antrieb download failed")
            data.extend(base64.b64decode(result.stdout or ""))
        if len(data) != size:
            raise RuntimeError(
                f"Antrieb download size mismatch for {source_path}: "
                f"expected {size}, received {len(data)}"
            )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "upload.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(Path(source_dir), arcname=".")
            remote_archive = "/tmp/.harbor-upload.tar.gz"
            await self.upload_file(archive, remote_archive)
            result = await self.exec(
                f"mkdir -p {shlex.quote(target_dir)} && "
                f"tar -xzf {remote_archive} -C {shlex.quote(target_dir)} && "
                f"rm -f {remote_archive}",
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "Antrieb directory upload failed")

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        remote_archive = "/tmp/.harbor-download.tar.gz"
        result = await self.exec(
            f"tar -czf {remote_archive} -C {shlex.quote(source_dir)} .", user="root"
        )
        if result.return_code != 0:
            raise FileNotFoundError(result.stderr or source_dir)
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "download.tar.gz"
            await self.download_file(remote_archive, archive)
            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target, filter="data")
        await self.exec(f"rm -f {remote_archive}", user="root")

    async def initialize(self) -> None:
        """Run trusted environment initialization before task preparation."""
        assert self.definition is not None
        self.initialization_results = await run_initializers(
            self, self.definition.initialize
        )
        if self.initialization_results:
            self._save_initialization_report()

    async def prepare(self) -> None:
        """Create optional task-authored state and capture its baseline."""
        assert self.definition is not None
        if not self.definition.prepare.enabled:
            return
        if self.definition.prepare.mode == "static":
            from harbor_antrieb.static_preparer import run_static_prepare

            await run_static_prepare(self, self.definition.prepare)
        else:
            from harbor_antrieb.ai_preparer import run_ai_prepare

            await run_ai_prepare(self, self.definition.prepare)
