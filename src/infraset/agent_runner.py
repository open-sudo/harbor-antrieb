from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rewardkit.agents import get_agent
from rewardkit.models import AgentJudge, MCPServerConfig

from infraset.errors import ClusterExpiredError
from infraset.exec_bridge import redact_text


def _modern_rewardkit_backend(backend: Any) -> bool:
    return not hasattr(backend, "ensure_installed") and hasattr(backend, "__aenter__")


async def _start_backend(backend: Any) -> None:
    if _modern_rewardkit_backend(backend):
        if getattr(backend, "name", None) == "codex":
            if shutil.which("codex") is None:
                raise FileNotFoundError("Agent CLI 'codex' is not available on PATH")
            return
        await backend.__aenter__()
    else:
        await asyncio.to_thread(backend.ensure_installed)


async def _stop_backend(backend: Any) -> None:
    if _modern_rewardkit_backend(backend):
        if getattr(backend, "name", None) == "codex":
            return
        await backend.__aexit__(None, None, None)
    else:
        backend.cleanup()


def _build_backend_command(
    backend: Any,
    agent_name: str,
    prompt: str,
    schema: dict[str, Any],
    allowed_tools: tuple[str, ...],
    workspace: Path,
) -> tuple[list[str], Path | None, bytes | None]:
    if not _modern_rewardkit_backend(backend):
        return (
            backend.build_command(prompt, schema, allowed_tools=allowed_tools),
            None,
            None,
        )
    if agent_name == "codex":
        schema_path = workspace / ".infraset-output-schema.json"
        schema_path.write_text(json.dumps(schema))
        return (
            [
                shutil.which("codex") or "codex",
                "exec",
                "-",
                "--output-schema",
                str(schema_path),
                "--skip-git-repo-check",
            ],
            None,
            prompt.encode(),
        )
    return backend._build_command(prompt, schema), None, None


def _attach_process_output(
    error: BaseException, stdout: bytes, stderr: bytes
) -> BaseException:
    error.add_note("The terminated agent's partial output was saved in its workspace")
    setattr(error, "infraset_stdout", stdout)
    setattr(error, "infraset_stderr", stderr)
    return error


def _write_process_logs(
    workspace: Path, agent_name: str, stdout: bytes, stderr: bytes
) -> tuple[str, str]:
    raw_stdout = stdout.decode(errors="replace")
    raw_stderr = stderr.decode(errors="replace")
    (workspace / f"{agent_name}-stdout.log").write_text(redact_text(raw_stdout))
    (workspace / f"{agent_name}-stderr.log").write_text(redact_text(raw_stderr))
    return raw_stdout, raw_stderr


def append_reasoning_effort(
    agent_name: str, command: list[str], reasoning_effort: str | None
) -> None:
    if reasoning_effort is None:
        return
    if agent_name == "claude-code":
        command.extend(["--effort", reasoning_effort])
    elif agent_name == "codex":
        command.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])


def append_service_tier(
    agent_name: str, command: list[str], service_tier: str | None
) -> None:
    """Apply an optional backend service tier without constraining model names."""
    if service_tier is None:
        return
    if not service_tier.strip():
        raise ValueError("service_tier must not be empty")
    if agent_name != "codex":
        raise ValueError("service_tier is currently supported only by Codex")
    command.extend(["-c", f"service_tier={json.dumps(service_tier)}"])


def append_audit_argument(bridge_args: list[str], audit_path: Path | None) -> None:
    """Pass a path that remains valid when the agent runs in its workspace."""
    if audit_path is not None:
        bridge_args.extend(["--audit-log", str(audit_path.resolve())])


def _fatal_message(fatal_path: Path) -> str | None:
    if not fatal_path.is_file():
        return None
    try:
        payload = json.loads(fatal_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "The Antrieb managed cluster lease expired"
    message = payload.get("message") if isinstance(payload, dict) else None
    return (
        message
        if isinstance(message, str) and message
        else "The Antrieb managed cluster lease expired"
    )


async def _wait_for_cluster_expiration(
    fatal_path: Path,
    lease_expires_at: datetime | None,
) -> str:
    while True:
        message = _fatal_message(fatal_path)
        if message is not None:
            return message
        if lease_expires_at is not None and datetime.now(UTC) >= lease_expires_at:
            return (
                "The Antrieb managed cluster lease expired at "
                f"{lease_expires_at.isoformat()}"
            )
        await asyncio.sleep(0.1)


async def _wait_for_evidence_completion(path: Path) -> None:
    while not path.is_file():
        await asyncio.sleep(0.1)


async def _communicate_with_cluster_guard(
    process: asyncio.subprocess.Process,
    *,
    input_data: bytes | None = None,
    timeout_sec: int,
    fatal_path: Path,
    lease_expires_at: datetime | None,
    agent_name: str,
    evidence_complete_path: Path | None = None,
    allow_offline_finalization: bool = False,
    offline_finalization_grace_sec: int = 120,
) -> tuple[bytes, bytes, str | None]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    communicate = asyncio.create_task(process.communicate(input_data))
    expiration = asyncio.create_task(
        _wait_for_cluster_expiration(fatal_path, lease_expires_at)
    )
    evidence_complete = (
        asyncio.create_task(_wait_for_evidence_completion(evidence_complete_path))
        if evidence_complete_path is not None
        else None
    )
    try:
        watched: set[asyncio.Task[Any]] = {communicate, expiration}
        if evidence_complete is not None:
            watched.add(evidence_complete)
        done, _ = await asyncio.wait(
            watched,
            timeout=timeout_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            process.kill()
            stdout, stderr = await communicate
            raise _attach_process_output(
                TimeoutError(f"{agent_name} timed out after {timeout_sec} seconds"),
                stdout,
                stderr,
            )
        if expiration in done:
            message = expiration.result()
            if not allow_offline_finalization:
                if process.returncode is None:
                    process.kill()
                stdout, stderr = await communicate
                raise _attach_process_output(
                    ClusterExpiredError(message), stdout, stderr
                )
            remaining = max(0.0, deadline - loop.time())
            grace = min(remaining, float(offline_finalization_grace_sec))
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communicate), timeout=grace
                )
            except TimeoutError:
                if process.returncode is None:
                    process.kill()
                stdout, stderr = await communicate
                # The CLI may already have emitted a complete structured report
                # before becoming stuck in shutdown. Return its captured streams so
                # the caller can validate and recover that report.
            return stdout, stderr, message
        if evidence_complete is not None and evidence_complete in done:
            expiration.cancel()
            with suppress(asyncio.CancelledError):
                await expiration
            remaining = max(0.0, deadline - loop.time())
            grace = min(remaining, float(offline_finalization_grace_sec))
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communicate), timeout=grace
                )
            except TimeoutError:
                if process.returncode is None:
                    process.kill()
                stdout, stderr = await communicate
                # Live evidence is already closed. Preserve the captured streams so
                # the caller can recover a complete schema-valid report emitted by a
                # CLI that became stuck during shutdown.
            return stdout, stderr, None
        stdout, stderr = communicate.result()
        message = _fatal_message(fatal_path)
        if message is not None:
            if allow_offline_finalization:
                return stdout, stderr, message
            raise _attach_process_output(ClusterExpiredError(message), stdout, stderr)
        return stdout, stderr, None
    finally:
        expiration.cancel()
        with suppress(asyncio.CancelledError):
            await expiration
        if evidence_complete is not None:
            evidence_complete.cancel()
            with suppress(asyncio.CancelledError):
                await evidence_complete


def _audit_contains_cluster_expiration(audit_path: Path | None) -> bool:
    if audit_path is None or not audit_path.is_file():
        return False
    for line in audit_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("outcome") == "cluster_expired":
            return True
    return False


def _parse_structured_output(
    backend: Any,
    raw_stdout: str,
    output_validator: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    parsed_output = (
        backend.parse_output(raw_stdout)
        if hasattr(backend, "parse_output")
        else raw_stdout
    )
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(parsed_output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(parsed_output, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)

    validation_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            if output_validator is not None:
                output_validator(candidate)
        except (RuntimeError, ValueError) as exc:
            validation_error = exc
            continue
        return candidate

    if validation_error is not None:
        raise validation_error
    raise RuntimeError("Evaluation agent returned no complete JSON object")


def configure_codex_mcp(command: list[str], server: MCPServerConfig) -> None:
    """Inject the restricted bridge without changing the user's Codex config."""
    encoded_args = json.dumps(list(server.args), separators=(",", ":"))
    command.extend(
        [
            "-c",
            'approval_policy="never"',
            "-c",
            f'mcp_servers.{server.name}.default_tools_approval_mode="approve"',
            "-c",
            f"mcp_servers.{server.name}.command={json.dumps(server.command)}",
            "-c",
            f"mcp_servers.{server.name}.args={encoded_args}",
            "-c",
            f'mcp_servers.{server.name}.env_vars=["ANTRIEB_TOKEN"]',
        ]
    )


def configure_claude_mcp(
    command: list[str], server: MCPServerConfig, workspace: Path
) -> Path:
    """Give Claude one invocation-scoped MCP server without user config changes."""
    fd, config_name = tempfile.mkstemp(
        prefix="infraset-mcp-", suffix=".json", dir=workspace
    )
    config_path = Path(config_name)
    with os.fdopen(fd, "w") as config_file:
        json.dump(
            {
                "mcpServers": {
                    server.name: {
                        "type": "stdio",
                        "command": server.command,
                        "args": list(server.args),
                    }
                }
            },
            config_file,
        )
    command.extend(["--mcp-config", str(config_path), "--strict-mcp-config"])
    return config_path


def configure_log_only_agent(
    agent_name: str, command: list[str], workspace: Path
) -> Path | None:
    """Isolate a postmortem invocation from user-configured MCP servers."""
    if agent_name == "codex":
        command.extend(
            [
                "--ignore-user-config",
                "-c",
                'approval_policy="never"',
                "--sandbox",
                "read-only",
                "--ephemeral",
            ]
        )
        return None
    if agent_name == "claude-code":
        fd, config_name = tempfile.mkstemp(
            prefix="infraset-empty-mcp-", suffix=".json", dir=workspace
        )
        config_path = Path(config_name)
        with os.fdopen(fd, "w") as config_file:
            json.dump({"mcpServers": {}}, config_file)
        command.extend(
            [
                "--mcp-config",
                str(config_path),
                "--strict-mcp-config",
                "--tools",
                "",
                "--no-session-persistence",
            ]
        )
        return config_path
    raise ValueError(f"Unsupported host-side agent: {agent_name}")


_HOST_SECRET_ENV_PREFIXES = ("INFRASET_INITIALIZE_",)


def agent_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Keep harness-owned initialization secrets out of agent subprocesses."""
    environment = dict(os.environ if source is None else source)
    for name in tuple(environment):
        if name.startswith(_HOST_SECRET_ENV_PREFIXES):
            environment.pop(name)
    return environment


def log_agent_environment() -> dict[str, str]:
    """Remove managed-cluster authority from a postmortem subprocess."""
    environment = agent_environment()
    for name in ("ANTRIEB_TOKEN", "ANTRIEB_MCP_URL", "ANTRIEB_SESSION_ID"):
        environment.pop(name, None)
    return environment


async def run_structured_agent(
    *,
    agent_name: str,
    model: str | None,
    prompt: str,
    schema: dict[str, Any],
    session_id: str,
    nodes: tuple[str, ...],
    endpoint: str,
    workspace: Path,
    timeout_sec: int,
    audit_path: Path | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    lease_expires_at: datetime | None = None,
    allow_offline_finalization: bool = False,
    output_validator: Callable[[dict[str, Any]], None] | None = None,
    max_exec_calls: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Run a host-side agent with access only to managed-cluster exec."""
    if max_exec_calls is not None and max_exec_calls < 1:
        raise ValueError("max_exec_calls must be positive")
    backend = get_agent(AgentJudge(agent=agent_name, model=model), str(workspace))
    server_name = "infraset"
    bridge_args = [
        "-m",
        "infraset.exec_bridge",
        "--session-id",
        session_id,
        "--endpoint",
        endpoint,
    ]
    for node in nodes:
        bridge_args.extend(["--node", node])
    if max_exec_calls is not None:
        bridge_args.extend(["--max-exec-calls", str(max_exec_calls)])
    append_audit_argument(bridge_args, audit_path)
    fatal_path = (workspace / f".{agent_name}-cluster-fatal.json").resolve()
    fatal_path.unlink(missing_ok=True)
    bridge_args.extend(["--fatal-log", str(fatal_path)])
    evidence_complete_path: Path | None = None
    allowed_tools = ["exec"]
    if allow_offline_finalization:
        evidence_complete_path = (
            workspace / "evidence-collection-complete.json"
        ).resolve()
        evidence_complete_path.unlink(missing_ok=True)
        bridge_args.extend(["--evidence-complete-log", str(evidence_complete_path)])
        allowed_tools.append("complete_evidence")
    server = MCPServerConfig(
        name=server_name,
        transport="stdio",
        command=sys.executable,
        args=tuple(bridge_args),
        allowed_tools=tuple(allowed_tools),
    )
    agent_mcp_config: Path | None = None
    try:
        await _start_backend(backend)
        allowed_tool_names = server.allowed_tool_names()
        if allow_offline_finalization:
            scoped_prompt = (
                f"Use only the {allowed_tool_names[0]} and "
                f"{allowed_tool_names[1]} tools. The exec tool's node and command "
                "arguments are the complete managed-cluster control surface; the "
                "Harbor session is injected by the bridge, so do not supply or "
                "discover a session ID. After all live checks, recovery actions, "
                f"and cleanup are complete, call {allowed_tool_names[1]} exactly "
                "once. That permanently closes exec; then compose and return the "
                "final report entirely offline. If exec reports that the managed "
                "cluster lease expired, do not call exec again: classify remaining "
                "live checks as indeterminate, call complete_evidence exactly once, "
                "and compose the report from evidence already collected. Do not use "
                "any other MCP server or built-in tool.\n\n"
                f"{prompt}"
            )
        else:
            scoped_prompt = (
                f"Use only the {allowed_tool_names[0]} tool. Its node and command "
                "arguments are the complete managed-cluster control surface; the "
                "Harbor session is injected by the bridge, so do not supply or "
                "discover a session ID. Do not use any other MCP server or built-in "
                f"tool.\n\n{prompt}"
            )
        command, structured_output_path, prompt_input = _build_backend_command(
            backend,
            agent_name,
            scoped_prompt,
            schema,
            allowed_tools=allowed_tool_names,
            workspace=workspace,
        )
        if model and not _modern_rewardkit_backend(backend):
            command.extend(backend.model_args(model))
        append_reasoning_effort(agent_name, command, reasoning_effort)
        append_service_tier(agent_name, command, service_tier)
        if agent_name == "codex":
            configure_codex_mcp(command, server)
        elif agent_name == "claude-code":
            agent_mcp_config = configure_claude_mcp(command, server, workspace)
        else:
            raise ValueError(f"Unsupported host-side agent: {agent_name}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=(
                asyncio.subprocess.PIPE
                if prompt_input is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
            env=agent_environment(getattr(backend, "env", None)),
        )
        try:
            stdout, stderr, expiration_message = await _communicate_with_cluster_guard(
                process,
                input_data=prompt_input,
                timeout_sec=timeout_sec,
                fatal_path=fatal_path,
                lease_expires_at=lease_expires_at,
                agent_name=agent_name,
                evidence_complete_path=evidence_complete_path,
                allow_offline_finalization=allow_offline_finalization,
            )
        except (ClusterExpiredError, TimeoutError) as exc:
            _write_process_logs(
                workspace,
                agent_name,
                getattr(exc, "infraset_stdout", b""),
                getattr(exc, "infraset_stderr", b""),
            )
            raise
        raw_stdout, raw_stderr = _write_process_logs(
            workspace, agent_name, stdout, stderr
        )
        parse_stdout = raw_stdout
        if structured_output_path is not None and structured_output_path.is_file():
            parse_stdout = structured_output_path.read_text()
        parsed: dict[str, Any] | None = None
        parse_error: Exception | None = None
        try:
            parsed = _parse_structured_output(backend, parse_stdout, output_validator)
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            parse_error = exc
        if expiration_message is not None:
            evidence_was_closed = (
                evidence_complete_path is not None and evidence_complete_path.is_file()
            )
            attempted_after_expiration = _audit_contains_cluster_expiration(audit_path)
            if not allow_offline_finalization or parsed is None:
                error = ClusterExpiredError(expiration_message)
                if parse_error is not None:
                    error.add_note(
                        f"No complete valid offline report was available: {parse_error}"
                    )
                raise _attach_process_output(error, stdout, stderr)
            (workspace / "offline-report-recovered.json").write_text(
                json.dumps(
                    {
                        "reason": expiration_message,
                        "evidence_collection_closed": evidence_was_closed,
                        "expired_exec_observed": attempted_after_expiration,
                        "recovered_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
        evidence_was_closed = (
            evidence_complete_path is not None and evidence_complete_path.is_file()
        )
        recovered_after_shutdown_failure = (
            allow_offline_finalization
            and evidence_was_closed
            and parsed is not None
            and expiration_message is None
            and process.returncode != 0
        )
        if recovered_after_shutdown_failure:
            (workspace / "offline-report-recovered.json").write_text(
                json.dumps(
                    {
                        "reason": (
                            f"{agent_name} exited with {process.returncode} after "
                            "closing live evidence"
                        ),
                        "recovered_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
        if (
            process.returncode != 0
            and expiration_message is None
            and not recovered_after_shutdown_failure
        ):
            raise RuntimeError(
                f"{agent_name} exited with {process.returncode}: "
                f"{raw_stderr or raw_stdout[:500]}"
            )
        if parsed is None:
            if parse_error is not None:
                raise parse_error
            raise RuntimeError("Evaluation agent returned no structured output")
        return parsed, raw_stdout
    finally:
        await _stop_backend(backend)
        if agent_mcp_config is not None:
            agent_mcp_config.unlink(missing_ok=True)
        fatal_path.unlink(missing_ok=True)


async def run_structured_log_agent(
    *,
    agent_name: str,
    model: str | None,
    prompt: str,
    schema: dict[str, Any],
    workspace: Path,
    timeout_sec: int,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Run a structured diagnostic agent with no managed-cluster MCP access."""
    backend = get_agent(AgentJudge(agent=agent_name, model=model), str(workspace))
    isolation_config: Path | None = None
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        await _start_backend(backend)
        command, structured_output_path, prompt_input = _build_backend_command(
            backend,
            agent_name,
            prompt,
            schema,
            allowed_tools=(),
            workspace=workspace,
        )
        if model and not _modern_rewardkit_backend(backend):
            command.extend(backend.model_args(model))
        append_reasoning_effort(agent_name, command, reasoning_effort)
        isolation_config = configure_log_only_agent(agent_name, command, workspace)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=(
                asyncio.subprocess.PIPE
                if prompt_input is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
            env=log_agent_environment(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt_input),
                timeout=timeout_sec,
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            _write_process_logs(workspace, agent_name, stdout, stderr)
            raise TimeoutError(
                f"{agent_name} log evaluator timed out after {timeout_sec} seconds"
            ) from None
        raw_stdout, raw_stderr = _write_process_logs(
            workspace, agent_name, stdout, stderr
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{agent_name} log evaluator exited with {process.returncode}: "
                f"{raw_stderr or raw_stdout[:500]}"
            )
        parse_stdout = raw_stdout
        if structured_output_path is not None and structured_output_path.is_file():
            parse_stdout = structured_output_path.read_text()
        parsed_output = (
            backend.parse_output(parse_stdout)
            if hasattr(backend, "parse_output")
            else parse_stdout
        )
        parsed = json.loads(parsed_output)
        if not isinstance(parsed, dict):
            raise RuntimeError("Log evaluator returned a non-object JSON value")
        return parsed, raw_stdout
    finally:
        await _stop_backend(backend)
        if isolation_config is not None:
            isolation_config.unlink(missing_ok=True)
