from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from infraset.client import AntriebClient
from infraset.errors import ClusterExpiredError


_AUDIT_TEXT_LIMIT = 16 * 1024
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b[A-Za-z0-9_-]*(?:password|passwd|token|secret|api[_-]?key|"
    r"authorization|wsrep_sst_auth|authtok|bindpw)[A-Za-z0-9_-]*\b"
    r"\s*(?:=|:)\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_SQL_SECRET = re.compile(
    r"(?i)(\bIDENTIFIED\s+(?:BY|WITH\s+\S+\s+AS)\s+)"
    r"(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_URL_SECRET = re.compile(r"(https?://[^\s:/]+:)[^\s/@]+(@)")
_KNOWN_TOKEN = re.compile(r"\b(?:ant|sk)-?[A-Za-z0-9_-]{16,}\b")
_ENCODED_BLOB = re.compile(r"\b[A-Fa-f0-9]{96,}\b|\b[A-Za-z0-9+/=_-]{160,}\b")


def redact_text(value: str, *, limit: int = _AUDIT_TEXT_LIMIT) -> str:
    """Return a bounded, best-effort credential-safe diagnostic string."""
    redacted = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", value)
    redacted = _SQL_SECRET.sub(r"\1[REDACTED]", redacted)
    redacted = _BEARER_SECRET.sub(r"\1[REDACTED]", redacted)
    redacted = _URL_SECRET.sub(r"\1[REDACTED]\2", redacted)
    redacted = _KNOWN_TOKEN.sub("[REDACTED]", redacted)
    redacted = _ENCODED_BLOB.sub("[REDACTED_ENCODED_BLOB]", redacted)
    if len(redacted) <= limit:
        return redacted
    half = max(1, limit // 2)
    omitted = len(redacted) - (half * 2)
    return (
        f"{redacted[:half]}\n...[TRUNCATED {omitted} CHARACTERS]...\n{redacted[-half:]}"
    )


def redact_data(value: Any) -> Any:
    """Recursively redact strings and values stored under secret-shaped keys."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_data(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(
                r"(?i)(?:password|passwd|token|secret|api[_-]?key|authorization|"
                r"authtok|bindpw)",
                str(key),
            ):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_data(item)
        return redacted
    return value


def _exec_result_details(result: dict[str, Any]) -> dict[str, Any]:
    """Extract diagnostic fields from an unmodified MCP exec result."""
    content = result.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return {"result_available": False}
    text = content[0].get("text")
    if not isinstance(text, str):
        return {"result_available": False}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"result_available": True, "stdout": redact_text(text)}
    if not isinstance(payload, dict):
        return {"result_available": True, "stdout": redact_text(text)}
    return {
        "result_available": True,
        "return_code": payload.get("exit_code", payload.get("return_code")),
        "stdout": redact_text(str(payload.get("stdout") or "")),
        "stderr": redact_text(str(payload.get("stderr") or "")),
    }


class RawToolClient(Protocol):
    async def call_tool_raw(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


EXEC_TOOL: dict[str, Any] = {
    "name": "exec",
    "description": (
        "Execute one command on one node of the Harbor-managed Antrieb cluster. "
        "The cluster lifecycle is controlled by Harbor and cannot be changed here."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
                "description": "Managed node name, such as node1.",
            },
            "command": {"type": "string", "description": "Literal shell command."},
            "secret_env": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["node", "command"],
        "additionalProperties": False,
    },
}

COMPLETE_EVIDENCE_TOOL: dict[str, Any] = {
    "name": "complete_evidence",
    "description": (
        "Permanently close live-cluster evidence collection after all checks, "
        "recovery actions, and cleanup are complete. After this call, exec is "
        "unavailable and the evaluator must compose its final report offline."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--session-id")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--audit-log", type=Path)
    parser.add_argument("--fatal-log", type=Path)
    parser.add_argument("--evidence-complete-log", type=Path)
    parser.add_argument("--max-exec-calls", type=int)
    parser.add_argument("--node", action="append", default=[])
    return parser


def _load_descriptor(path: Path | None) -> dict[str, Any]:
    descriptor_path = path or Path("/run/infraset/session.json")
    try:
        loaded = json.loads(descriptor_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_token(token_file: Path | None) -> str | None:
    token = os.environ.get("ANTRIEB_TOKEN")
    if token:
        return token
    try:
        return (token_file or Path("/etc/infraset/token")).read_text().strip()
    except OSError:
        return None


class ExecBridge:
    def __init__(
        self,
        *,
        client: RawToolClient,
        session_id: str,
        nodes: tuple[str, ...],
        audit_log: Path | None,
        fatal_log: Path | None = None,
        evidence_complete_log: Path | None = None,
        max_exec_calls: int | None = None,
    ) -> None:
        if max_exec_calls is not None and max_exec_calls < 1:
            raise ValueError("max_exec_calls must be positive")
        self.client = client
        self.session_id = session_id
        self.nodes = nodes
        self.audit_log = audit_log
        self.fatal_log = fatal_log
        self.evidence_complete_log = evidence_complete_log
        self.max_exec_calls = max_exec_calls
        self.exec_calls = 0

    def _assert_node(self, node: str) -> None:
        if self.nodes and node not in self.nodes:
            raise ValueError(f"Unknown managed node: {node}")

    def _audit(
        self,
        node: str,
        command: str,
        outcome: str,
        **details: Any,
    ) -> None:
        if self.audit_log is None:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "node": node,
            "command": command,
            "outcome": outcome,
            **details,
        }
        entry = redact_data(entry)
        with self.audit_log.open("a") as audit_file:
            audit_file.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def _fatal(self, error: ClusterExpiredError) -> None:
        if self.fatal_log is None:
            return
        self.fatal_log.write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": redact_text(str(error)),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        )

    def _evidence_is_complete(self) -> bool:
        return (
            self.evidence_complete_log is not None
            and self.evidence_complete_log.is_file()
        )

    def _complete_evidence(self) -> None:
        if self.evidence_complete_log is None:
            raise ValueError("complete_evidence is not available for this invocation")
        if not self._evidence_is_complete():
            self.evidence_complete_log.write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": "complete",
                    }
                )
            )

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        if request_id is None:
            return None
        method = message.get("method")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "infraset", "version": "0.1.0"},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            tools: list[dict[str, Any]] = [EXEC_TOOL]
            if self.evidence_complete_log is not None:
                tools.append(COMPLETE_EVIDENCE_TOOL)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools},
            }
        if method != "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        params = message.get("params")
        if not isinstance(params, dict):
            raise ValueError("tool call parameters must be an object")
        tool_name = params.get("name")
        if tool_name == "complete_evidence":
            arguments = params.get("arguments")
            if arguments not in ({}, None):
                raise ValueError("complete_evidence does not accept arguments")
            self._complete_evidence()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Evidence collection is closed. Compose the final "
                                "report without calling any more tools."
                            ),
                        }
                    ]
                },
            }
        if tool_name != "exec":
            available = (
                "exec and complete_evidence" if self.evidence_complete_log else "exec"
            )
            raise ValueError(f"Only {available} are available for this invocation")
        if self._evidence_is_complete():
            raise RuntimeError(
                "Live evidence collection is already complete; exec is permanently "
                "closed for this invocation"
            )
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("exec arguments must be an object")
        node = arguments.get("node")
        command = arguments.get("command")
        if not isinstance(node, str) or not isinstance(command, str):
            raise ValueError("exec requires string node and command arguments")
        secret_env = arguments.get("secret_env")
        if secret_env is not None and not isinstance(secret_env, dict):
            raise ValueError("secret_env must be an object")
        self._assert_node(node)
        if self.max_exec_calls is not None and self.exec_calls >= self.max_exec_calls:
            self._audit(
                node,
                command,
                "budget_exhausted",
                max_exec_calls=self.max_exec_calls,
            )
            raise RuntimeError(
                f"exec command budget exhausted ({self.max_exec_calls} calls)"
            )
        self.exec_calls += 1
        self._audit(node, command, "requested")
        started_at = time.monotonic()
        try:
            result = await self.client.call_tool_raw(
                "exec",
                {
                    "session_id": self.session_id,
                    "node": node,
                    "command": command,
                    **({"secret_env": secret_env} if secret_env is not None else {}),
                },
            )
        except ClusterExpiredError as exc:
            self._audit(
                node,
                command,
                "cluster_expired",
                error=str(exc),
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            self._fatal(exc)
            raise
        except Exception as exc:
            self._audit(
                node,
                command,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            raise
        self._audit(
            node,
            command,
            "completed",
            duration_ms=round((time.monotonic() - started_at) * 1000),
            **_exec_result_details(result),
        )
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


async def _serve(bridge: ExecBridge) -> None:
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            message: dict[str, Any] | None = None
            try:
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError("MCP message must be an object")
                message = loaded
                response = await bridge.handle(message)
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id") if message is not None else None,
                    "error": {"code": -32000, "message": str(exc)},
                }
            if response is not None:
                sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                sys.stdout.flush()
    finally:
        await bridge.client.close()


def main() -> None:
    args = _argument_parser().parse_args()
    descriptor = _load_descriptor(args.descriptor)
    endpoint = (
        args.endpoint
        or os.environ.get("ANTRIEB_MCP_URL")
        or descriptor.get("endpoint")
        or "https://antrieb.sh/mcp"
    )
    session_id = (
        args.session_id
        or os.environ.get("ANTRIEB_SESSION_ID")
        or descriptor.get("session_id")
    )
    token = _load_token(args.token_file)
    if not isinstance(session_id, str) or not token:
        raise SystemExit(
            "infraset bridge requires ANTRIEB_TOKEN and a managed session descriptor"
        )
    descriptor_nodes = descriptor.get("nodes", [])
    node_values = args.node
    if not node_values and isinstance(descriptor_nodes, list):
        node_values = descriptor_nodes
    bridge = ExecBridge(
        client=AntriebClient(str(endpoint), token),
        session_id=session_id,
        nodes=tuple(str(node) for node in node_values),
        audit_log=args.audit_log,
        fatal_log=args.fatal_log,
        evidence_complete_log=args.evidence_complete_log,
        max_exec_calls=args.max_exec_calls,
    )
    asyncio.run(_serve(bridge))


if __name__ == "__main__":
    main()
