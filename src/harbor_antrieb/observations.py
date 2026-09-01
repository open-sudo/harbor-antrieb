from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor_antrieb.exec_bridge import redact_data, redact_text

_UNSUPPORTED = "__HARBOR_ANTRIEB_UNSUPPORTED__"
_OUTPUT_LIMIT = 16 * 1024
_OBSERVATION_TIMEOUT_SEC = 45


@dataclass(frozen=True)
class ObservationSpec:
    """One fixed, provider-owned observation collected on every managed node."""

    id: str
    description: str
    command: str


GLOBAL_OBSERVATIONS: tuple[ObservationSpec, ...] = (
    ObservationSpec(
        id="platform",
        description="Managed node identity and operating-system release.",
        command=(
            "printf 'node=%s\\naddress=%s\\n' \"$NODE_NAME\" \"$NODE_IP\"; "
            "uname -srmo 2>/dev/null || uname -a; "
            "if [ -r /etc/os-release ]; then "
            "sed -n '1,40p' /etc/os-release; fi"
        ),
    ),
    ObservationSpec(
        id="temporary-files",
        description="Bounded inventory of entries below /tmp.",
        command=(
            "if [ -d /tmp ]; then "
            "find /tmp -mindepth 1 -maxdepth 2 -print 2>/dev/null "
            "| LC_ALL=C sort | head -n 500; "
            "else printf '%s\\n' '__HARBOR_ANTRIEB_UNSUPPORTED__: /tmp'; fi"
        ),
    ),
    ObservationSpec(
        id="failed-services",
        description="Systemd units currently in the failed state, when available.",
        command=(
            "if command -v systemctl >/dev/null 2>&1; then "
            "systemctl --failed --no-legend --plain 2>/dev/null; "
            "else printf '%s\\n' '__HARBOR_ANTRIEB_UNSUPPORTED__: systemctl'; fi"
        ),
    ),
    ObservationSpec(
        id="listening-sockets",
        description="Bounded inventory of listening network sockets.",
        command=(
            "if command -v ss >/dev/null 2>&1; then ss -lntupH 2>/dev/null; "
            "elif command -v netstat >/dev/null 2>&1; then "
            "netstat -lntup 2>/dev/null; "
            "else printf '%s\\n' '__HARBOR_ANTRIEB_UNSUPPORTED__: sockets'; fi "
            "| head -n 300"
        ),
    ),
    ObservationSpec(
        id="package-health",
        description="Native package-manager consistency check, when supported.",
        command=(
            "if command -v dpkg >/dev/null 2>&1; then "
            "printf 'manager=dpkg\\n'; dpkg --audit; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "printf 'manager=dnf\\n'; dnf -q check; "
            "elif command -v yum >/dev/null 2>&1; then "
            "printf 'manager=yum\\n'; yum -q check; "
            "elif command -v apk >/dev/null 2>&1; then "
            "printf 'manager=apk\\n'; apk audit --system 2>/dev/null; "
            "elif command -v pacman >/dev/null 2>&1; then "
            "printf 'manager=pacman\\n'; pacman -Dk; "
            "else printf '%s\\n' "
            "'__HARBOR_ANTRIEB_UNSUPPORTED__: package-manager-health'; fi"
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded(value: str | None) -> str:
    return redact_text(value or "", limit=_OUTPUT_LIMIT)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact_data(value), indent=2) + "\n")
    temporary.replace(path)


def _append_audit(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(redact_data(record), separators=(",", ":")) + "\n")


async def _collect_one(
    environment: Any,
    *,
    node: str,
    spec: ObservationSpec,
    phase: str,
    audit_path: Path | None,
) -> dict[str, Any]:
    started = time.monotonic()
    requested_at = _utc_now()
    _append_audit(
        audit_path,
        {
            "timestamp": requested_at,
            "phase": phase,
            "node": node,
            "observation": spec.id,
            "command": spec.command,
            "outcome": "requested",
        },
    )
    try:
        observe = getattr(environment, "observe_on_node", None)
        if not callable(observe):
            observe = environment.exec_on_node
        result = await asyncio.wait_for(
            observe(node, spec.command, user="root"),
            timeout=_OBSERVATION_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001 - unavailable evidence must be recorded
        duration_ms = round((time.monotonic() - started) * 1000)
        error = _bounded(f"{type(exc).__name__}: {exc}")
        record = {
            "id": f"global:{phase}:{node}:{spec.id}",
            "description": spec.description,
            "status": "unavailable",
            "return_code": None,
            "stdout": "",
            "stderr": "",
            "error": error,
            "duration_ms": duration_ms,
        }
        _append_audit(
            audit_path,
            {
                "timestamp": _utc_now(),
                "phase": phase,
                "node": node,
                "observation": spec.id,
                "command": spec.command,
                "outcome": "unavailable",
                "error": error,
                "duration_ms": duration_ms,
            },
        )
        return record

    stdout = _bounded(getattr(result, "stdout", ""))
    stderr = _bounded(getattr(result, "stderr", ""))
    return_code = getattr(result, "return_code", None)
    status = "unsupported" if _UNSUPPORTED in stdout else "observed"
    duration_ms = round((time.monotonic() - started) * 1000)
    record = {
        "id": f"global:{phase}:{node}:{spec.id}",
        "description": spec.description,
        "status": status,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
    }
    _append_audit(
        audit_path,
        {
            "timestamp": _utc_now(),
            "phase": phase,
            "node": node,
            "observation": spec.id,
            "command": spec.command,
            "outcome": status,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
        },
    )
    return record


async def collect_global_observations(
    environment: Any,
    *,
    phase: str,
    output_path: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Collect bounded cross-task observations without task-authored commands."""

    nodes = tuple(str(node) for node in getattr(environment, "nodes", ()))
    exec_on_node = getattr(environment, "exec_on_node", None)
    observe_on_node = getattr(environment, "observe_on_node", None)
    if not nodes or not (callable(exec_on_node) or callable(observe_on_node)):
        report = {
            "schema_version": 1,
            "phase": phase,
            "captured_at": _utc_now(),
            "nodes": [],
            "limitations": [
                "The managed environment did not expose nodes and exec_on_node."
            ],
        }
        _write_json(output_path, report)
        return report

    async def collect_node(node: str) -> dict[str, Any]:
        observations = list(
            await asyncio.gather(
                *(
                    _collect_one(
                        environment,
                        node=node,
                        spec=spec,
                        phase=phase,
                        audit_path=audit_path,
                    )
                    for spec in GLOBAL_OBSERVATIONS
                )
            )
        )
        image = getattr(environment, "node_images", {}).get(node, {})
        return {
            "name": node,
            "image": redact_data(image),
            "observations": observations,
        }

    collected = await asyncio.gather(*(collect_node(node) for node in nodes))
    report = {
        "schema_version": 1,
        "phase": phase,
        "captured_at": _utc_now(),
        "nodes": list(collected),
        "limitations": [],
    }
    _write_json(output_path, report)
    return report


def compare_global_observations(
    baseline: dict[str, Any], post: dict[str, Any]
) -> dict[str, Any]:
    """Return line-oriented changes while retaining unavailable observations."""

    def indexed(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        values: dict[tuple[str, str], dict[str, Any]] = {}
        for node in report.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_name = str(node.get("name", ""))
            for observation in node.get("observations", []):
                if isinstance(observation, dict):
                    evidence_id = str(observation.get("id", ""))
                    category = evidence_id.rsplit(":", 1)[-1]
                    values[(node_name, category)] = observation
        return values

    before = indexed(baseline)
    after = indexed(post)
    changes: list[dict[str, Any]] = []
    for node, observation_id in sorted(set(before) | set(after)):
        old = before.get((node, observation_id))
        new = after.get((node, observation_id))
        old_lines = set(str((old or {}).get("stdout", "")).splitlines())
        new_lines = set(str((new or {}).get("stdout", "")).splitlines())
        added = sorted(new_lines - old_lines)
        removed = sorted(old_lines - new_lines)
        old_status = (old or {}).get("status", "missing")
        new_status = (new or {}).get("status", "missing")
        old_rc = (old or {}).get("return_code")
        new_rc = (new or {}).get("return_code")
        if added or removed or old_status != new_status or old_rc != new_rc:
            changes.append(
                {
                    "node": node,
                    "observation": observation_id,
                    "baseline_status": old_status,
                    "post_status": new_status,
                    "baseline_return_code": old_rc,
                    "post_return_code": new_rc,
                    "added_lines": added[:500],
                    "removed_lines": removed[:500],
                }
            )
    return {
        "schema_version": 1,
        "baseline_captured_at": baseline.get("captured_at"),
        "post_captured_at": post.get("captured_at"),
        "changes": changes,
        "limitations": [
            *[str(item) for item in baseline.get("limitations", [])],
            *[str(item) for item in post.get("limitations", [])],
        ],
    }
