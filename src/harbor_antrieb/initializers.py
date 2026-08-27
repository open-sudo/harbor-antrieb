from __future__ import annotations

import asyncio
import os
import re
import shlex
import stat
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InitializationResult:
    """Sanitized result from one trusted environment initializer."""

    name: str
    scope: str
    targets: tuple[str, ...]
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["targets"] = list(self.targets)
        return value


Initializer = Callable[[Any], Awaitable[InitializationResult]]

_INITIALIZE_CREDENTIALS_FILE_ENV = "HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE"
_RHSM_USERNAME_ENV = "HARBOR_ANTRIEB_INITIALIZE_USERNAME"
_RHSM_PASSWORD_ENV = "HARBOR_ANTRIEB_INITIALIZE_PASSWORD"
_RHSM_INITIALIZE_COMMAND = """\
set -eu
report_status() {
  printf '%s\n' "HARBOR_ANTRIEB_INITIALIZE_STATUS=$1"
}
if ! command -v subscription-manager >/dev/null 2>&1; then
  report_status subscription-manager-missing
  exit 127
fi
install -d -m 0750 /var/lib/rhsm/cache
if subscription-manager identity >/dev/null 2>&1; then
  report_status already-registered
  exit 0
fi

network_ready=false
attempt=0
while [ "$attempt" -lt 30 ]; do
  if getent hosts subscription.rhsm.redhat.com >/dev/null 2>&1; then
    network_ready=true
    break
  fi
  attempt=$((attempt + 1))
  sleep 2
done
if [ "$network_ready" != true ]; then
  report_status provider-dns-unavailable
  exit 69
fi

subscription-manager clean >/dev/null 2>&1 || true
install -d -m 0750 /var/lib/rhsm/cache
set +e
timeout 120 subscription-manager register \
  --username "$HARBOR_ANTRIEB_INITIALIZE_USERNAME" \
  --password "$HARBOR_ANTRIEB_INITIALIZE_PASSWORD" \
  --force >/dev/null 2>&1
register_rc=$?
set -e
unset HARBOR_ANTRIEB_INITIALIZE_USERNAME HARBOR_ANTRIEB_INITIALIZE_PASSWORD

if ! subscription-manager identity >/dev/null 2>&1; then
  if [ "$register_rc" -eq 0 ]; then
    report_status identity-unavailable
    exit 70
  fi
  exit "$register_rc"
fi
report_status registered
"""


def _load_initialize_credentials() -> str:
    """Load the opaque credential payload supplied to an initializer."""

    configured_path = os.environ.get(_INITIALIZE_CREDENTIALS_FILE_ENV)
    if not configured_path:
        raise RuntimeError(
            "initialization requires HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE"
        )
    path = Path(configured_path)
    if not path.is_absolute():
        raise RuntimeError(
            "HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE must be an absolute path"
        )
    try:
        metadata = path.stat()
        contents = path.read_text()
    except OSError as exc:
        raise RuntimeError("Unable to read the initialize credentials file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("The initialize credentials path is not a file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(
            "The initialize credentials file must not be group/world accessible"
        )
    if not contents or "\0" in contents:
        raise RuntimeError(
            "The initialize credentials file must contain a nonempty text payload"
        )
    return contents


def _rhel_nodes(environment: Any) -> tuple[str, ...]:
    node_images = getattr(environment, "node_images", None)
    if not isinstance(node_images, dict):
        raise RuntimeError("rhsm initialization requires provisioned node_images")
    nodes = tuple(
        node
        for node, image in node_images.items()
        if isinstance(node, str)
        and isinstance(image, dict)
        and isinstance(image.get("ani"), str)
        and re.fullmatch(r"[^:]+:rhel(?:[0-9]+(?:\.[0-9]+)*)?:[^:]+", image["ani"])
    )
    if not nodes:
        raise RuntimeError("rhsm initialization requires at least one RHEL node")
    return nodes


def _command_for_image_interface(
    environment: Any,
    node: str,
    command: str,
    secret_env_names: Sequence[str] = (),
) -> str:
    image = environment.node_images[node]
    image_interface = image.get("image_interface")
    if image_interface is None:
        return command
    if not isinstance(image_interface, dict):
        raise RuntimeError(f"node {node} returned a malformed image_interface")
    execution = image_interface.get("execution")
    if execution is None:
        return command
    if not isinstance(execution, dict):
        raise RuntimeError(f"node {node} returned a malformed execution interface")
    privilege_escalation = execution.get("privilegeEscalation")
    if privilege_escalation is None:
        return command
    if privilege_escalation != "sudo-noninteractive":
        raise RuntimeError(
            f"node {node} returned unsupported privilege escalation "
            f"{privilege_escalation!r}"
        )
    invalid_names = [
        name
        for name in secret_env_names
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
    ]
    if invalid_names:
        raise RuntimeError(f"invalid secret environment names: {invalid_names}")
    forwarded = " ".join(f'{name}="${name}"' for name in secret_env_names)
    env_command = f"env {forwarded} " if forwarded else ""
    return f"sudo -n {env_command}/bin/sh -c {shlex.quote(command)}"


async def _initialize_rhsm(environment: Any) -> InitializationResult:
    lines = _load_initialize_credentials().splitlines()
    if len(lines) != 2 or not all(lines):
        raise RuntimeError(
            "rhsm credentials must contain username on line 1 and password on line 2"
        )
    username, password = lines
    nodes = _rhel_nodes(environment)

    async def initialize_node(node: str) -> tuple[bool, str]:
        command = _command_for_image_interface(
            environment,
            node,
            _RHSM_INITIALIZE_COMMAND,
            (_RHSM_USERNAME_ENV, _RHSM_PASSWORD_ENV),
        )
        result = await environment._exec_on_node_raw(
            node,
            command,
            {
                _RHSM_USERNAME_ENV: username,
                _RHSM_PASSWORD_ENV: password,
            },
        )
        status = next(
            (
                line.removeprefix("HARBOR_ANTRIEB_INITIALIZE_STATUS=")
                for line in reversed((result.stdout or "").splitlines())
                if line.startswith("HARBOR_ANTRIEB_INITIALIZE_STATUS=")
            ),
            f"exit-{result.return_code}",
        )
        return result.return_code == 0, status

    tasks: dict[str, asyncio.Task[tuple[bool, str]]] = {}
    async with asyncio.TaskGroup() as group:
        for node in nodes:
            tasks[node] = group.create_task(initialize_node(node))
    failed = {
        node: status
        for node, task in tasks.items()
        for succeeded, status in [task.result()]
        if not succeeded
    }
    if failed:
        details = ", ".join(f"{node}={status}" for node, status in failed.items())
        raise RuntimeError(f"rhsm initialization failed: {details}")
    return InitializationResult(name="rhsm", scope="environment", targets=nodes)


_INITIALIZERS: dict[str, Initializer] = {
    "rhsm": _initialize_rhsm,
}


async def run_initializers(
    environment: Any, names: Sequence[str]
) -> tuple[InitializationResult, ...]:
    """Run ordered, trusted initializers against the complete environment."""

    unknown = [name for name in names if name not in _INITIALIZERS]
    if unknown:
        raise ValueError(
            f"Unknown Antrieb initializers {unknown}; "
            f"available: {sorted(_INITIALIZERS)}"
        )
    results: list[InitializationResult] = []
    for name in names:
        results.append(await _INITIALIZERS[name](environment))
    return tuple(results)
