from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor_antrieb.exec_bridge import redact_data, redact_text
from harbor_antrieb.observations import (
    collect_global_observations,
    compare_global_observations,
)

_PHASES = ("before_prepare", "after_prepare", "after_executor")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact_data(value), indent=2) + "\n")
    temporary.replace(path)


def _unavailable_snapshot(phase: str, limitation: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "captured_at": None,
        "nodes": [],
        "limitations": [limitation],
    }


class AntriebCollector:
    """Capture provider-owned lifecycle observations for one Harbor trial.

    The collector is deliberately independent of task-authored verification. It
    records the same bounded observations at stable lifecycle boundaries so the
    resulting artifacts can be evaluated offline and published with the job.
    """

    def __init__(self, trial_dir: Path | str) -> None:
        self.trial_dir = Path(trial_dir)
        self.root = self.trial_dir / "collector"
        self._active_attempt: int | None = None

    def _attempt_dir(self, attempt: int) -> Path:
        return self.root / "attempts" / f"{attempt:02d}"

    def _manifest_path(self, attempt: int) -> Path:
        return self._attempt_dir(attempt) / "manifest.json"

    def _snapshot_path(self, attempt: int, phase: str) -> Path:
        return self._attempt_dir(attempt) / "snapshots" / (
            phase.replace("_", "-") + ".json"
        )

    def _audit_path(self, attempt: int) -> Path:
        return self._attempt_dir(attempt) / "commands.jsonl"

    def _manifest(self, attempt: int) -> dict[str, Any]:
        loaded = _read_json(self._manifest_path(attempt), {})
        if isinstance(loaded, dict):
            return loaded
        return {}

    def _save_manifest(self, attempt: int, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now()
        _write_json(self._manifest_path(attempt), manifest)

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.trial_dir))

    async def begin_attempt(
        self,
        environment: Any,
        *,
        attempt: int,
        prepare_enabled: bool,
    ) -> dict[str, Any]:
        """Start a retry-scoped collection and capture the pre-prepare state."""

        self._active_attempt = attempt
        attempt_dir = self._attempt_dir(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        self._audit_path(attempt).write_text("")
        manifest = {
            "schema_version": 1,
            "attempt": attempt,
            "cluster_number": attempt,
            "prepare_enabled": prepare_enabled,
            "created_at": _utc_now(),
            "phases": {},
        }
        self._save_manifest(attempt, manifest)
        return await self._capture(
            environment,
            attempt=attempt,
            phase="before_prepare",
            lifecycle_outcome="ready_for_prepare",
        )

    async def finish_prepare(
        self,
        environment: Any,
        *,
        prepare_enabled: bool,
        outcome: str,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Capture prepared state, or alias it when no preparer was configured."""

        attempt = self._require_active_attempt(environment)
        if not prepare_enabled:
            manifest = self._manifest(attempt)
            phases = manifest.setdefault("phases", {})
            phases["after_prepare"] = {
                "status": "alias",
                "same_as": "before_prepare",
                "recorded_at": _utc_now(),
                "lifecycle_outcome": "not_configured",
            }
            self._save_manifest(attempt, manifest)
            snapshot = self._load_phase(attempt, "after_prepare")
        else:
            snapshot = await self._capture(
                environment,
                attempt=attempt,
                phase="after_prepare",
                lifecycle_outcome=outcome,
                error=error,
            )
        _write_json(self.trial_dir / "environment-baseline.json", snapshot)
        return snapshot

    async def finish_executor(
        self,
        environment: Any,
        *,
        outcome: str,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Capture final executor state before retry, teardown, or verification."""

        attempt = self._require_active_attempt(environment)
        return await self._capture(
            environment,
            attempt=attempt,
            phase="after_executor",
            lifecycle_outcome=outcome,
            error=error,
        )

    def _require_active_attempt(self, environment: Any) -> int:
        if self._active_attempt is not None:
            return self._active_attempt
        attempt = int(getattr(environment, "clusters_provisioned", 0))
        if attempt < 1:
            raise RuntimeError("collector has no active managed-cluster attempt")
        self._active_attempt = attempt
        return attempt

    async def _capture(
        self,
        environment: Any,
        *,
        attempt: int,
        phase: str,
        lifecycle_outcome: str,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        if phase not in _PHASES:
            raise ValueError(f"unknown collector phase: {phase}")
        path = self._snapshot_path(attempt, phase)
        collection_error: str | None = None
        try:
            snapshot = await collect_global_observations(
                environment,
                phase=phase,
                output_path=path,
                audit_path=self._audit_path(attempt),
            )
            status = "captured"
        except Exception as exc:  # noqa: BLE001 - collection must not mask task work
            collection_error = redact_text(f"{type(exc).__name__}: {exc}")
            snapshot = _unavailable_snapshot(
                phase, f"Lifecycle observation collection failed: {collection_error}"
            )
            _write_json(path, snapshot)
            status = "unavailable"

        manifest = self._manifest(attempt)
        phases = manifest.setdefault("phases", {})
        record: dict[str, Any] = {
            "status": status,
            "path": self._relative(path),
            "recorded_at": _utc_now(),
            "captured_at": snapshot.get("captured_at"),
            "lifecycle_outcome": lifecycle_outcome,
        }
        if error is not None:
            record["lifecycle_error"] = redact_text(
                f"{type(error).__name__}: {error}"
            )
        if collection_error is not None:
            record["collection_error"] = collection_error
        phases[phase] = record
        self._save_manifest(attempt, manifest)
        return snapshot

    def _latest_attempt(self) -> int | None:
        if self._active_attempt is not None:
            return self._active_attempt
        attempts_dir = self.root / "attempts"
        if not attempts_dir.is_dir():
            return None
        attempts = sorted(
            int(path.name)
            for path in attempts_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        return attempts[-1] if attempts else None

    def _load_phase(self, attempt: int, phase: str) -> dict[str, Any]:
        manifest = self._manifest(attempt)
        phases = manifest.get("phases", {})
        record = phases.get(phase, {}) if isinstance(phases, dict) else {}
        if isinstance(record, dict) and isinstance(record.get("same_as"), str):
            return self._load_phase(attempt, record["same_as"])
        path_value = record.get("path") if isinstance(record, dict) else None
        path = (
            self.trial_dir / path_value
            if isinstance(path_value, str)
            else self._snapshot_path(attempt, phase)
        )
        loaded = _read_json(path, None)
        if isinstance(loaded, dict):
            return loaded
        return _unavailable_snapshot(
            phase, f"Collector snapshot {phase!r} was not captured."
        )

    def load_bundle(self, attempt: int | None = None) -> dict[str, Any]:
        """Load lifecycle snapshots and baseline-relative comparisons offline."""

        selected = attempt if attempt is not None else self._latest_attempt()
        if selected is None:
            return self._load_legacy_bundle()
        manifest = self._manifest(selected)
        before = self._load_phase(selected, "before_prepare")
        prepared = self._load_phase(selected, "after_prepare")
        final = self._load_phase(selected, "after_executor")
        return {
            "schema_version": 1,
            "attempt": selected,
            "manifest": manifest,
            "before_prepare": before,
            "after_prepare": prepared,
            "after_executor": final,
            "preparation_comparison": compare_global_observations(before, prepared),
            "executor_comparison": compare_global_observations(prepared, final),
        }

    def _load_legacy_bundle(self) -> dict[str, Any]:
        baseline = _read_json(self.trial_dir / "environment-baseline.json", None)
        post = _read_json(
            self.trial_dir / "verifier" / "environment-post.json", None
        )
        before = _unavailable_snapshot(
            "before_prepare",
            "This legacy job predates pre-prepare lifecycle collection.",
        )
        prepared = (
            baseline
            if isinstance(baseline, dict)
            else _unavailable_snapshot(
                "after_prepare", "The pre-executor baseline was unavailable."
            )
        )
        final = (
            post
            if isinstance(post, dict)
            else _unavailable_snapshot(
                "after_executor", "The post-executor snapshot was unavailable."
            )
        )
        return {
            "schema_version": 1,
            "attempt": None,
            "manifest": {
                "schema_version": 1,
                "legacy": True,
                "phases": {},
            },
            "before_prepare": before,
            "after_prepare": prepared,
            "after_executor": final,
            "preparation_comparison": compare_global_observations(before, prepared),
            "executor_comparison": compare_global_observations(prepared, final),
        }
