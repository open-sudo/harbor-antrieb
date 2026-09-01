import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor_antrieb.collector import AntriebCollector
from harbor_antrieb.observations import GLOBAL_OBSERVATIONS


class CollectorEnvironment:
    def __init__(self) -> None:
        self.nodes = ("node1",)
        self.node_images = {"node1": {"ani": "antrieb:ubuntu24.04:v1"}}
        self.clusters_provisioned = 1
        self.state = "initial"
        self.calls: list[tuple[str, str]] = []

    async def exec_on_node(
        self, node: str, command: str, *, user: str | int | None = None
    ) -> SimpleNamespace:
        assert user == "root"
        self.calls.append((self.state, command))
        return SimpleNamespace(
            return_code=0,
            stdout=f"state={self.state}\n",
            stderr="",
        )


@pytest.mark.asyncio
async def test_collector_captures_all_three_lifecycle_boundaries(
    tmp_path: Path,
) -> None:
    environment = CollectorEnvironment()
    collector = AntriebCollector(tmp_path)

    await collector.begin_attempt(
        environment, attempt=1, prepare_enabled=True
    )
    environment.state = "prepared"
    await collector.finish_prepare(
        environment, prepare_enabled=True, outcome="completed"
    )
    environment.state = "executed"
    await collector.finish_executor(environment, outcome="completed")

    attempt_dir = tmp_path / "collector" / "attempts" / "01"
    assert sorted(path.name for path in (attempt_dir / "snapshots").iterdir()) == [
        "after-executor.json",
        "after-prepare.json",
        "before-prepare.json",
    ]
    manifest = json.loads((attempt_dir / "manifest.json").read_text())
    assert manifest["prepare_enabled"] is True
    assert set(manifest["phases"]) == {
        "before_prepare",
        "after_prepare",
        "after_executor",
    }
    assert len(environment.calls) == len(GLOBAL_OBSERVATIONS) * 3

    bundle = collector.load_bundle()
    assert bundle["attempt"] == 1
    assert bundle["before_prepare"]["phase"] == "before_prepare"
    assert bundle["after_prepare"]["phase"] == "after_prepare"
    assert bundle["after_executor"]["phase"] == "after_executor"
    assert bundle["preparation_comparison"]["changes"]
    assert bundle["executor_comparison"]["changes"]
    assert json.loads((tmp_path / "environment-baseline.json").read_text())[
        "phase"
    ] == "after_prepare"


@pytest.mark.asyncio
async def test_collector_collapses_prepare_boundary_when_prepare_is_disabled(
    tmp_path: Path,
) -> None:
    environment = CollectorEnvironment()
    collector = AntriebCollector(tmp_path)

    await collector.begin_attempt(
        environment, attempt=1, prepare_enabled=False
    )
    await collector.finish_prepare(
        environment, prepare_enabled=False, outcome="not_configured"
    )
    environment.state = "executed"
    await collector.finish_executor(environment, outcome="completed")

    snapshots = tmp_path / "collector" / "attempts" / "01" / "snapshots"
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "after-executor.json",
        "before-prepare.json",
    ]
    manifest = json.loads(
        (tmp_path / "collector" / "attempts" / "01" / "manifest.json").read_text()
    )
    assert manifest["phases"]["after_prepare"]["status"] == "alias"
    assert manifest["phases"]["after_prepare"]["same_as"] == "before_prepare"
    bundle = collector.load_bundle()
    assert bundle["after_prepare"] == bundle["before_prepare"]
    assert bundle["preparation_comparison"]["changes"] == []
    assert len(environment.calls) == len(GLOBAL_OBSERVATIONS) * 2


@pytest.mark.asyncio
async def test_collector_keeps_retries_in_separate_attempt_directories(
    tmp_path: Path,
) -> None:
    environment = CollectorEnvironment()
    collector = AntriebCollector(tmp_path)

    await collector.begin_attempt(environment, attempt=1, prepare_enabled=False)
    await collector.finish_prepare(
        environment, prepare_enabled=False, outcome="not_configured"
    )
    await collector.finish_executor(environment, outcome="blocked")

    environment.clusters_provisioned = 2
    environment.state = "fresh-cluster"
    await collector.begin_attempt(environment, attempt=2, prepare_enabled=False)
    await collector.finish_prepare(
        environment, prepare_enabled=False, outcome="not_configured"
    )
    await collector.finish_executor(environment, outcome="completed")

    assert (tmp_path / "collector" / "attempts" / "01" / "manifest.json").is_file()
    assert (tmp_path / "collector" / "attempts" / "02" / "manifest.json").is_file()
    assert collector.load_bundle()["attempt"] == 2
    assert collector.load_bundle(attempt=1)["attempt"] == 1
