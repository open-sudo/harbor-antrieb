# Harbor Antrieb

Harbor provider for running datasets on Antrieb-managed infrastructure. It
provides a custom environment, lifecycle collector, agent, and verifier for any
Harbor dataset.

## Normal use

Users do not need to clone Harbor or this repository. A dataset runner can fetch
Harbor and this provider from GitHub:

```bash
export ANTRIEB_TOKEN='ant_...'
uv run --no-project \
  --with 'harbor-antrieb @ git+https://github.com/open-sudo/harbor-antrieb.git' \
  harbor run --path /path/to/dataset/task ...
```

Dataset repositories own their task layout, runner options, and result storage.
This provider only requires the Antrieb connection settings described below.

## Provider import paths

Datasets select the provider through Harbor's standard import paths:

```text
harbor_antrieb.agent:AntriebHostAgent
harbor_antrieb.environment:AntriebEnvironment
harbor_antrieb.verifier:AntriebVerifier
```

No Harbor source changes or Docker image are required.

## Dataset tasks

The exact task layout is defined by the dataset. A typical Harbor task contains:

```text
task-name/
├── task.toml
├── environment/
├── instruction.md
├── prepare/                 # optional dataset-owned preparation
└── tests/test.sh            # Harbor compatibility sentinel
```

Preparation, execution, and verification happen on the Harbor host. Dataset
authors define the problem and optional initial state; they do not need to write
task-specific verifier commands.

## Collection, execution, and verification

The provider includes a lifecycle collector
(`harbor_antrieb.collector:AntriebCollector`). For every managed-cluster attempt
it records fixed, bounded observations at three boundaries:

1. after provider initialization and before task preparation;
2. after task preparation and before the executor;
3. after the executor, before verification, retry, or teardown.

When preparation is disabled, the second boundary is an alias of the first, so
only two physical snapshots are stored. Fresh-cluster retries receive separate
numbered collector directories.

The host agent records every managed-node command and gives each command a stable
evidence ID. After completing the work, the executor identifies the commands that
best demonstrate each material outcome. This selection does not determine the
score: the verifier receives the public task, topology, preparation baseline,
complete redacted command timeline, and selected evidence, then independently
determines the outcome and confidence.

The fixed observations include temporary-file inventory, failed services,
listening sockets, package-manager health, and platform identity. Comparing
after-prepare with after-executor supplies cross-task operational-hygiene evidence
without task-local probes. Comparing before-prepare with after-prepare identifies
state created by the preparer. Unsupported or unavailable observations remain
explicit and lower confidence rather than becoming task failures.

Collector artifacts are stored under
`collector/attempts/<number>/{manifest.json,commands.jsonl,snapshots/}`. The
verifier reads these files offline and does not issue live node commands. Each
trial also records `executor-evidence.json`, `global-observations.json`, and
`evaluation-report.json` for later analysis. `environment-baseline.json` remains
as a compatibility copy of the effective after-prepare snapshot.

## Development

```bash
git clone https://github.com/open-sudo/harbor-antrieb.git
cd harbor-antrieb
uv sync
uv run pytest -q tests/unit
```

The distribution is named `harbor-antrieb`; its Python import package is
`harbor_antrieb`.

## Credentials

Set `ANTRIEB_TOKEN` on the Harbor host. It is used to access the Antrieb MCP
service and is not copied to managed nodes or included in task artifacts.

Model authentication uses the selected Harbor agent's normal host-side login. The
verifier runs as a log-only model process with no Antrieb token or live-cluster
tools. Verifier results separate `reward`, which measures the observed outcome,
from `confidence`, which measures how directly and completely the recorded evidence
supports that conclusion.
RHEL subscription credentials, when required, use
`HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE` and are not stored in task files.
