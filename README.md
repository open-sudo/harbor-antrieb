# Harbor Antrieb

Harbor provider for running datasets on Antrieb-managed infrastructure. It
provides a custom environment, agent, and verifier that any Harbor dataset can
load through import paths.

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
├── instructions.md
├── prepare/                 # optional dataset-owned preparation
└── verifier/                # optional dataset-owned checks
```

Preparation, execution, and verification happen on the Harbor host. Dataset
authors decide which task files and checks are appropriate.

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

Model authentication uses the selected Harbor agent's normal host-side login.
RHEL subscription credentials, when required, use
`HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE` and are not stored in task files.
