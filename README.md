# Harbor Antrieb

Harbor integration for running InfraSet tasks on Antrieb-managed infrastructure.
It provides the custom environment, agent, and verifier used by InfraSet tasks.

## Normal use

Users do not need to clone Harbor or this repository. The InfraSet task runner
fetches Harbor and this provider from GitHub:

```bash
cd ~/infraset
export ANTRIEB_TOKEN='ant_...'
./run-task.sh ./tasks/greenfield/<task-name>
```

Results are written to `~/infraset/jobs/<task-name>`.

The task repository owns any runner-specific configuration. This provider only
requires the Antrieb connection settings described below.

## Provider import paths

Tasks select the provider through Harbor's standard import paths:

```text
infraset.agent:InfraSetHostAgent
infraset.environment:InfraSetEnvironment
infraset.verifier:InfraSetVerifier
```

No Harbor source changes or Docker image are required.

## Task structure

An InfraSet task is a directory containing `task.toml` and usually:

```text
task-name/
├── task.toml
├── environment/infraset.toml
├── instructions.md
├── prepare/                 # optional trusted brownfield setup
└── verifier/                # task-specific semantic checks
```

The task declares its topology in `environment/infraset.toml`. Preparation,
execution, and verification happen on the Harbor host.

## Development

```bash
git clone https://github.com/open-sudo/harbor-antrieb.git
cd harbor-antrieb
uv sync
uv run pytest -q tests/unit
```

The distribution is named `harbor-antrieb`; its Python import package is
`infraset`.

## Credentials

Set `ANTRIEB_TOKEN` on the Harbor host. It is used to access the Antrieb MCP
service and is not copied to managed nodes or included in task artifacts.

Model authentication uses the selected Harbor agent's normal host-side login.
RHEL subscription credentials, when required, use
`HARBOR_ANTRIEB_INITIALIZE_CREDENTIALS_FILE` and are not stored in task files.
