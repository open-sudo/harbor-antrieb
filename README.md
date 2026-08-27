# InfraSet

Standalone Harbor extension for evaluating agents on managed infrastructure. The
initial provider is Antrieb. InfraSet uses Harbor 0.22's
existing custom environment, custom agent, and custom verifier import paths; it does
not patch or register anything in Harbor core, and it does not require a Dockerfile.

## Runtime contract

- Harbor provisions clusters from `environment/infraset.toml` and always deletes
  the active cluster during cleanup. Tasks default to one cluster and may declare a
  bounded `max_clusters` quota for fresh-cluster executor retries.
- After provisioning, InfraSet runs ordered, trusted `initialize()` actions against
  the complete managed environment, then runs optional task-owned `prepare()` state
  construction. Initialization may configure nodes, networks, provider access, or
  licenses and is not itself brownfield task state.
- Static preparation, the executor, and verifier run from the Harbor host. No agent
  CLI, agent credential, or harness bridge is installed on a managed node.
- Before provisioning, the environment reads only the task-declared Antrieb base
  runbooks and keeps their bodies in memory. Their platform guidance is included in
  host-side AI prompts, but their contents are not written to trial artifacts.
- The Python stdio bridge binds every live call to Harbor's session and exposes no
  provision, delete, save, or search tool to an agent. Executors and preparers
  receive only `exec`. AI evaluators additionally receive the local
  `complete_evidence` tool;
  calling it permanently closes `exec` for that invocation. The bridge contains no
  distribution-specific shell-command allowlist.
- Host-side AI executors and evaluators access every node independently through the
  bridge and unchanged Antrieb `exec` API. Static preparation and verification call
  that same API directly. None relies on a controller node.
- AI verification levels 3 through 10 start one bounded, read-only inspector per
  managed node in parallel. Each inspector receives only that node through the
  unchanged `exec` API and the same level-scaled command budget. A level-scaled
  coordinator then consumes the node reports and serializes cross-node probes,
  controlled failures, recovery, and reboots while balancing remaining commands
  across suitable nodes. Levels 1 and 2 skip the parallel pass and run only a small
  coordinator-led smoke evaluation. Level 1 scores only functionality; level 2
  scores functionality and security. Other dimensions are deterministically marked
  unverified at those depths.
- `control_node` is used only for Harbor's internal artifact compatibility. It is
  not where the executor runs.
- Static preparation may create a task-defined brownfield state before the executor.
  Its subsequent task-authored baseline observations inspect the resulting state.
  Static checks may reboot nodes and inject controlled failures, but cannot replace
  the cluster.
- Each live-cluster host-side AI invocation receives an invocation-scoped
  `infraset` MCP server and is instructed to use only its managed `exec` tool.
- AI evaluators explicitly close evidence collection after their last cluster
  check, then compose and schema-validate the report offline. If the cluster expires
  during finalization, InfraSet accepts the last complete valid report when the
  command audit proves that no live operation was attempted after expiration.
- If an executor reports `blocked`, reaches its per-attempt timeout, or loses an
  expired cluster, a cluster-disconnected postmortem evaluator reads the redacted
  command audit and CLI logs. When quota remains, Harbor replaces the cluster,
  repeats the same preparation, and gives the written diagnosis to a new executor
  invocation.
- Executor authentication, CLI startup, parsing, and other harness failures are not
  infrastructure retries. Reprovisioning cannot correct them.

## Install the provider

For normal use, do not clone Harbor or this provider. The task runner installs
Harbor and fetches this provider from GitHub automatically:

```bash
export ANTRIEB_TOKEN=ant_...
```

For provider development, clone this repository and install its development
environment with:

```bash
uv sync
```

The executor uses RewardKit's host CLI integration, so configure its model credential
on the Harbor host as usual. Static verification uses no model credential. The
Antrieb token remains on the host and is never copied to a managed node or included
in an agent prompt.

## Task files

Preparation and evaluation implementations are split by execution type:

```text
ai_preparer.py
static_preparer.py
verifier.py
```

`environment/infraset.toml` defines the managed topology:

```toml
cluster = ["ubuntu24.04 x3"]
max_clusters = 3
initialize = []
base_runbooks = ["antrieb/primer"]
control_node = "node1"
endpoint = "https://antrieb.sh/mcp"
```

`antrieb/primer` is required and is the default. Tasks using custom networks add
`antrieb/networking-primer`; tasks using a specialized appliance add its reference,
for example `antrieb/vyos-reference`. Only base primers and appliance references in
the read-only `antrieb` namespace are accepted. Scenario runbooks are intentionally
excluded so they cannot supply a task recipe.

`max_clusters` is both the task's retry ceiling and its infrastructure budget. It
defaults to `1`. Every executor attempt receives one cluster, and every replacement
uses the task's original topology and runs preparation again. The Galera examples
use three; the simpler examples retain the one-cluster default.

### Trusted initialization

`initialize` is an ordered list of allowlisted, harness-owned actions. It runs after
each provision or reprovision and before task preparation or any agent. Initializers
receive the complete provisioned environment, so future actions may target a node,
network, or the cluster as a whole. Tasks cannot provide initializer commands or
credential values.

The built-in `rhsm` initializer securely registers every exact RHEL node reported
in the provision response:

```toml
cluster = ["rhel9.8 x2"]
initialize = ["rhsm"]
```

Set `INFRASET_INITIALIZE_CREDENTIALS_FILE` on the Harbor host to an absolute,
regular file that is not group/world accessible. The `rhsm` initializer splits the
file on line breaks and requires exactly two nonempty lines: the Red Hat username
on line 1 and password on line 2. InfraSet passes those values separately through
`secret_env` only for the trusted initializer call. The credentials are not placed
in task files, provision requests, reports, agent prompts, audit logs, agent
subprocess environments, or persistent VM state.
Initialization failure aborts preparation and deletes the cluster. Antrieb performs
RHSM unregister/clean during normal cluster deletion.

### Static brownfield preparation

Enable deterministic scenario setup and baseline capture in the same environment
definition.

```toml
[prepare]
enabled = true
mode = "static"
setup = "prepare/setup.toml"
baseline = "prepare/baseline.toml"
```

`prepare/setup.toml` contains mutating commands. Commands in one stage run in
parallel; later stages wait for earlier ones.

```toml
timeout_sec = 300

[[steps]]
id = "seed-existing-data-node1"
stage = 10
node = "node1"
command = '''
install -d -m 0750 /srv/customer-data
printf '%s\n' 'must survive' > /srv/customer-data/record.txt
'''
```

`prepare/baseline.toml` then captures task-specific facts without changing the
prepared cluster:

```toml
timeout_sec = 120

[[observations]]
id = "customer-data-checksum"
node = "node1"
command = "sha256sum /srv/customer-data/record.txt"
required = true
```

Any failed setup step or required baseline observation aborts the trial and deletes
the cluster before the executor starts. Optional observations use `required = false`
and are recorded as limitations instead. Preparation produces:

```text
prepare/setup-commands.jsonl
prepare/setup-report.json
prepare/baseline-commands.jsonl
prepare/baseline-report.json
```

A static verifier check can compare its output with a captured observation. This is
useful for proving that existing data was preserved:

```toml
[[checks]]
id = "customer-data-preserved"
node = "node1"
dimension = "operational_hygiene"
command = "sha256sum /srv/customer-data/record.txt"
baseline_observation = "customer-data-checksum"
baseline_mode = "equal"
```

Use `baseline_mode = "different"` when the task requires changing the captured
value. Setup retries are opt-in because mutating commands should normally run once.

For generated brownfield state, select the AI preparer and provide a task-specific
prompt. The AI creates the starting state, then the same static baseline capture runs
against the actual nodes:

```toml
[prepare]
enabled = true
mode = "ai"
agent = "codex"
model = "gpt-5.6-sol"
reasoning_effort = "medium"
timeout_sec = 1200
prompt = "prepare/prompt.md"
baseline = "prepare/baseline.toml"
```

The backend and model have no built-in defaults in AI mode. They can be overridden
per run without changing the task:

```bash
--ek prepare_agent=codex \
--ek prepare_model=gpt-5.6-sol \
--ek prepare_reasoning_effort=medium
```

AI preparation writes `prepare/ai-setup-commands.jsonl`,
`prepare/ai-preparer-output.json`, and `prepare/ai-preparer-raw.json`. Failure or a
`blocked` result aborts the trial before the executor starts.

Explicit `[[networks]]` and `[nics]` use the same values accepted by Antrieb's
unchanged `provision` tool.

The task needs no agent credentials or MCP configuration:

```toml
[environment]
network_mode = "public"
```

Keep the task's instruction explicit that the cluster is managed and that the
executor must not attempt cluster lifecycle operations.

The private `verifier/checks.toml` contains bounded evidence probes and
atomic, authored assertions. The semantic verifier collects evidence with the
configured AI backend, while the harness computes scores from assertion points.

InfraSet examples are grouped by initial state and operational complexity:

```text
../infraset/tasks/greenfield/  # pristine managed nodes; no preparation
../infraset/tasks/brownfield/  # task-authored preparation and baseline state
../infraset/tasks/complex/greenfield/  # complex tasks on pristine nodes
../infraset/tasks/complex/brownfield/  # complex tasks with prepared state
```

The complex category is reserved for tightly coupled distributed services,
consensus/replication plus recovery, specialized multi-NIC appliances, or workflows
whose realistic execution can consume most of a managed-cluster lease. A task is not
complex merely because it uses several nodes.

For authored natural-language checks with deterministic scoring, use
`InfraSetVerifier`. A task's `verifier/checks.toml` separates
bounded evidence `[[probes]]` from atomic scored `[[assertions]]`. Probes declare
cumulative level, target nodes, allowed effect, command budget, procedure, and
cleanup. Assertions reference a probe and declare a dimension, points, optional
criticality and prerequisites, and explicit pass/fail conditions. The AI executes
probes and classifies assertions but never assigns scores; the harness computes
dimension scores from authored points.

An assertion the evaluator cannot establish is `indeterminate`. Command errors,
expired access, ambiguous output, cleanup failure, and unmet prerequisites affect
only the corresponding assertions. They reduce `evaluation_coverage` without
becoming executor failures. By default, anything below full coverage sets
`evaluation_complete = false`, makes the report ineligible for publication, and
omits the primary `reward`; the report retains a diagnostic `provisional_reward`.

```bash
--verifier infraset.verifier:InfraSetVerifier \
--verifier-kwarg agent=codex \
--verifier-kwarg model=gpt-5.6-sol \
--verifier-kwarg reasoning_effort=low \
--verifier-kwarg service_tier=fast \
--verifier-kwarg level=2 \
--verifier-kwarg minimum_coverage=1.0
```

Semantic verification levels are task-authored and cumulative. Controlled failures
are rejected below level 6 and reboots below level 8. Reports include per-assertion
evidence, deterministic point totals, `evaluation_coverage`,
`evaluation_complete`, and `publication_eligible`.

Harbor 0.22 validates the presence of `tests/test.sh` before it knows which runtime
custom verifier was selected. Until that upstream validation becomes verifier-aware,
the example includes a fail-closed compatibility sentinel for the semantic verifier.

## Run

```bash
uv run --no-project \
  --with 'harbor-antrieb @ git+https://github.com/open-sudo/harbor-antrieb.git' \
  harbor run \
  --path ../infraset/tasks/greenfield/managed \
  --agent infraset.agent:InfraSetHostAgent \
  --model gpt-5.6-sol \
  --agent-kwarg agent_name=codex \
  --agent-kwarg reasoning_effort=medium \
  --agent-kwarg service_tier=fast \
  --agent-kwarg diagnostic_agent=codex \
  --agent-kwarg diagnostic_model=gpt-5.6-sol \
  --agent-kwarg diagnostic_reasoning_effort=medium \
  --env infraset.environment:InfraSetEnvironment \
  --verifier infraset.verifier:InfraSetVerifier \
  --verifier-kwarg agent=codex \
  --verifier-kwarg model=gpt-5.6-sol \
  --verifier-kwarg reasoning_effort=low \
  --verifier-kwarg level=2 \
  --verifier-kwarg minimum_coverage=1.0
```

This reuses the host's existing Codex login; no model credential is sent to an
Antrieb node. Increase `level` for progressively deeper semantic evaluation; use
`level=10` (or omit the kwarg) for complete authored coverage.
The optional `service_tier=fast` executor kwarg requests Codex Fast mode when the
selected model and account advertise that tier. Omit it to use the account default.

Run an InfraSet task with the task runner:

```bash
cd /home/antrieb-studio/infraset
./run-task.sh \
  ./tasks/complex/greenfield/<task-name>
```

The task runner uses semantic verification and accepts the task path as its first
argument. Run separate tasks or build a Harbor dataset directory when evaluating a
larger batch.

Equivalent persisted job configuration uses:

```json
{
  "agents": [{
    "import_path": "infraset.agent:InfraSetHostAgent",
    "model_name": "gpt-5.6-sol",
    "kwargs": {
      "agent_name": "codex",
      "reasoning_effort": "medium",
      "diagnostic_agent": "codex",
      "diagnostic_model": "gpt-5.6-sol",
      "diagnostic_reasoning_effort": "medium"
    }
  }],
  "environment": {
    "import_path": "infraset.environment:InfraSetEnvironment"
  },
  "verifier": {
    "import_path": "infraset.verifier:InfraSetVerifier"
  }
}
```

The trial verifier directory receives semantic evidence and evaluation reports,
alongside deterministic reward details and the final reward.

The trial agent directory receives a combined `executor-commands.jsonl`,
`agent-output.json`, and `attempt-history.json`. Detailed artifacts are kept under
`attempts/NN/`, including each executor audit/output and, for failed attempts,
`postmortem/diagnosis.json`. Persisted executor commands, stdout, stderr, reports,
and postmortems are bounded and best-effort redacted; task authors should still
require agents to generate credentials on-node and use `secret_env` instead of
placing literals in commands.
