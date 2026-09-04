from pathlib import Path

from harbor_antrieb.agent_runner import _build_backend_command


class ModernCodexBackend:
    name = "codex"

    async def __aenter__(self) -> None:
        pass


class ModernClaudeCodeBackend:
    """Mirrors the installed rewardkit/harbor-rewardkit release, which embeds
    the prompt directly in argv (`claude -p <prompt> ...`)."""

    name = "claude-code"

    async def __aenter__(self) -> None:
        pass

    def _build_command(self, prompt: str, schema: dict) -> list[str]:
        return ["claude", "-p", prompt, "--output-format", "json"]


class ModernClaudeCodeBackendWithoutEmbeddedPrompt:
    """A hypothetical fixed backend that already omits the prompt from argv."""

    name = "claude-code"

    async def __aenter__(self) -> None:
        pass

    def _build_command(self, prompt: str, schema: dict) -> list[str]:
        return ["claude", "-p", "--output-format", "json"]


def test_modern_claude_code_prompt_is_sent_via_stdin(tmp_path: Path) -> None:
    prompt = "large prompt" * 20_000

    command, structured_output_path, prompt_input = _build_backend_command(
        ModernClaudeCodeBackend(),
        "claude-code",
        prompt,
        {"type": "object"},
        allowed_tools=(),
        workspace=tmp_path,
    )

    assert command == ["claude", "-p", "--output-format", "json"]
    assert prompt not in command
    assert prompt_input == prompt.encode()
    assert structured_output_path is None


def test_modern_claude_code_backend_without_embedded_prompt_is_left_alone(
    tmp_path: Path,
) -> None:
    prompt = "large prompt" * 20_000

    command, structured_output_path, prompt_input = _build_backend_command(
        ModernClaudeCodeBackendWithoutEmbeddedPrompt(),
        "claude-code",
        prompt,
        {"type": "object"},
        allowed_tools=(),
        workspace=tmp_path,
    )

    assert command == ["claude", "-p", "--output-format", "json"]
    assert prompt_input is None
    assert structured_output_path is None


def test_modern_codex_prompt_is_sent_via_stdin(tmp_path: Path) -> None:
    prompt = "large prompt" * 20_000

    command, structured_output_path, prompt_input = _build_backend_command(
        ModernCodexBackend(),
        "codex",
        prompt,
        {"type": "object"},
        allowed_tools=(),
        workspace=tmp_path,
    )

    assert command[1:3] == ["exec", "-"]
    assert prompt not in command
    assert prompt_input == prompt.encode()
    assert structured_output_path is None


class _StubAgent:
    """Minimal stand-in exposing only what _has_scorable_evidence touches."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir


def _evidence_layout(tmp_path: Path, *, commands: bool, snapshot: bool) -> Path:
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    if commands:
        (agent_dir / "executor-commands.jsonl").write_text('{"command_id":"cmd-1"}\n')
    snapshots = tmp_path / "trial" / "collector" / "attempts" / "01" / "snapshots"
    snapshots.mkdir(parents=True)
    if snapshot:
        (snapshots / "after-executor.json").write_text("{}")
    return agent_dir


def test_expired_run_with_evidence_is_scorable(tmp_path: Path) -> None:
    from harbor_antrieb.agent import AntriebHostAgent

    agent_dir = _evidence_layout(tmp_path, commands=True, snapshot=True)
    assert AntriebHostAgent._has_scorable_evidence(_StubAgent(agent_dir)) is True


def test_run_without_commands_is_not_scorable(tmp_path: Path) -> None:
    from harbor_antrieb.agent import AntriebHostAgent

    agent_dir = _evidence_layout(tmp_path, commands=False, snapshot=True)
    assert AntriebHostAgent._has_scorable_evidence(_StubAgent(agent_dir)) is False


def test_run_without_snapshot_is_not_scorable(tmp_path: Path) -> None:
    from harbor_antrieb.agent import AntriebHostAgent

    agent_dir = _evidence_layout(tmp_path, commands=False, snapshot=False)
    assert AntriebHostAgent._has_scorable_evidence(_StubAgent(agent_dir)) is False


def test_per_attempt_audit_counts_as_evidence(tmp_path: Path) -> None:
    from harbor_antrieb.agent import AntriebHostAgent

    agent_dir = _evidence_layout(tmp_path, commands=False, snapshot=True)
    attempt = agent_dir / "attempts" / "01"
    attempt.mkdir(parents=True)
    (attempt / "executor-commands.jsonl").write_text('{"command_id":"cmd-1"}\n')
    assert AntriebHostAgent._has_scorable_evidence(_StubAgent(agent_dir)) is True
