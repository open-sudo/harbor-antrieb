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
