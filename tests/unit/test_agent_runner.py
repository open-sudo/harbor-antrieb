from pathlib import Path

from harbor_antrieb.agent_runner import _build_backend_command


class ModernCodexBackend:
    name = "codex"

    async def __aenter__(self) -> None:
        pass


class ModernClaudeCodeBackend:
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

    assert prompt not in command
    assert prompt_input == prompt.encode()
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
