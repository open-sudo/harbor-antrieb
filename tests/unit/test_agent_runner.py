from pathlib import Path

from infraset.agent_runner import _build_backend_command


class ModernCodexBackend:
    name = "codex"

    async def __aenter__(self) -> None:
        pass


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
