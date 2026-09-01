from __future__ import annotations

import shlex


def preferred_shell_command(command: str, *, login: bool = False) -> str:
    """Run a command with Bash when installed, otherwise with POSIX sh."""
    bash_option = "-lc" if login else "-c"
    selector = (
        "if command -v bash >/dev/null 2>&1; then "
        f"exec bash {bash_option} {shlex.quote(command)}; "
        f"else exec /bin/sh -c {shlex.quote(command)}; fi"
    )
    return f"/bin/sh -c {shlex.quote(selector)}"
