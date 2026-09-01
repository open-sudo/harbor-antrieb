from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harbor_antrieb.agent import AntriebHostAgent
    from harbor_antrieb.ai_preparer import run_ai_prepare
    from harbor_antrieb.collector import AntriebCollector
    from harbor_antrieb.environment import AntriebEnvironment
    from harbor_antrieb.static_preparer import run_static_prepare
    from harbor_antrieb.verifier import AntriebVerifier

__all__ = [
    "AntriebCollector",
    "AntriebEnvironment",
    "AntriebHostAgent",
    "AntriebVerifier",
    "run_ai_prepare",
    "run_static_prepare",
]


def __getattr__(name: str) -> Any:
    if name == "AntriebCollector":
        from harbor_antrieb.collector import AntriebCollector

        return AntriebCollector
    if name == "AntriebHostAgent":
        from harbor_antrieb.agent import AntriebHostAgent

        return AntriebHostAgent
    if name == "AntriebEnvironment":
        from harbor_antrieb.environment import AntriebEnvironment

        return AntriebEnvironment
    if name == "run_static_prepare":
        from harbor_antrieb.static_preparer import run_static_prepare

        return run_static_prepare
    if name == "run_ai_prepare":
        from harbor_antrieb.ai_preparer import run_ai_prepare

        return run_ai_prepare
    if name == "AntriebVerifier":
        from harbor_antrieb.verifier import AntriebVerifier

        return AntriebVerifier
    raise AttributeError(name)
