from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infraset.agent import InfraSetHostAgent
    from infraset.ai_preparer import run_ai_prepare
    from infraset.environment import InfraSetEnvironment
    from infraset.verifier import InfraSetVerifier
    from infraset.static_preparer import run_static_prepare

__all__ = [
    "InfraSetEnvironment",
    "InfraSetHostAgent",
    "InfraSetVerifier",
    "run_ai_prepare",
    "run_static_prepare",
]


def __getattr__(name: str) -> Any:
    if name == "InfraSetHostAgent":
        from infraset.agent import InfraSetHostAgent

        return InfraSetHostAgent
    if name == "InfraSetEnvironment":
        from infraset.environment import InfraSetEnvironment

        return InfraSetEnvironment
    if name == "run_static_prepare":
        from infraset.static_preparer import run_static_prepare

        return run_static_prepare
    if name == "run_ai_prepare":
        from infraset.ai_preparer import run_ai_prepare

        return run_ai_prepare
    if name == "InfraSetVerifier":
        from infraset.verifier import InfraSetVerifier

        return InfraSetVerifier
    raise AttributeError(name)
