from __future__ import annotations

__all__ = ["DISABLED_EVALUATION_DIMENSIONS"]


# Keep authored task criteria intact so the dimension can be restored later without
# rewriting every task. Verifiers must exclude these dimensions from evidence
# collection, completeness calculations, aggregation, and emitted rewards.
DISABLED_EVALUATION_DIMENSIONS = frozenset({"security"})
