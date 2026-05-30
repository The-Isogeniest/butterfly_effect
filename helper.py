# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import  math
def _safe_log2(x: float) -> float:
    """log₂(x), returning -∞ for x ≤ 0."""
    if x <= 0:
        return -math.inf
    return math.log2(x)


def _log2_sum(log_a: float, log_b: float) -> float:
    """Numerically stable log₂(2^a + 2^b)."""
    if log_a > log_b:
        return log_a + _safe_log2(1 + 2 ** (log_b - log_a))
    return log_b + _safe_log2(1 + 2 ** (log_a - log_b))