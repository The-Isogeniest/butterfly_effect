
from  helper import _log2_sum, _safe_log2
"""
Get log2(error) after evaluating circuits of depth d according to the different schemes LWR1, LWR2, and LWE
"""

def _get_error_LWR1(del_r, k, d, l, t, s_i=1, w=2, modulo_diff_power=4):
    """Construction 1 error bound — matches get_error in Construction_1.ipynb."""
    s_norm = k * s_i
    p2byp1 = 2 ** (-modulo_diff_power)
    log_del_r = _safe_log2(del_r)
    log_t = _safe_log2(t)
    log_s_norm = _safe_log2(s_norm) if s_norm > 0 else -float('inf')

    log_p2byp1 = -modulo_diff_power  # log2(2^{-modulo_diff_power})

    # -----------------------------
    # Common c1 (all schemes)
    # c1 = (del_r*s_norm + 4)*(del_r*t) + del_r/2
    # -----------------------------
    log_term1 = _safe_log2(del_r * s_norm + 4)
    log_term2 = log_del_r + log_t
    log_part1 = log_term1 + log_term2
    log_part2 = log_del_r - 1
    log_c1 = _log2_sum(log_part1, log_part2)

    l_eff = l - 2 * modulo_diff_power

    log_y_in = _log2_sum(
        log_p2byp1 + log_del_r + log_s_norm,
        -1,
    )

    log_A = 2 * log_t + log_del_r + _safe_log2(del_r * s_norm / 2 + 2.5)
    log_B = 2 * log_p2byp1 + 2 * log_del_r + 2 * log_s_norm
    log_C = log_p2byp1 + log_del_r + log_s_norm
    log_D = 0.0
    log_inner = _log2_sum(_log2_sum(log_B, log_C), log_D) - 1
    log_E = _safe_log2(l_eff * w * k) + 2 * log_del_r + log_s_norm + log_p2byp1 - 1
    log_F = _safe_log2(k * l_eff * w) + log_del_r - 1

    log_c2 = _log2_sum(_log2_sum(_log2_sum(log_A, log_inner), log_E), log_F)
    log_sum = _log2_sum(log_c1 + log_y_in, log_c2)
    return (d - 1) * log_c1 + log_sum



def _get_error_LWR2(del_r, k, d, l, t, s_i=1, w=2,  modulo_diff_power=4):
    """Construction 2 error bound — matches get_error in Construction_2.ipynb."""
    s_norm = k * s_i
    p2byp1 = 2 ** (-modulo_diff_power)
    log_del_r = _safe_log2(del_r)
    log_t = _safe_log2(t)
    log_s_norm = _safe_log2(s_norm) if s_norm > 0 else -float('inf')

    log_p2byp1 = -modulo_diff_power  # log2(2^{-modulo_diff_power})

    # -----------------------------
    # Common c1 (all schemes)
    # c1 = (del_r*s_norm + 4)*(del_r*t) + del_r/2
    # -----------------------------
    log_term1 = _safe_log2(del_r * s_norm + 4)
    log_term2 = log_del_r + log_t
    log_part1 = log_term1 + log_term2
    log_part2 = log_del_r - 1
    log_c1 = _log2_sum(log_part1, log_part2)

    l_eff = l - 2 * modulo_diff_power

    log_y_in = _log2_sum(
        _safe_log2(p2byp1 + 1) + log_del_r + log_s_norm - 1,
        -1,
    )

    log_A = 2 * log_t + log_del_r + _safe_log2(del_r * s_norm / 2 + 2.5)
    log_B = 2 * log_del_r + 2 * log_s_norm
    log_C = log_del_r + log_s_norm
    log_inner = _log2_sum(_log2_sum(log_B, log_C), 0.0) - 1
    log_E = _safe_log2(3 * l_eff * w * k / 4) + 2 * log_del_r + log_s_norm + log_p2byp1
    log_F = _safe_log2(k * l_eff * w) + log_del_r - 1
    log_G = log_del_r + log_s_norm - 1

    log_c2 = _log2_sum(
        _log2_sum(_log2_sum(_log2_sum(log_A, log_inner), log_E), log_F), log_G
    )
    log_sum = _log2_sum(log_c1 + log_y_in, log_c2)
    return (d - 1) * log_c1 + log_sum



def _get_error_LWE(del_r, k, d, l, t, s_i=1, w=2, modulo_diff_power=4):
    """Construction 2 error bound — matches get_error in Construction_2.ipynb."""

    s_norm = k * s_i
    log_del_r = _safe_log2(del_r)
    log_t = _safe_log2(t)
    log_s_norm = _safe_log2(s_norm) if s_norm > 0 else -float('inf')


    # -----------------------------
    # Common c1 (all schemes)
    # c1 = (del_r*s_norm + 4)*(del_r*t) + del_r/2
    # -----------------------------
    log_term1 = _safe_log2(del_r * s_norm + 4)
    log_term2 = log_del_r + log_t
    log_part1 = log_term1 + log_term2
    log_part2 = log_del_r - 1
    log_c1 = _log2_sum(log_part1, log_part2)

    log_e1 = modulo_diff_power - 1

    # y_in2
    log_y1 = _safe_log2(2) + log_del_r + log_e1 + log_s_norm
    log_y_in = _log2_sum(log_y1, log_e1)

    # c2
    log_A = 2 * log_t + log_del_r + _safe_log2(del_r * s_norm / 2 + 2.5)

    log_B = 2 * log_del_r + 2 * log_s_norm
    log_C = log_del_r + log_s_norm
    log_D = 0

    log_inner = _log2_sum(log_B, log_C)
    log_inner = _log2_sum(log_inner, log_D)
    log_inner -= 1

    log_E = _safe_log2(3 * l * w * k) + 2 * log_del_r + log_s_norm + log_e1
    log_F = _safe_log2(2 * l * w * k) + log_del_r + log_e1 + log_s_norm

    log_c2 = _log2_sum(log_A, log_inner)
    log_c2 = _log2_sum(log_c2, log_E)
    log_c2 = _log2_sum(log_c2, log_F)

    log_c1_y = log_c1 + log_y_in
    log_sum = _log2_sum(log_c1_y, log_c2)

    log_d_e = (d - 1) * log_c1 + log_sum

    return log_d_e #### divide by the number of the parties

