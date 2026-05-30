"""
parameters.py — Parameter Selection for RLWR-Based MKFHE
=========================================================

Provides closed-form error-growth bounds for three scheme variants and a
binary-search routine that jointly optimises the modulus log q and the
achievable circuit depth d under a target lattice security level.

Scheme identifiers
------------------
SchemeID.LWR_EXTERNAL  (0) — MKFHE built on the LWR-based FHE of
                             (ePrint 2024/960).
SchemeID.LWR_OURS      (1) — MKFHE built on our own LWR-based FHE
                             (Construction 2 of this work).
SchemeID.LWE           (2) — Baseline MKFHE over LWE, included for
                             comparison.

All error bounds are expressed as log₂ values to avoid catastrophic
cancellation when exponents exceed 10³.

"""

from __future__ import annotations

import math
from enum import IntEnum
from dataclasses import dataclass, field

from helper import _safe_log2, _log2_sum
from  error import _get_error_LWR1, _get_error_LWR2, _get_error_LWE





# ---------------------------------------------------------------------------
# Scheme enumeration
# ---------------------------------------------------------------------------

class SchemeID(IntEnum):
    """Identifies which underlying single-key FHE scheme is being lifted."""
    LWR_EXTERNAL = 0   # Construction 1 based on (ePrint 2024/960)
    LWR_OURS     = 1   # This work (Construction 2 / our RLWR FHE)
    LWE          = 2   # LWE baseline


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SchemeParameters:
    """
    A concrete parameter set for one of the three MKFHE constructions.

    Attributes
    ----------
    log_N           : log₂ of the polynomial ring degree N = 2^log_N
    log_q           : log₂ of the outer modulus q
    log_p1          : log₂ of the first rounding modulus (LWR schemes only)
    log_p2          : log₂ of the second rounding modulus (LWR schemes only)
    max_depth       : achievable multiplicative circuit depth
    k_parties       : number of parties
    security_bits   : estimated concrete security level (bits)
    scheme_id       : which construction these parameters target
    modulo_diff_power : gap between consecutive moduli in log₂ (default 4)
    """
    log_N:             int
    log_q:             int
    max_depth:         int
    k_parties:         int
    security_bits:     float
    scheme_id:         SchemeID
    log_p1:            int  = field(default=0)
    log_p2:            int  = field(default=0)
    modulo_diff_power: int  = field(default=4)

    def __str__(self) -> str:
        lwr = self.scheme_id in (SchemeID.LWR_EXTERNAL, SchemeID.LWR_OURS)
        mod_str = (
            f"log(q)={self.log_q}, log(p1)={self.log_p1}, log(p2)={self.log_p2}"
            if lwr
            else f"log(q)={self.log_q}"
        )
        return (
            f"SchemeParameters("
            f"scheme={self.scheme_id.name}, "
            f"N=2^{self.log_N}, "
            f"{mod_str}, "
            f"depth={self.max_depth}, "
            f"k={self.k_parties}, "
            f"security≈{self.security_bits:.1f} bits)"
        )


# ---------------------------------------------------------------------------
# Error-growth bound  (log-domain)
# ---------------------------------------------------------------------------

def error_bound_log2(
    del_r:             int,
    k:                 int,
    d:                 int,
    l:                 float = 10.0,
    s_i:               int   = 1,
    w:                 int   = 2,
    security_level:    int   = 128,
    scheme_id:         SchemeID = SchemeID.LWR_EXTERNAL,
    modulo_diff_power: int   = 4,
    t:                 int   = 3,
) -> float:
    """
    Return log₂ of the required modulus lower bound so that a depth-*d*
    circuit evaluated by *k* parties decrypts correctly at *security_level*
    bits of lattice security.

    This is a tight closed-form bound derived from the noise-propagation
    analysis in the paper.  For LWR-based schemes the output is a lower
    bound on log₂(p₂); for the LWE baseline it is a lower bound on log₂(q).

    Parameters
    ----------
    del_r             : ring degree N = 2^log_N
    k                 : number of parties
    d                 : target multiplicative circuit depth
    l                 : log₂ of the starting modulus (used in decomposition
                        level count)
    s_i               : ∞-norm of each party's secret (typically 1 for
                        ternary secrets)
    w                 : decomposition base (Gadget / KS key)
    security_level    : target λ in bits
    scheme_id         : which construction to analyse
    modulo_diff_power : Δ = log₂(q/p₁) = log₂(p₁/p₂)  (default 4)
    t                 : plaintext modulus  (default 3)

    Returns
    -------
    float  — log₂ of the required lower bound on the relevant modulus
    """


    # ---------------------------------------------------------------
    # Scheme 0 — based on the FHE from (https://eprint.iacr.org/2024/960)
    # ---------------------------------------------------------------
    if scheme_id == SchemeID.LWR_EXTERNAL:
        return _get_error_LWR1(del_r,k,d,l,t,s_i,w,modulo_diff_power)+ security_level

    # ---------------------------------------------------------------
    # Scheme 1 — LWR_OURS  (this work)
    # ---------------------------------------------------------------
    if scheme_id == SchemeID.LWR_OURS:
        return _get_error_LWR2(del_r,k,d,l,t,s_i,w,modulo_diff_power)+ security_level

    # ---------------------------------------------------------------
    # Scheme 2 — LWE  (baseline)
    # ---------------------------------------------------------------
    if scheme_id == SchemeID.LWE:
        return _get_error_LWE(del_r,k,d,l,t,s_i,w,modulo_diff_power)+ security_level

    raise ValueError(f"Unknown scheme_id: {scheme_id}")


# ---------------------------------------------------------------------------
# Required modulus from error bound
# ---------------------------------------------------------------------------

def required_log_modulus(
    log_n:          int,
    k:              int,
    d:              int,
    scheme_id:      SchemeID,
    t:              int = 3,
    w:              int = 2,
    target_level:   int = 128,
) -> int:
    """
    Compute the minimum modulus bitlength needed so that a depth-*d* circuit
    is correct, rounding up to the nearest integer.

    Parameters
    ----------
    log_n        : log₂ N
    k            : number of parties
    d            : circuit depth
    scheme_id    : construction identifier
    t            : plaintext modulus
    w            : decomposition base
    target_level : target security level in bits

    Returns
    -------
    int  — ⌈ log₂ q_min ⌉
    """
    log2_err = error_bound_log2(
        del_r=2 ** log_n,
        k=k,
        d=d,
        l=0,          # no starting modulus needed for the growth formula
        s_i=1,
        w=w,
        security_level=target_level,
        scheme_id=scheme_id,
    )
    return math.ceil(log2_err + math.log2(2 * t))


# ---------------------------------------------------------------------------
# Security oracle  (wraps lattice-estimator)
# ---------------------------------------------------------------------------

_security_cache: dict[tuple[int, int], float] = {}


def lwe_security(log_n: int, log_q: int) -> float:
    """
    Estimate the concrete security of LWE_{2^log_n, 2^log_q} with ternary
    secrets and discrete-Gaussian errors (σ ≈ 4.61) using the
    lattice-estimator library.

    Results are cached to avoid redundant estimation calls.

    Parameters
    ----------
    log_n : log₂ of the LWE dimension
    log_q : log₂ of the modulus

    Returns
    -------
    float — minimum log₂(rop) across primal-BDD, primal-uSVP, and
            dual-hybrid attacks
    """
    key = (log_n, log_q)
    if key in _security_cache:
        return _security_cache[key]


    from estimator.estimator import LWE, ND  # type: ignore[import]

    Xs = ND.UniformMod(3)
    Xe = ND.DiscreteGaussian(4.61)
    params = LWE.Parameters(n=2**log_n, q=2**log_q, Xs=Xs, Xe=Xe)
    sec = min(
        math.log2(alg["rop"])
        for alg in [
            LWE.primal_bdd(params),
            LWE.primal_usvp(params),
            LWE.dual_hybrid(params),
        ]
    )


    _security_cache[key] = sec
    return sec


# ---------------------------------------------------------------------------
# Main parameter search
# ---------------------------------------------------------------------------

def find_parameters(
    log_n:          int,
    k:              int,
    scheme_id:      SchemeID,
    target_level:   int = 128,
    max_depth_cap:  int = 100,
    t:              int = 3,
    w:              int = 2,
    modulo_diff_power: int = 4,
) -> SchemeParameters | None:
    """
    Binary-search for the largest circuit depth *d* that is simultaneously
    correct (noise budget allows decryption) and secure (≥ *target_level*
    bits against known lattice attacks), then package the result.

    Parameters
    ----------
    log_n            : log₂ of the ring degree N
    k                : number of parties
    scheme_id        : which construction to parametrise
    target_level     : target λ in bits (default 128)
    max_depth_cap    : upper search bound on d (default 100)
    t                : plaintext modulus (default 3)
    w                : decomposition base (default 2)
    modulo_diff_power: Δ in log₂ units (default 4)

    Returns
    -------
    SchemeParameters — best feasible parameters, or None if no depth ≥ 1
                       is achievable.
    """
    lo, hi = 1, max_depth_cap
    best: SchemeParameters | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        log_q  = required_log_modulus(log_n, k, mid, scheme_id, t, w, target_level)
        sec    = lwe_security(log_n, log_q)

        if sec >= target_level:
            p = SchemeParameters(
                log_N=log_n,
                log_q=log_q,
                log_p1=log_q + modulo_diff_power if scheme_id != SchemeID.LWE else 0,
                log_p2=log_q + 2 * modulo_diff_power if scheme_id != SchemeID.LWE else 0,
                max_depth=mid,
                k_parties=k,
                security_bits=sec,
                scheme_id=scheme_id,
                modulo_diff_power=modulo_diff_power,
            )
            best = p
            lo = mid + 1        # try to squeeze out more depth
        else:
            hi = mid - 1        # modulus too large; reduce depth

    return best


def select_parameters(
    scheme_id:      SchemeID,
    log_n_range:    range    = range(13, 17),
    k_values:       list[int] | None = None,
    target_level:   int = 128,
    verbose:        bool = True,
) -> dict[tuple[int, int], SchemeParameters]:
    """
    Sweep over ring sizes and party counts, returning all feasible parameter
    sets as a dictionary keyed by (log_N, k).

    Parameters
    ----------
    scheme_id    : which construction to sweep
    log_n_range  : iterable of log₂(N) values to test
    k_values     : list of party counts to test (default [2, 4, 8, 128])
    target_level : target security level in bits
    verbose      : if True, print results as they are found

    Returns
    -------
    dict mapping (log_N, k) → SchemeParameters
    """
    if k_values is None:
        k_values = [2, 4, 8, 128]

    results: dict[tuple[int, int], SchemeParameters] = {}

    for log_n in log_n_range:
        for k in k_values:
            params = find_parameters(log_n, k, scheme_id, target_level)
            if params is None:
                if verbose:
                    print(
                        f"  [!] No feasible parameters for "
                        f"log_N={log_n}, k={k}, scheme={scheme_id.name}"
                    )
                continue
            results[(log_n, k)] = params
            if verbose:
                print(f"  [+] {params}")

    return results
