"""
benchmark.py — Correctness Tests and Circuit-Depth Benchmarks
=============================================================

Verifies that the achieved multiplicative circuit depth matches the
theoretical predictions, across the paper's parameter sets.

Usage
-----
    python benchmark.py --construction 1 --attempts 10
    python benchmark.py --construction 2 --k 4

"""

from __future__ import annotations

import argparse
import math
import numpy as np

from poly_utils import poly_mul
from construction1 import MKFHE_Construction1
from construction2 import MKFHE_Construction2
from helper import _safe_log2, _log2_sum

from error import _get_error_LWR1, _get_error_LWR2, _get_error_LWE


# ---------------------------------------------------------------------------
# Preselected parameter sets from the paper
# ---------------------------------------------------------------------------

PAPER_PARAMS_C1 = [
    {"N": 2**13, "r": 2**206, "q": 2**202, "p": 2**198},
    {"N": 2**14, "r": 2**425, "q": 2**421, "p": 2**417},
    {"N": 2**15, "r": 2**868, "q": 2**864, "p": 2**860},
    {"N": 2**16, "r": 2**1743, "q": 2**1739, "p": 2**1735},
]

PAPER_PARAMS_C2 = [
    {"N": 2**13, "q": 2**209, "p1": 2**205, "p2": 2**201},
    {"N": 2**14, "q": 2**428, "p1": 2**424, "p2": 2**420},
    {"N": 2**15, "q": 2**871, "p1": 2**867, "p2": 2**863},
    {"N": 2**16, "q": 2**1746, "p1": 2**1742, "p2": 2**1738},
]


# ---------------------------------------------------------------------------
# Error-magnitude helpers  (matching your get_error functions exactly)
# ---------------------------------------------------------------------------

def _get_error_c1(del_r, k, d, l, t, s_i=1, w=2, security_level=128, modulo_diff_power=4):
    """Construction 1 error bound — matches get_error in Construction_1.ipynb."""

    return _get_error_LWR1(del_r,k,d,l,t,s_i,w,modulo_diff_power)- _safe_log2(k) ### divide by the number of the parties


def _get_error_c2(del_r, k, d, l, t, s_i=1, w=2, security_level=128, modulo_diff_power=4):
    """Construction 2 error bound — matches get_error in Construction_2.ipynb."""

    return _get_error_LWR2(del_r,k,d,l,t,s_i,w,modulo_diff_power)- _safe_log2(k) ### divide by the number of the parties


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_construction1(k_parties: int = 2, attempts: int = 10) -> None:
    """Run Construction 1 depth benchmarks over the paper's parameter sets."""
    print("\n" + "=" * 70)
    print("  Construction 1 (RLWR-MKFHE from Bolboceanu et al. (ePrint 2024/960.)")
    print(f"  k = {k_parties} parties,  {attempts} trials per parameter set")
    print("=" * 70)

    for params in PAPER_PARAMS_C1:
        N = params["N"]
        r = params["r"]
        q = params["q"]
        p = params["p"]
        t = 3

        scheme = MKFHE_Construction1(N=N, r=r, q=q, p=p, t=t)
        print(f"\n  {scheme}")
        print(f"  testing for parameter sets N={N}, log2(r)={math.log2(r)}, "
              f"log2(q)={math.log2(q)}, log2(p)={math.log2(p)}")

        sk_list, pk_list = scheme.keygen_multiparty(k_parties)
        epk  = scheme.key_extension(pk_list)
        rkey = scheme.relinkey_gen_multiparty(sk_list)

        avg = 0

        for trial in range(attempts):
            msg1 = scheme.random_message()
            ct1  = scheme.encrypt(msg1, epk)
            i    = 0

            while True:
                msg2       = scheme.random_message()
                ct2        = scheme.encrypt(msg2, epk)
                ct_mul     = scheme.multiply(ct1, ct2, rkey)
                msg_mul    = poly_mul(msg1, msg2, N) % t

                error_magnitude = _get_error_c1(
                    N, k_parties, i + 1, math.log2(r),
                    t, s_i=1, w=2, security_level=128, modulo_diff_power=4,
                )
                dec = scheme.decrypt(sk_list, ct_mul, lamda=128 + error_magnitude)

                if not np.array_equal(dec, msg_mul):
                    print("  failure")
                    break

                print(f"  multiplicative depth: {i}")
                i   += 1
                msg1 = msg_mul
                ct1  = ct_mul

            avg += i
            print(f"  depth: {i}")

        print(f"  The average depth for this parameter set: {avg / attempts}")
        print("  " + "-" * 66)


def benchmark_construction2(k_parties: int = 2, attempts: int = 10) -> None:
    """Run Construction 2 depth benchmarks over the paper's parameter sets."""
    print("\n" + "=" * 70)
    print("  Construction 2 (RLWR-MKFHE, this work)")
    print(f"  k = {k_parties} parties,  {attempts} trials per parameter set")
    print("=" * 70)

    for params in PAPER_PARAMS_C2:
        N  = params["N"]
        q  = params["q"]
        p1 = params["p1"]
        p2 = params["p2"]
        t  = 3

        scheme = MKFHE_Construction2(N=N, q=q, p1=p1, p2=p2, t=t)
        print(f"\n  {scheme}")
        print(f"  testing for parameter sets N={N}, log2(q)={math.log2(q)}, "
              f"log2(p1)={math.log2(p1)}, log2(p2)={math.log2(p2)}")

        sk_list, pk_list = scheme.keygen_multiparty(k_parties)
        epk  = scheme.key_extension(pk_list)
        rkey = scheme.relinkey_gen_multiparty(sk_list)

        avg = 0

        for trial in range(attempts):
            msg1 = scheme.random_message()
            ct1  = scheme.encrypt(msg1, epk)
            i    = 0

            while True:
                msg2    = scheme.random_message()
                ct2     = scheme.encrypt(msg2, epk)
                ct_mul  = scheme.multiply(ct1, ct2, rkey)
                msg_mul = poly_mul(msg1, msg2, N) % t

                error_magnitude = _get_error_c2(
                    N, k_parties, i + 1, math.log2(q),
                    t, s_i=1, w=2, security_level=128, modulo_diff_power=4,
                )
                dec = scheme.decrypt(sk_list, ct_mul, lamda=128 + error_magnitude)

                if not np.array_equal(dec, msg_mul % t):
                    print("  failure")
                    break

                print(f"  multiplicative depth: {i}")
                i   += 1
                msg1 = msg_mul
                ct1  = ct_mul

            avg += i
            print(f"  depth: {i}")

        print(f"  The average depth for this parameter set: {avg / attempts}")
        print("  " + "-" * 66)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RLWR-MKFHE circuit depth."
    )
    parser.add_argument(
        "--construction", type=int, choices=[1, 2], default=1,
        help="Which construction to benchmark (default: 1).",
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="Number of parties (default: 2).",
    )
    parser.add_argument(
        "--attempts", type=int, default=10,
        help="Number of trials per parameter set (default: 10).",
    )
    args = parser.parse_args()

    if args.construction == 1:
        benchmark_construction1(k_parties=args.k, attempts=args.attempts)
    else:
        benchmark_construction2(k_parties=args.k, attempts=args.attempts)
