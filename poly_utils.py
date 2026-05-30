"""
poly_utils.py — Ring Arithmetic for RLWR-Based MKFHE
=====================================================

Polynomial operations over Z_q[X] / (X^N + 1), including coefficient-wise
reduction, base decomposition, and NTT-accelerated multiplication.

References:
    https://github.com/rtitiu/polymul-approx-ffts

"""

from __future__ import annotations

import math
from math import ceil, log2

import numpy as np
import scipy.fft


# ---------------------------------------------------------------------------
# Centred modular reduction
# ---------------------------------------------------------------------------

def mod_vec(x_vec: np.ndarray, modulus: int) -> np.ndarray:
    """
    Reduce each coefficient of *x_vec* into the centred interval
    (-modulus/2, modulus/2].

    Internally we use a bitmask trick (valid for power-of-two moduli) to
    obtain the positive representative and then fold around the midpoint.

    Parameters
    ----------
    x_vec   : integer array (any dtype)
    modulus : positive integer; must be a power of two

    Returns
    -------
    np.ndarray[object]  centred representatives
    """
    x_vec = np.array([int(i) for i in x_vec], dtype=object)
    positive_r = np.bitwise_and(x_vec, modulus - 1)
    r_vec = positive_r - (
        modulus
        * np.round(np.divide(positive_r, modulus).astype(float)).astype(int)
    )
    return r_vec


# ---------------------------------------------------------------------------
# Rounding (the LWR rounding map)
# ---------------------------------------------------------------------------

def round_vec(vec_x: np.ndarray, pp: int, qq: int) -> np.ndarray:
    """
    Apply the LWR rounding map  x  ↦  ⌊ x · pp / qq ⌉  coefficient-wise.

    Parameters
    ----------
    vec_x : integer array
    pp    : target modulus
    qq    : source modulus  (pp < qq)

    Returns
    -------
    np.ndarray[object]
    """
    vec_x = np.array(vec_x)
    return (2 * pp * vec_x + qq) // (2 * qq)


# ---------------------------------------------------------------------------
# Base decomposition (for gadget / relinearisation key generation)
# ---------------------------------------------------------------------------

def poly_base_decomposition(f: np.ndarray, B: int) -> np.ndarray:
    """
    Decompose every coefficient of *f* in base *B*, yielding a 2-D array
    of shape (N, k) where k = ⌈log_B(2 · max|f| + 1)⌉.

    Parameters
    ----------
    f : degree-N polynomial (integer array)
    B : decomposition base

    Returns
    -------
    decomposed_f : np.ndarray of shape (N, k), dtype=object
    """
    N = len(f)
    f = np.array(f, dtype=object)
    k = ceil(log2(2 * int(max(f)) + 1) / log2(B))
    decomposed_f = np.zeros((N, k), dtype=object)
    for j in range(k):
        r = mod_vec(f, B)
        decomposed_f[:, j] = r
        f = (f - r) // B
    return decomposed_f


def int2base(n: int, b: int) -> list[int]:
    """
    Non-recursive base-*b* decomposition of a non-negative integer *n*.

    Parameters
    ----------
    n : non-negative integer
    b : base (>= 2)

    Returns
    -------
    list of digits, least-significant first
    """
    if n < b:
        return [n]
    return [n % b] + int2base(n // b, b)


# ---------------------------------------------------------------------------
# FFT-based polynomial multiplication over Z[X] / (X^N + 1)
# ---------------------------------------------------------------------------
#
#
#   1. Base-decompose each factor so that intermediate FFT values stay in
#      the range where double-precision rounding errors are harmless.
#   2. Compute the full-length product in the frequency domain.
#   3. Reduce modulo X^N + 1 by negacyclic folding.
#
# Reference: https://github.com/rtitiu/polymul-approx-ffts

_FFT_BASE = 2**19  # safe split base for double-precision FFT


def _fast_mult(
    decomposed_f: np.ndarray,
    decomposed_g: np.ndarray,
    B: int = _FFT_BASE,
) -> np.ndarray:
    """Low-level FFT multiplication of two base-decomposed polynomials."""
    N = decomposed_f.shape[0]
    k_f = decomposed_f.shape[1]
    k_g = decomposed_g.shape[1]
    decomposed_fg = np.zeros((N + 1, k_f + k_g - 1), dtype="complex128")

    for i in range(k_f):
        for j in range(k_g):
            decomposed_fg[:, i + j] += scipy.fft.rfft(
                decomposed_f[:, i], 2 * N
            ) * scipy.fft.rfft(decomposed_g[:, j], 2 * N)

    fg_recovered = np.array([0] * (2 * N), dtype=object)
    for j in reversed(range(k_f + k_g - 1)):
        rounded_term = scipy.fft.irfft(decomposed_fg[:, j], 2 * N)
        rounded_term = np.round(rounded_term.real).astype(int)
        fg_recovered *= B
        fg_recovered += rounded_term
    return fg_recovered


def fast_poly_mult(f: np.ndarray, g: np.ndarray, base: int = _FFT_BASE) -> np.ndarray:
    """
    Multiply two integer polynomials using the split-FFT technique.

    Returns the coefficient array of f · g in Z[X] (unreduced; caller must
    apply negacyclic folding and modular reduction).

    Parameters
    ----------
    f, g : coefficient arrays (length N each)
    base : split base for FFT stability  (default 2^19)

    Returns
    -------
    np.ndarray of length 2N, dtype=object
    """
    f = np.array(f, dtype=object)
    g = np.array(g, dtype=object)
    return _fast_mult(
        poly_base_decomposition(f, base),
        poly_base_decomposition(g, base),
        base,
    )


# ---------------------------------------------------------------------------
# Ring polynomial operations  (reduction modulo X^N + 1)
# ---------------------------------------------------------------------------

def poly_add(
    p1: np.ndarray,
    p2: np.ndarray,
    modulus: int | None = None,
) -> np.ndarray:
    """
    Coefficient-wise addition of two degree-(N-1) polynomials.

    Parameters
    ----------
    p1, p2  : integer arrays of equal length
    modulus : optional integer modulus

    Returns
    -------
    np.ndarray[object]
    """
    p1 = np.array(p1, dtype=object)
    p2 = np.array(p2, dtype=object)
    result = p1 + p2
    if modulus is None:
        return result
    return np.array([x % modulus for x in result], dtype=object)


def poly_mul(p1: np.ndarray, p2: np.ndarray, N: int) -> np.ndarray:
    """
    Multiply two polynomials in Z[X] / (X^N + 1) using the negacyclic
    convolution identity:  X^N ≡ −1.

    Parameters
    ----------
    p1, p2 : coefficient arrays of length N
    N      : ring degree (must equal len(p1) == len(p2))

    Returns
    -------
    np.ndarray[object] of length N
    """
    p1 = np.array(p1, dtype=object)
    p2 = np.array(p2, dtype=object)
    product = fast_poly_mult(p1, p2)
    product = np.concatenate((product, [0] * (2 * N - len(product))), None)
    return np.array(
        [int(product[i]) - int(product[i + N]) for i in range(N)],
        dtype=object,
    )
