"""
construction2.py — Multi-Key FHE from RLWR  (Construction 2)
=============================================================

Implements the multi-key fully homomorphic encryption scheme obtained by
lifting our own Ring-LWR single-key FHE to the multi-key setting.

Ring structure:  R = Z[X] / (X^N + 1),  modulus chain q ≥ p₁ ≥ p₂.

Key hierarchy
-------------
  - Each party i holds  sk_i ∈ {-1, 0, 1}^N  (ternary secret).
  - The public key for party i is  (a, b_i)  where
        b_i  =  −⌊(p₁/q) · a · s_i⌋_{p₁}.
  - Extended public key (EPK) sums the b_i components mod p₁.
  - Multi-party relinearisation key encodes ∑ s_i² over the q-gadget.

Ciphertext format:  (c₀, c₁)  ∈  R_{p₂} × R_{p₂}.

Decryption:
    sum_p_i = ∑ (c₁ · s_i + e_sm)  mod p₂
    noisy_m = (c₀ + sum_p_i)  mod p₂
    m       = ⌊(t/p₂) · noisy_m⌋  mod t

References
----------
[This work]  Construction 2 from RLWR
"""

from __future__ import annotations

import math
from math import ceil, log2
from random import SystemRandom
from typing import Tuple

import numpy as np

from poly_utils import mod_vec, round_vec, int2base, poly_add, poly_mul



Poly       = np.ndarray
PublicKey  = Tuple[Poly, Poly]
SecretKey  = Poly
RelinKey   = list
Ciphertext = Tuple[Poly, Poly]     # (c₀, c₁)  both in R_{p₂}


# ---------------------------------------------------------------------------
# MKFHE_Construction2
# ---------------------------------------------------------------------------

class MKFHE_Construction2:
    """
    Multi-Key Fully Homomorphic Encryption over RLWR  (Construction 2).

    Parameters
    ----------
    N  : ring degree  (must be a power of two)
    q  : outermost modulus
    p1 : intermediate modulus  (p1 < q)
    p2 : innermost modulus  (p2 < p1)
    t  : plaintext modulus

    Example
    -------
    >>> scheme = MKFHE_Construction2(N=2**13, q=2**209, p1=2**205, p2=2**201, t=3)
    >>> sk_list, pk_list = scheme.keygen_multiparty(k=2)
    >>> epk = scheme.key_extension(pk_list)
    >>> rkey = scheme.relinkey_gen_multiparty(sk_list)
    >>> msg = scheme.random_message()
    >>> ct = scheme.encrypt(msg, epk)
    >>> dec = scheme.decrypt(sk_list, ct)
    >>> import numpy as np; assert np.array_equal(msg, dec)
    """

    def __init__(self, N: int, q: int, p1: int, p2: int, t: int = 3) -> None:
        self.N     = N
        self.q     = q
        self.p1    = p1
        self.p2    = p2
        self.t     = t
        self.Delta = p2 // t
        self._rng  = SystemRandom()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _uniform(self, lo: int, hi: int) -> Poly:
        """Uniform polynomial in [lo, hi]^N via a cryptographic RNG."""
        return np.array(
            [self._rng.randrange(hi - lo + 1) + lo for _ in range(self.N)],
            dtype=object,
        )

    def _poly_mul(self, a: Poly, b: Poly) -> Poly:
        return poly_mul(a, b, self.N)

    # -----------------------------------------------------------------------
    # Key generation
    # -----------------------------------------------------------------------

    def keygen_multiparty(self, k: int) -> Tuple[list, list]:
        """
        Generate key pairs for *k* parties sharing a common *a* from R_q.

        For party i:
            b_i  =  −⌊(p₁/q) · a · s_i⌋_{p₁}   mod p₁

        Parameters
        ----------
        k : number of parties

        Returns
        -------
        (sk_list, pk_list)
        """
        a = np.array(self._uniform(0, self.q - 1), dtype=object)
        sk_list, pk_list = [], []

        for _ in range(k):
            sk = np.array(self._uniform(-1, 1), dtype=object)
            b  = round_vec(-1 * self._poly_mul(a, sk), self.p1, self.q) % self.p1
            sk_list.append(sk)
            pk_list.append((a, b))

        return sk_list, pk_list

    def key_extension(self, pk_list: list) -> PublicKey:
        """
        Aggregate individual public keys.

        b_ext  =  ∑ b_i  mod p₁

        Parameters
        ----------
        pk_list : list of individual public keys

        Returns
        -------
        epk : (a, b_ext)
        """
        a = pk_list[0][0]
        b = pk_list[0][1]
        for i in range(1, len(pk_list)):
            b = (b + pk_list[i][1]) % self.p1
        return (a, b)

    def relinkey_gen_multiparty(self, sk_list: list, base: int = 2) -> RelinKey:
        """
        Distributed relinearisation key generation (two-round protocol).

        Round 1: each party i samples u_i and publishes h0_i, h1_i.
        Round 2: each party i publishes hp0_i, hp1_i using u_i − s_i.

        Parameters
        ----------
        sk_list : list of secret keys
        base    : gadget decomposition base  (default 2)

        Returns
        -------
        rkey : list of l pairs (r0_j, r1_j), where l = ⌈log_base(q)⌉
        """
        k_parties = len(sk_list)
        l         = ceil(log2(self.q) / log2(base))
        w_vec     = [base ** j for j in range(l)]
        a_r       = [np.array(self._uniform(0, self.q - 1), dtype=object) for _ in range(l)]

        # ---- Round 1 -------------------------------------------------------
        u_list, h0_i_list, h1_i_list = [], [], []

        for s_i in sk_list:
            u_i = np.array(self._uniform(-1, 1), dtype=object)
            u_list.append(u_i)
            h0_i, h1_i = [], []

            for j in range(l):
                term1_0 = round_vec(-1 * self._poly_mul(u_i, a_r[j]), self.p1, self.q) % self.p1
                term2_0 = round_vec(s_i * w_vec[j], self.p1, self.p2) % self.p1
                h0_i.append((term1_0 + term2_0) % self.p1)

                term1_1 = round_vec(self._poly_mul(s_i, a_r[j]), self.p1, self.q) % self.p1
                h1_i.append(term1_1)

            h0_i_list.append(h0_i)
            h1_i_list.append(h1_i)

        h0 = np.sum(h0_i_list, axis=0) % self.p1
        h1 = np.sum(h1_i_list, axis=0) % self.p1

        # ---- Round 2 -------------------------------------------------------
        hp0_i_list, hp1_i_list = [], []

        for s_i, u_i in zip(sk_list, u_list):
            u_minus_s = u_i - s_i          # NOTE: u - s (not s - u)
            hp0_i, hp1_i = [], []

            for j in range(l):
                hp0_val = round_vec(self._poly_mul(s_i, h0[j]), self.p2, self.p1) % self.p2
                hp0_i.append(hp0_val)

                hp1_val = round_vec(self._poly_mul(u_minus_s, h1[j]), self.p2, self.p1) % self.p2
                hp1_i.append(hp1_val)

            hp0_i_list.append(hp0_i)
            hp1_i_list.append(hp1_i)

        hp0 = np.sum(hp0_i_list, axis=0) % self.p2
        hp1 = np.sum(hp1_i_list, axis=0) % self.p2

        r0 = (hp0 + hp1) % self.p2
        r1 = h1

        return list(zip(r0, r1))

    # -----------------------------------------------------------------------
    # Encryption / Decryption
    # -----------------------------------------------------------------------

    def encrypt(self, message: Poly, epk: PublicKey) -> Ciphertext:
        """
        Encrypt *message* under the extended public key.

        c₀ = ⌊(p₂/p₁) · b · rnd⌋_{p₂}  +  Δ·m   mod p₂
        c₁ = ⌊(p₂/q)  · a · rnd⌋_{p₂}              mod p₂

        Parameters
        ----------
        message : polynomial in R_t
        epk     : extended public key (a, b)

        Returns
        -------
        ct : (c₀, c₁) ∈ R_{p₂} × R_{p₂}
        """
        a, b = epk
        rnd  = self._uniform(-1, 1)
        c0   = round_vec(self._poly_mul(b, rnd), self.p2, self.p1) % self.p2
        encoded_message = self.Delta * np.array(message, dtype=object)
        c0   = poly_add(c0, encoded_message) % self.p2
        c1   = round_vec(self._poly_mul(a, rnd), self.p2, self.q) % self.p2
        return (c0, c1)

    def decrypt(self, sk_list: list, ct: Ciphertext, lamda: int = 128) -> Poly:
        """
        Threshold decryption with smudging noise.

        Each party computes  p_i = c₁ · s_i + e_sm  (mod p₂).
        Final message:  m = ⌊(t/p₂) · (c₀ + ∑ p_i)⌋_t

        Parameters
        ----------
        sk_list : list of all parties' secret keys
        ct      : ciphertext (c₀, c₁)
        lamda   : smudging parameter (128 + circuit_error_magnitude)

        Returns
        -------
        Poly — plaintext in R_t
        """
        c0, c1    = ct
        k_parties = len(sk_list)

        
        sum_p_i = np.array(self._uniform(0, 0), dtype=object)

        # Partial decryption — multiply against c₁ (not c₀)
        for s_i in sk_list:
            e_sm    = np.array(self._uniform(0, 2**lamda), dtype=object)
            mul_term = self._poly_mul(c1, s_i) % self.p2
            p_i     = (mul_term + e_sm) % self.p2
            sum_p_i = (sum_p_i + p_i) % self.p2

        # Final decryption — add to c₀ (not subtract)
        noisy_m = (c0 + sum_p_i) % self.p2
        m       = round_vec(noisy_m, self.t, self.p2) % self.t
        return m

    # -----------------------------------------------------------------------
    # Homomorphic operations
    # -----------------------------------------------------------------------

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """
        Homomorphic addition, both components mod p₂.

        Parameters
        ----------
        ct1, ct2 : input ciphertexts

        Returns
        -------
        ct_sum : Ciphertext
        """
        return (
            (ct1[0] + ct2[0]) % self.p2,
            (ct1[1] + ct2[1]) % self.p2,
        )

    def multiply(self, ct1: Ciphertext, ct2: Ciphertext, rkey: RelinKey) -> Ciphertext:
        """
        Homomorphic multiplication with relinearisation.

        Three-term tensor product, gadget-decomposes c₂ (the c₁·c₁' term)
        over the q-sized basis, and folds back using the relinearisation key.

        The output format is (v, w) ∈ R_{p₂} × R_{p₂}.

        Parameters
        ----------
        ct1, ct2 : input ciphertexts
        rkey     : multi-party relinearisation key

        Returns
        -------
        ct_prod : (v, w) ∈ R_{p₂} × R_{p₂}
        """
        base = 2
        l    = ceil(log2(self.q) / log2(base))

        c0 = round_vec(self._poly_mul(ct1[0], ct2[0]), self.t, self.p2) % self.p2
        c1 = round_vec(
            self._poly_mul(ct1[0], ct2[1]) + self._poly_mul(ct1[1], ct2[0]),
            self.t, self.p2,
        ) % self.p2
        c2 = round_vec(self._poly_mul(ct1[1], ct2[1]), self.t, self.p2) % self.p2

        # Gadget decomposition of c2
        decomposed_c2 = np.zeros((self.N, l), dtype=object)
        for i in range(self.N):
            into_base = int2base(int(c2[i]), base)
            decomposed_c2[i] = np.array(
                into_base + [0] * (l - len(into_base)), dtype=object
            )

        # Relinearisation
        v = c0
        w = np.array([0] * self.N, dtype=object)
        for j in range(l):
            v = (v + self._poly_mul(rkey[j][0], decomposed_c2[:, j])) % self.p2
            w = (w + self._poly_mul(rkey[j][1], decomposed_c2[:, j])) % self.p1

        w = (c1 + round_vec(w, self.p2, self.p1)) % self.p2
        return (v, w)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def random_message(self) -> Poly:
        """Sample a uniformly random plaintext in R_t = (Z_t)^N."""
        return np.array(self._uniform(-1, 1), dtype=object) % self.t

    def __repr__(self) -> str:
        return (
            f"MKFHE_Construction2("
            f"N=2^{int(math.log2(self.N))}, "
            f"log(q)={int(math.log2(self.q))}, "
            f"log(p1)={int(math.log2(self.p1))}, "
            f"log(p2)={int(math.log2(self.p2))}, "
            f"t={self.t})"
        )
