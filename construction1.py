"""
construction1.py — Multi-Key FHE from RLWR  (Construction 1)
=============================================================

Implements the multi-key fully homomorphic encryption scheme obtained by
lifting the Ring-LWR single-key FHE of (ePrint 2024/960) to
the multi-key setting.

Ring structure:  R = Z[X] / (X^N + 1),  with modulus chain r ≥ q ≥ p.

Key hierarchy
-------------
  - Each party i holds  sk_i ∈ {-1, 0, 1}^N  (ternary secret).
  - The public key for party i is  (a, b_i)  where
        b_i  =  ⌊(q/r) · a · s_i⌋_q.
  - An extended public key (EPK) is formed by summing all b_i.
  - A multi-party relinearisation key encodes ∑ s_i² under a gadget
    decomposition, enabling homomorphic multiplication.

Ciphertext format:  (c₀, c₁)  ∈  R_q × R_p.

Decryption:
    scaled_sum  = ⌊(p/q) · ∑ (c₀ · s_i + e_sm)⌋
    noisy_m     = (c₁ − scaled_sum)  mod p
    m           = ⌊(t/p) · noisy_m⌋  mod t

References
----------
 ePrint 2024/960.


"""

from __future__ import annotations

import math
from math import ceil, log2
from random import SystemRandom
from typing import Tuple

import numpy as np

from poly_utils import mod_vec, round_vec, int2base, poly_add, poly_mul
from error import _get_error_LWR1, _get_error_LWR2, _get_error_LWE



Poly       = np.ndarray
PublicKey  = Tuple[Poly, Poly]
SecretKey  = Poly
RelinKey   = list
Ciphertext = Tuple[Poly, Poly]     # (c₀, c₁)


# ---------------------------------------------------------------------------
# MKFHE_Construction1
# ---------------------------------------------------------------------------

class MKFHE_Construction1:
    """
    Multi-Key Fully Homomorphic Encryption over RLWR  (Construction 1).

    Parameters
    ----------
    N  : ring degree  (must be a power of two)
    r  : outermost modulus  (r > q)
    q  : intermediate modulus
    p  : innermost modulus  (p < q)
    t  : plaintext modulus

    Example
    -------
    >>> scheme = MKFHE_Construction1(N=2**13, r=2**206, q=2**202, p=2**198, t=3)
    >>> sk_list, pk_list = scheme.keygen_multiparty(k=2)
    >>> epk = scheme.key_extension(pk_list)
    >>> rkey = scheme.relinkey_gen_multiparty(sk_list)
    >>> msg = scheme.random_message()
    >>> ct = scheme.encrypt(msg, epk)
    >>> dec = scheme.decrypt(sk_list, ct)
    >>> import numpy as np; assert np.array_equal(msg, dec)
    """

    def __init__(self, N: int, r: int, q: int, p: int, t: int = 3) -> None:
        self.N     = N
        self.r     = r
        self.q     = q
        self.p     = p
        self.t     = t
        self.Delta = p // t
        self._rng  = SystemRandom()



    def _uniform(self, lo: int, hi: int) -> Poly:
        """Uniform polynomial in [lo, hi]^N using a cryptographic RNG."""
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
        Generate individual key pairs for *k* parties sharing a common *a*.

        Parameters
        ----------
        k : number of parties

        Returns
        -------
        (sk_list, pk_list)
        """


        a = np.array(self._uniform(0, self.r - 1), dtype=object)
        sk_list, pk_list = [], []

        for _ in range(k):
            sk = np.array(self._uniform(-1, 1), dtype=object)
            b  = round_vec(self._poly_mul(a, sk), self.q, self.r) % self.q
            sk_list.append(sk)
            pk_list.append((a, b))

        return sk_list, pk_list

    def key_extension(self, pk_list: list) -> PublicKey:
        """
        Aggregate individual public keys into an extended public key.

        b_ext = ∑ b_i  mod q

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
            b = (b + pk_list[i][1]) % self.q
        return (a, b)

    def relinkey_gen_multiparty(self, sk_list: list, base: int = 2) -> RelinKey:
        """
        Distributed relinearisation key generation (two-round protocol).

        Parameters
        ----------
        sk_list : list of secret keys
        base    : gadget decomposition base  (default 2)

        Returns
        -------
        rkey : list of l pairs (r0_j, r1_j), where l = ⌈log_base(r)⌉
        """
        k_parties = len(sk_list)
        l         = ceil(log2(self.r) / log2(base))
        w_vec     = [base ** j for j in range(l)]
        a_r       = [np.array(self._uniform(0, self.r - 1), dtype=object) for _ in range(l)]

        # ---- Round 1 -------------------------------------------------------
        u_list, h0_i_list, h1_i_list = [], [], []

        for s_i in sk_list:
            u_i = np.array(self._uniform(-1, 1), dtype=object)
            u_list.append(u_i)
            h0_i, h1_i = [], []

            for j in range(l):
                term1_0 = round_vec(self._poly_mul(u_i, a_r[j]), self.q, self.r) % self.q
                term2_0 = round_vec(s_i * w_vec[j], self.p, self.q)
                h0_i.append((term1_0 + term2_0) % self.q)

                term1_1 = round_vec(self._poly_mul(s_i, a_r[j]), self.q, self.r) % self.q
                h1_i.append(term1_1)

            h0_i_list.append(h0_i)
            h1_i_list.append(h1_i)

        h0 = np.sum(h0_i_list, axis=0) % self.q
        h1 = np.sum(h1_i_list, axis=0) % self.q

        # ---- Round 2 -------------------------------------------------------
        hp0_i_list, hp1_i_list = [], []

        for s_i, u_i in zip(sk_list, u_list):
            s_minus_u = s_i - u_i
            hp0_i, hp1_i = [], []

            for j in range(l):
                hp0_val = round_vec(self._poly_mul(s_i, h0[j]), self.p, self.q)
                hp0_i.append(hp0_val % self.p)

                hp1_val = round_vec(self._poly_mul(s_minus_u, h1[j]), self.p, self.q)
                hp1_i.append(hp1_val % self.p)

            hp0_i_list.append(hp0_i)
            hp1_i_list.append(hp1_i)

        hp0 = np.sum(hp0_i_list, axis=0) % self.p
        hp1 = np.sum(hp1_i_list, axis=0) % self.p

        r0 = (hp0 + hp1) % self.p
        r1 = h1

        return list(zip(r0, r1))

    # -----------------------------------------------------------------------
    # Encryption / Decryption
    # -----------------------------------------------------------------------

    def encrypt(self, message: Poly, epk: PublicKey) -> Ciphertext:
        """
        Encrypt *message* under the extended public key.

        c₀ = ⌊(q/r) · a · rnd⌋_q
        c₁ = ⌊(p/q) · b · rnd⌋_p + Δ·m  mod p

        Parameters
        ----------
        message : polynomial in R_t
        epk     : extended public key (a, b)

        Returns
        -------
        ct : (c₀, c₁) ∈ R_q × R_p
        """
        a, b = epk
        rnd  = self._uniform(-1, 1)
        c0   = round_vec(self._poly_mul(a, rnd), self.q, self.r) % self.q
        c1   = round_vec(self._poly_mul(b, rnd), self.p, self.q) % self.p
        encoded_message = self.Delta * np.array(message, dtype=object)
        c1   = (c1 + encoded_message) % self.p
        return (c0, c1)

    def decrypt(self, sk_list: list, ct: Ciphertext, lamda: int = 128) -> Poly:
        """
        Threshold decryption via partial-decryption aggregation with
        noise smudging.

        Parameters
        ----------
        sk_list : list of all parties' secret keys
        ct      : ciphertext (c₀, c₁)
        lamda   : smudging parameter (128 + circuit_error_magnitude)

        Returns
        -------
        Poly — message in R_t
        """
        c0, c1    = ct
        k_parties = len(sk_list)


        sum_p_i   = np.array(self._uniform(0, 0), dtype=object)

        # Partial decryption
        for s_i in sk_list:
            e_sm    = np.array(self._uniform(0, 2**lamda), dtype=object)
            mul_term = self._poly_mul(c0, s_i) % self.q
            p_i     = (mul_term + e_sm) % self.q
            sum_p_i = (sum_p_i + p_i) % self.q

        # Final decryption
        scaled_sum = round_vec(sum_p_i, self.p, self.q)
        noisy_m    = (c1 - scaled_sum) % self.p
        m          = round_vec(noisy_m, self.t, self.p) % self.t
        return m

    # -----------------------------------------------------------------------
    # Homomorphic operations
    # -----------------------------------------------------------------------

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """
        Homomorphic addition.

        Parameters
        ----------
        ct1, ct2 : input ciphertexts

        Returns
        -------
        ct_sum : Ciphertext
        """
        return (
            poly_add(ct1[0], ct2[0], self.q),
            poly_add(ct1[1], ct2[1], self.p),
        )

    def multiply(self, ct1: Ciphertext, ct2: Ciphertext, rkey: RelinKey) -> Ciphertext:
        """
        Homomorphic multiplication with relinearisation.

        Ciphertext tensor product followed by gadget-decomposition of the
        c₀c₀' term (the R_q × R_q cross-term) and key-switching back to a
        degree-1 ciphertext.

        Ciphertext format is (c₀, c₁) where c₀ ∈ R_q, c₁ ∈ R_p.
        The three-term tensor is:
            c2  =  ⌊(t/p) · c₀ · c₀'⌋  mod r      ← R_q × R_q, decomposed
            c1  =  ⌊(t/p) · (c₀·c₁' + c₁·c₀')⌋   mod q
            c0  =  ⌊(t/p) · c₁ · c₁'⌋              mod p

        The relinearisation key folds c2 back into (c1, c0) via gadget
        decomposition in base *base* with l = ⌈log_base(r)⌉ levels.

        Output: (w, v) — same (R_q, R_p) layout as the inputs.

        Parameters
        ----------
        ct1, ct2 : input ciphertexts  (c₀ ∈ R_q, c₁ ∈ R_p)
        rkey     : multi-party relinearisation key

        Returns
        -------
        ct_prod : (w, v) ∈ R_q × R_p
        """
        base = 2
        l    = ceil(log2(self.r) / log2(base))   # ⌈log_base(r)⌉

        # Three-term tensor product
        # ct[0] = c₀ ∈ R_q,  ct[1] = c₁ ∈ R_p
        c2 = round_vec(self._poly_mul(ct1[0], ct2[0]), self.t, self.p) % self.r   # R_q × R_q → mod r
        c1 = round_vec(
            poly_add(
                self._poly_mul(ct1[0], ct2[1]),
                self._poly_mul(ct1[1], ct2[0]),
            ),
            self.t, self.p,
        ) % self.q                                                                  # cross terms → mod q
        c0 = round_vec(
            self._poly_mul(ct1[1], ct2[1]), self.t, self.p
        ) % self.p                                                                  # R_p × R_p → mod p

        # Gadget decomposition of c2 (the high-modulus R_q × R_q term)
        decomposed_c2 = np.zeros((self.N, l), dtype=object)
        for i in range(self.N):
            into_base = int2base(int(c2[i]), base)
            decomposed_c2[i] = np.array(
                into_base + [0] * (l - len(into_base)), dtype=object
            )

        # Relinearisation: fold c2 back using the relin key.
        # rkey[j] = (r0_j, r1_j) where r0_j ∈ R_p and r1_j = h1_j ∈ R_q.
        # v (R_p accumulator) uses r0_j;  w (R_q accumulator) uses r1_j.
        v = c0   # R_p
        w = c1   # R_q
        for j in range(l):
            v = (v + self._poly_mul(rkey[j][0], decomposed_c2[:, j])) % self.p
            w = (w + self._poly_mul(rkey[j][1], decomposed_c2[:, j])) % self.q

        return (w, v)   # (R_q, R_p) — same layout as input ciphertexts

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def random_message(self) -> Poly:
        """Sample a uniformly random plaintext in R_t = (Z_t)^N."""
        return np.array(self._uniform(-1, 1), dtype=object) % self.t

    def noise_estimate(self, ct: Ciphertext, sk: Poly, msg: Poly) -> Poly:
        """
        Compute p·q·noise for a correctly decrypting single-key ciphertext.
        Useful for debugging noise growth in deep circuits.

        Based on the decryption equation:
            t/p · (−p/q · c₀ · s  +  c₁)  =  msg + noise

        Parameters
        ----------
        ct  : ciphertext
        sk  : single secret key (aggregated ∑ s_i for multi-key)
        msg : known plaintext  (Decrypt(ct) must equal msg)

        Returns
        -------
        pq_noise : scaled noise vector
        """
        c0, c1 = ct
        msg    = np.array(msg, dtype=object)
        c0     = np.array(c0, dtype=object)
        c1     = np.array(c1, dtype=object)
        pq_noise = poly_add(
            self.q * self.t * c1,
            -self.p * self.t * self._poly_mul(c0, sk),
        )
        pq_noise = poly_add(pq_noise, -self.p * self.q * msg)
        return mod_vec(pq_noise, self.p * self.q)

    def __repr__(self) -> str:
        return (
            f"MKFHE_Construction1("
            f"N=2^{int(math.log2(self.N))}, "
            f"log(r)={int(math.log2(self.r))}, "
            f"log(q)={int(math.log2(self.q))}, "
            f"log(p)={int(math.log2(self.p))}, "
            f"t={self.t})"
        )