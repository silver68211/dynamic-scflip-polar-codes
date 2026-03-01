import numpy as np
from collections import Counter
from typing import Tuple


# -----------------------------
# Gaussian Approximation (GA) construction
# -----------------------------
class GA:
    """
    Gaussian Approximation (GA) construction for polar codes.

    Parameters
    ----------
    snr_db : float
        Design SNR in dB (typically Es/N0 or Eb/N0 depending on your convention).
    n_log2 : int
        Code length exponent, N = 2**n_log2.
    rate : float
        Code rate K/N.
    """

    def __init__(self, snr_db: float, n_log2: int = 7, rate: float = 0.5):
        if not (0.0 < rate <= 1.0):
            raise ValueError("rate must be in (0, 1].")
        if n_log2 < 1:
            raise ValueError("n_log2 must be >= 1.")

        self.n_log2 = int(n_log2)
        self.N = 2 ** self.n_log2
        self.K = int(round(rate * self.N))
        self.rate = float(rate)

        # Design mean LLR for AWGN approximation commonly used in polar GA
        # design_llr = 4 * R * 10^(snr/10)
        self.design_llr = 4.0 * self.rate * (10.0 ** (snr_db / 10.0))

        # Reliability sequence (smaller -> worse channel in your current convention)
        init_mean = np.full(self.N, self.design_llr, dtype=np.float64)
        self.sequence = self.gaussian_approx(init_mean)

        # Sort by reliability (ascending)
        order = np.argsort(self.sequence)
        self.frozen_pos = order[: self.N - self.K]
        self.info_pos = order[self.N - self.K :]

    # ---- phi approximation ----
    @staticmethod
    def phi_new(x: np.ndarray) -> np.ndarray:
        """
        Vectorized phi approximation used in GA construction.
        Uses two regimes (x < 10 and x >= 10).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)

        m = x < 10.0
        # regime 1
        y[m] = np.exp(-0.4527 * (x[m] ** 0.86) + 0.0218)

        # regime 2 (asymptotic)
        xm = x[~m]
        # avoid division by zero
        xm = np.maximum(xm, 1e-12)
        y[~m] = np.sqrt(np.pi / xm) * (1.0 - 10.0 / (7.0 * xm)) * np.exp(-xm / 4.0)

        # clip to (0,1] for numerical stability
        return np.clip(y, 1e-15, 1.0)

    @classmethod
    def phi_inv(cls, y: np.ndarray, lo: float = 0.0, hi: float = 10_000.0, tol: float = 1e-2) -> np.ndarray:
        """
        Vectorized inverse of phi_new via bisection.

        Solves phi_new(x) = y for x >= 0.
        """
        y = np.asarray(y, dtype=np.float64)
        y = np.clip(y, 1e-15, 1.0)

        a = np.full_like(y, lo, dtype=np.float64)
        b = np.full_like(y, hi, dtype=np.float64)

        # bisection loop (vectorized)
        while np.max(b - a) >= tol:
            c = 0.5 * (a + b)
            fc = cls.phi_new(c) - y
            fa = cls.phi_new(a) - y

            # if sign change between a and c => root in [a,c], else in [c,b]
            left = (fc * fa) < 0
            b = np.where(left, c, b)
            a = np.where(left, a, c)

        return 0.5 * (a + b)

    def gaussian_approx(self, init_mean: np.ndarray) -> np.ndarray:
        """
        Compute GA reliabilities (mean LLRs) for all synthetic channels.

        Vectorized O(N log N) version.
        """
        m = np.asarray(init_mean, dtype=np.float64)
        if m.shape != (self.N,):
            raise ValueError(f"init_mean must have shape ({self.N},), got {m.shape}.")

        # Stage-by-stage recursion, vectorized within each stage
        cur = m.copy()
        for stage in range(1, self.n_log2 + 1):
            half = 2 ** (stage - 1)
            size = 2 ** stage

            # We only need previous stage values for indices 0..half-1
            prev = cur[:half]

            # upper: phi^{-1}(1 - (1-phi(T))^2)
            phiT = self.phi_new(prev)
            upper = self.phi_inv(1.0 - (1.0 - phiT) ** 2)

            # lower: 2T
            lower = 2.0 * prev

            # interleave upper/lower into the next stage buffer
            nxt = np.empty(size, dtype=np.float64)
            nxt[0::2] = upper
            nxt[1::2] = lower

            # write back into cur (only first "size" entries are meaningful at this stage)
            cur[:size] = nxt

        return cur[: self.N]


# -----------------------------
# RM / RM-Polar design utilities
# -----------------------------
def bin_rep(n: int) -> np.ndarray:
    """
    Binary representations for integers 0..2^n-1, shape (2^n, n).
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    x = np.arange(2 ** n, dtype=np.uint32)[:, None]
    bits = (x >> np.arange(n, dtype=np.uint32)) & 1
    return bits.astype(np.uint8)


def RM_Design_Rule(N: int, K: int) -> np.ndarray:
    """
    Classical RM design: choose highest Hamming-weight rows.
    (Requires r_gp defined elsewhere in your codebase; kept as-is.)
    """
    n = int(np.log2(N))
    T = [2] * n
    P = r_gp(T)  # <-- depends on your existing implementation
    wt = P.sum(1)
    return np.argsort(wt)[::-1][:K]


def RM_Polar_Design_Rule(N: int, K: int, crc: int = 0, ebno_db: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    RM-Polar selection rule (your variant): filter by RM order then pick best by Bhattacharyya-like metric.

    Returns
    -------
    info_pos : np.ndarray
        Selected information positions, shape (K+crc,)
    zw : np.ndarray
        Z-like metric over all N positions
    """
    n = int(np.log2(N))
    rate = K / N

    ga = GA(ebno_db, n_log2=n, rate=rate)

    # your original computations (kept), but vectorized and stabilized
    sigma = 2.0 / ga.sequence
    sigma = np.maximum(sigma, 1e-12)

    zw = np.exp(-1.0 / (2.0 * sigma))

    # Hamming weight of binary index (RM order filtering)
    bits = bin_rep(n)
    weight = bits.sum(axis=1)

    wt_dict = Counter(weight)
    # sort weights descending
    swt = sorted(wt_dict.items(), key=lambda kv: kv[0], reverse=True)

    rK = 0
    lowest_wt = None
    target = K + crc
    for wt_val, count in swt:
        rK += count
        lowest_wt = wt_val
        if rK >= target:
            break

    remaining_idx = np.where(weight >= lowest_wt)[0]
    remaining_zw = zw[remaining_idx]

    # choose smallest zw (most reliable)
    pick = np.argsort(remaining_zw)[:target]
    info_pos = remaining_idx[pick]

    return info_pos.astype(np.int32), zw.astype(np.float64)


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    N = 1024
    K = 512
    n = int(np.log2(N))

    design_snr_db = 2.0
    rate = K / N

    ga = GA(design_snr_db, n_log2=n, rate=rate)
    sequence = ga.sequence
    order = np.argsort(sequence)

    frozen = order[: N - K]
    info_pos = np.setdiff1d(np.arange(N), frozen)

    print("info_pos:\n", info_pos)
    print("frozen:\n", frozen)