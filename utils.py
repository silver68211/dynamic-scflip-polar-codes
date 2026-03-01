

from __future__ import annotations

import os
import textwrap
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches



import numbers
import warnings
from pathlib import Path
from typing import Tuple, List, Optional


from scipy.special import comb
from numpy.core.numerictypes import issubdtype


# -----------------------------
# Matplotlib style (set once)
# -----------------------------
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Times New Roman",
    "text.usetex": False,
})


# -----------------------------
# 5G reliability order loader (cached)
# -----------------------------
@lru_cache(maxsize=1)
def _load_5g_channel_order(path: str = "codes/polar_5G.csv") -> np.ndarray:
    """Load 5G channel reliability order from CSV once."""
    ch_order = np.genfromtxt(path, delimiter=";").astype(np.int32)
    # Sort by reliability ranking (column 1), keep first N later
    ch_order = ch_order[np.argsort(ch_order[:, 1])]
    return ch_order


def generate_5g_ranking(k: int, n: int, sort: bool = True,
                        path: str = "codes/polar_5G.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns frozen and information bit positions based on 5G ranking table.

    Parameters
    ----------
    k : int
        Number of information bits (including CRC if used).
    n : int
        Codeword length (power of two, 32 <= n <= 1024).
    sort : bool
        If True, return indices sorted in ascending order.
    path : str
        CSV path for the 5G ranking table.

    Returns
    -------
    frozen_pos : np.ndarray, shape (n-k,)
    info_pos   : np.ndarray, shape (k,)
    """
    if not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k and n must be integers.")
    if not isinstance(sort, bool):
        raise TypeError("sort must be bool.")
    if k < 0 or k > 1024:
        raise ValueError("k must be in [0, 1024].")
    if n < 32 or n > 1024:
        raise ValueError("n must be in [32, 1024].")
    if n < k:
        raise ValueError("Invalid coderate: n < k.")
    if int(np.log2(n)) != np.log2(n):
        raise ValueError("n must be a power of 2.")

    ch_order = _load_5g_channel_order(path)

    # restrict to first n channels (already sorted by reliability)
    ch_n = ch_order[:n]

    # sort by bit-index (col 0) to align indices; then select frozen/info by reliability col1
    ch_n = ch_n[np.argsort(ch_n[:, 0])]

    frozen_pos = ch_n[: n - k, 1].astype(np.int32)
    info_pos = ch_n[n - k :, 1].astype(np.int32)

    if sort:
        frozen_pos = np.sort(frozen_pos)
        info_pos = np.sort(info_pos)

    return frozen_pos, info_pos


# -----------------------------
# Plot FER curves (efficient)
# -----------------------------
def plot_graph(
    fers: Dict[str, np.ndarray],
    title: str,
    xlabel: str = r"$E_b/N_0$ (dB)",
    ylabel: str = "BLER",
    loc: str = "lower left",
    marker: Optional[Sequence[str]] = None,
    color: Optional[Sequence[str]] = None,
    linestyle: Optional[Sequence[str]] = None,
    markevery: int = 1,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
    figsize: Tuple[float, float] = (6, 8),
    save: bool = False,
):
    if color is None:
        color = ["black", "blue", "green", "red", "purple", "orange"]
    if linestyle is None:
        linestyle = ["-", "--", "--", "--", "--", "--"]
    if marker is None:
        marker = ["o"] * 6

    plt.figure(figsize=figsize)

    for (label, arr), c, ls, mk in zip(fers.items(), color, linestyle, marker):
        # expected arr shape: [num_points, 2] -> [x, y]
        plt.semilogy(
            arr[:, 0],
            arr[:, 1],
            marker=mk,
            color=c,
            linestyle=ls,
            markevery=markevery,
            label=label,
            markerfacecolor="white",
            linewidth=1.8,
            markersize=9,
            markeredgewidth=2,
        )

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    if xticks is not None:
        plt.xticks(xticks, fontsize=16)
    else:
        plt.xticks(fontsize=16)

    if yticks is not None:
        plt.yticks(yticks, fontsize=16)
    else:
        plt.yticks(fontsize=16)

    plt.grid(which="major", linestyle="-.", linewidth=0.8, color="gray", alpha=0.8)
    plt.minorticks_on()
    plt.grid(which="minor", axis="y", linestyle=":", linewidth=0.4, color="gray", alpha=0.2)
    plt.tick_params(axis="both", which="major", direction="in", length=6, width=1)
    plt.tick_params(axis="both", which="minor", direction="in", length=1.5, width=0.5)

    plt.legend(loc=loc, fontsize=14)
    plt.tight_layout()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save:
        os.makedirs("figs", exist_ok=True)
        plt.savefig(f"figs/{title}.pdf", dpi=300, bbox_inches="tight")


# -----------------------------
# Efficient bitmask image plot (ONE function only)
# -----------------------------
def plot_construction_bitmask_image(
    constructions: Dict[str, np.ndarray],
    dec: str = "SC",
    N: int = 64,
    height: int = 16,
    out_dir: str = "figs",
):
    """
    Plot polar code constructions as stacked subplots, each rendered as a grid image.

    Parameters
    ----------
    constructions : dict[label -> mask]
        mask is length-N array with 1 for frozen (or vice versa as you prefer).
    dec : str
        Used in output file name.
    N : int
        Code length.
    height : int
        Number of rows in the grid. Width is N//height.
    out_dir : str
        Directory to save figure.
    """
    if N % height != 0:
        raise ValueError(f"N must be divisible by height. Got N={N}, height={height}.")

    num_constructions = len(constructions)
    width = N // height

    # Prebuild colormaps once (fast)
    base_colors = ["b", "red", "purple", "m", "orange", "green"]
    cmaps = [
        mcolors.LinearSegmentedColormap.from_list(f"white_{c}", ["white", c], N=2)
        for c in base_colors
    ]

    # Figure size tuned so pixels remain distinguishable
    pixel_size = 0.11
    fig_height = pixel_size * height * num_constructions + 1.0
    fig_width = pixel_size * width + 1.0

    fig, axes = plt.subplots(num_constructions, 1, figsize=(fig_width, fig_height), sharex=True)
    if num_constructions == 1:
        axes = [axes]

    legend_handles = []

    for i, (ax, (label, mask)) in enumerate(zip(axes, constructions.items())):
        mask = np.asarray(mask).astype(np.int32).reshape(-1)
        if mask.size != N:
            raise ValueError(f"Mask for '{label}' must have length N={N}, got {mask.size}.")

        # Reshape to [height, width]
        grid = mask.reshape(width, height).T  # shape [height, width]
        cmap = cmaps[i % len(cmaps)]

        ax.imshow(grid, cmap=cmap, aspect="equal", interpolation="nearest", origin="lower")

        # Grid lines (cheap)
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-.", linewidth=0.2)
        ax.tick_params(which="both", length=0)

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Left labels
        ax.set_yticks([0, height - 1])
        ax.set_yticklabels([height, 1], fontsize=14)

        # Right labels
        sec = ax.secondary_yaxis("right")
        sec.set_yticks([0, height - 1])
        sec.set_yticklabels([f"{N}", f"{N - height}"], fontsize=14)
        sec.tick_params(length=0)

        legend_handles.append(
            mpatches.Patch(
                facecolor=cmap(1.0),
                edgecolor="black",
                label=textwrap.fill(label, width=18),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=1.0,
        handleheight=1.0,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bitmask_{N}_{dec}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path




# ============================================================
# 5G Polar ranking utilities
# ============================================================

def generate_5g_ranking(
    k: int,
    n: int,
    sort: bool = True,
    csv_path: str | Path = "codes/polar_5G.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return frozen and information bit positions for the 5G polar reliability order.

    Parameters
    ----------
    k : int
        Number of information bits.
    n : int
        Code length (must be power of 2, 32 <= n <= 1024).
    sort : bool
        If True, returned indices are sorted ascending.
    csv_path : str | Path
        Path to the 5G reliability order CSV (semicolon-separated).

    Returns
    -------
    frozen_pos : np.ndarray, shape (n-k,)
        Frozen bit indices.
    info_pos : np.ndarray, shape (k,)
        Information bit indices.
    """
    if not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k and n must be integers.")
    if not isinstance(sort, bool):
        raise TypeError("sort must be bool.")
    if k < 0:
        raise ValueError("k cannot be negative.")
    if n < 32:
        raise ValueError("n must be >= 32.")
    if n > 1024 or k > 1024:
        raise ValueError("k and n must be <= 1024.")
    if n < k:
        raise ValueError("Invalid code rate (>1). Require n >= k.")
    if int(np.log2(n)) != np.log2(n):
        raise ValueError("n must be a power of 2.")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"5G ranking CSV not found: {csv_path}")

    ch_order = np.genfromtxt(csv_path, delimiter=";").astype(int)
    if ch_order.ndim != 2 or ch_order.shape[1] < 2:
        raise ValueError("Invalid CSV format. Expected at least 2 columns.")

    # Column 1: some index, Column 2: channel position (per your original code)
    # Step 1: sort by channel position column (col=1)
    ch_sorted = ch_order[np.argsort(ch_order[:, 1])]

    # Step 2: keep first n rows (positions relevant to length n)
    ch_n = ch_sorted[:n, :]

    # Step 3: sort these again by reliability (col=0)
    ch_n = ch_n[np.argsort(ch_n[:, 0])]

    # Split frozen/info positions
    frozen_pos = ch_n[: (n - k), 1].astype(int)
    info_pos = ch_n[(n - k) :, 1].astype(int)

    if sort:
        frozen_pos = np.sort(frozen_pos)
        info_pos = np.sort(info_pos)

    return frozen_pos, info_pos


# ============================================================
# Polar transform matrix
# ============================================================

def generate_polar_transform_mat(n_lift: int) -> np.ndarray:
    """
    Generate the polar transform matrix G_N = F^{\\otimes n_lift}, where
    F = [[1, 0], [1, 1]] and N = 2^{n_lift}.

    Parameters
    ----------
    n_lift : int
        Kronecker power (>=0). Output size is N x N with N=2^n_lift.

    Returns
    -------
    gm : np.ndarray, shape (2^n_lift, 2^n_lift)
        Polar transformation matrix over GF(2) represented with {0,1}.
    """
    if int(n_lift) != n_lift:
        raise TypeError("n_lift must be an integer.")
    n_lift = int(n_lift)
    if n_lift < 0:
        raise ValueError("n_lift must be >= 0.")
    if n_lift >= 12:
        warnings.warn("n_lift >= 12 leads to very large matrices (N >= 4096).")

    F = np.array([[1, 0], [1, 1]], dtype=np.int8)

    gm = np.array([[1]], dtype=np.int8)
    for _ in range(n_lift):
        gm = np.kron(gm, F)  # much faster/cleaner than nested loops
    return gm.astype(np.int8)


# ============================================================
# Reed-Muller (RM) construction helper
# ============================================================

def generate_rm_code(r: int, m: int):
    """
    Generate frozen and information positions for RM(r,m).

    Frozen positions correspond to indices whose binary Hamming weight < (m-r).

    Parameters
    ----------
    r : int
        RM order.
    m : int
        log2(N) where N is code length.

    Returns
    -------
    frozen_pos, info_pos, n, k, d_min
    """
    if not isinstance(r, int) or not isinstance(m, int):
        raise TypeError("r and m must be integers.")
    if r < 0 or m < 0:
        raise ValueError("r and m must be non-negative.")
    if r > m:
        raise ValueError("order r cannot be larger than m.")

    n = 2**m
    d_min = 2 ** (m - r)

    # expected dimension k
    k = int(sum(comb(m, i) for i in range(r + 1)))

    # Vectorized Hamming weights for indices 0..n-1
    # Represent indices as uint16/uint32 depending on m
    dtype = np.uint16 if m <= 16 else np.uint32
    idx = np.arange(n, dtype=dtype)

    # unpackbits works on uint8, so view as bytes and sum bits
    bytes_view = idx.view(np.uint8).reshape(n, -1)
    w = np.unpackbits(bytes_view, axis=1).sum(axis=1)

    frozen_vec = w < (m - r)
    info_vec = ~frozen_vec

    frozen_pos = np.flatnonzero(frozen_vec).astype(int)
    info_pos = np.flatnonzero(info_vec).astype(int)

    if info_pos.size != k:
        raise RuntimeError(f"Inconsistent RM dimension: got {info_pos.size}, expected {k}.")

    return frozen_pos, info_pos, n, k, d_min


# ============================================================
# Dense polar matrices (naive)
# ============================================================

def generate_dense_polar(frozen_pos: np.ndarray, n: int, verbose: bool = True):
    """
    Generate dense parity-check matrix H and generator matrix G for a polar code.

    Parameters
    ----------
    frozen_pos : np.ndarray of int
        Frozen indices (length n-k).
    n : int
        Code length (power of 2).
    verbose : bool
        If True, prints shapes and plots H sparsity.

    Returns
    -------
    pcm : np.ndarray, shape (n-k, n)
        Parity-check matrix.
    gm_true : np.ndarray, shape (k, n)
        Generator matrix (rows at info positions).
    """
    if not isinstance(n, numbers.Number):
        raise TypeError("n must be a number.")
    n = int(n)
    if int(np.log2(n)) != np.log2(n):
        raise ValueError("n must be a power of 2.")
    if not issubdtype(frozen_pos.dtype, int):
        raise TypeError("frozen_pos must contain integers.")
    if frozen_pos.size > n:
        raise ValueError("len(frozen_pos) cannot be greater than n.")

    k = n - frozen_pos.size
    info_pos = np.setdiff1d(np.arange(n), frozen_pos)
    if info_pos.size != k:
        raise RuntimeError("Internal error: invalid info_pos generated.")

    gm_mat = generate_polar_transform_mat(int(np.log2(n))).astype(int)

    gm_true = gm_mat[info_pos, :]
    pcm = gm_mat[:, frozen_pos].T

    if verbose:
        print("Shape of generator matrix:", gm_true.shape)
        print("Shape of parity-check matrix:", pcm.shape)
        plt.spy(pcm)

    # Verify H*G^T = 0 (mod2)
    s = (pcm @ gm_true.T) & 1
    if np.any(s):
        raise RuntimeError("Non-zero syndrome for H*G^T (invalid pair).")

    return pcm, gm_true


# ============================================================
# Bit/integer conversion helpers
# ============================================================

def bin2int(arr: List[int] | np.ndarray) -> Optional[int]:
    """Convert MSB->LSB binary array/list to integer. Returns None if empty."""
    if len(arr) == 0:
        return None
    return int("".join(str(int(x)) for x in arr), 2)


def bin2int_tf(arr: tf.Tensor) -> tf.Tensor:
    """
    Convert binary tensor (MSB->LSB in last axis) to int tensor.

    Note: arr should be integer type for tf.bitwise ops.
    """
    arr = tf.cast(arr, tf.int64)
    length = tf.shape(arr)[-1]
    shifts = tf.range(length - 1, -1, -1, dtype=tf.int64)
    return tf.reduce_sum(tf.bitwise.left_shift(arr, shifts), axis=-1)


def int2bin(num: int, len_: int) -> List[int]:
    """Convert integer to MSB->LSB list of length len_."""
    if num < 0:
        raise ValueError("num must be non-negative.")
    if len_ < 0:
        raise ValueError("len_ must be non-negative.")
    if len_ == 0:
        return []
    return [int(x) for x in format(num, f"0{len_}b")[-len_:]]


def int2bin_tf(ints: tf.Tensor, len_: int) -> tf.Tensor:
    """Convert int tensor to bits (MSB->LSB) in last axis of length len_."""
    if len_ < 0:
        raise ValueError("len_ must be non-negative.")
    ints = tf.cast(ints, tf.int64)
    shifts = tf.range(len_ - 1, -1, -1, dtype=tf.int64)
    bits = tf.bitwise.right_shift(tf.expand_dims(ints, -1), shifts) & 1
    return tf.cast(bits, tf.int64)


# ============================================================
# Alist and matrix conversions (kept mostly as-is)
# ============================================================

def alist2mat(alist, verbose=True):
    # (Your original body unchanged except for style improvements would be long)
    # Keeping your implementation as-is is fine; it's already reasonable.
    # If you want, I can refactor it too (type hints, faster checks, etc.).
    assert len(alist) > 4, "Invalid alist format."

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]
    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    assert np.sum(vn_profile) == np.sum(cn_profile), "Invalid alist format."
    assert np.max(vn_profile) == v_max, "Invalid alist format."
    assert np.max(cn_profile) == c_max, "Invalid alist format."

    vn_only = len(alist) == len(vn_profile) + 4
    if vn_only and verbose:
        print("Note: .alist does not contain CN perspective. Recovering from VN only.")

    pcm = np.zeros((m, n), dtype=np.int8)
    num_edges = 0

    for idx_v in range(n):
        deg = vn_profile[idx_v]
        for idx_i in range(deg):
            idx_c = alist[4 + idx_v][idx_i] - 1
            pcm[idx_c, idx_v] = 1
            num_edges += 1

    if not vn_only:
        for idx_c in range(m):
            deg = cn_profile[idx_c]
            for idx_i in range(deg):
                idx_v = alist[4 + n + idx_c][idx_i] - 1
                assert pcm[idx_c, idx_v] == 1

    if verbose:
        print("n:", n, "m:", m, "k:", k, "rate:", coderate)
        print("edges:", num_edges, "v_max:", v_max, "c_max:", c_max)
        plt.spy(pcm)

    return pcm.astype(int), k, n, coderate


def load_alist(path: str) -> list:
    alist = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            row = [int(word) for word in line.split()]
            if row:
                alist.append(row)
    return alist


# ============================================================
# Fast modulo-2 for TF
# ============================================================

def int_mod_2(x: tf.Tensor) -> tf.Tensor:
    """
    Efficient modulo-2 for integer tensors using bitwise AND.

    Notes
    -----
    - Works best when x is integer type.
    - If x is float, it will be cast to int64 first.
    """
    if not x.dtype.is_integer:
        x = tf.cast(x, tf.int64)
    # AND with 1 gives modulo 2
    y = tf.bitwise.bitwise_and(x, tf.constant(1, x.dtype))
    return y


# ============================================================
# Bhattacharyya parameter recursion (memoized)
# ============================================================

@lru_cache(maxsize=None)
def bhattacharyya_parameter(i: int, N: int, alpha: float) -> float:
    """
    Compute Bhattacharyya parameter Z(W_N^{(i)}) recursively for BEC-like recursion.

    Parameters
    ----------
    i : int
        1-based bit-channel index (as in many polar references).
    N : int
        Block length (power of 2).
    alpha : float
        Base parameter for N=1.

    Returns
    -------
    float
        Bhattacharyya parameter for (i,N).

    Notes
    -----
    This is memoized to avoid exponential recursion.
    """
    if N == 1:
        if i != 1:
            raise ValueError("For N=1, i must be 1.")
        return float(alpha)

    if N % 2 != 0:
        raise ValueError("N must be power of 2 (even during recursion).")
    if i < 1 or i > N:
        raise ValueError("Require 1 <= i <= N.")

    half = N // 2
    if i == 1:
        z = bhattacharyya_parameter(1, half, alpha)
        return 2 * z - z * z

    if i % 2 == 0:
        z = bhattacharyya_parameter(i // 2, half, alpha)
        return z * z

    z = bhattacharyya_parameter((i + 1) // 2, half, alpha)
    return 2 * z - z * z