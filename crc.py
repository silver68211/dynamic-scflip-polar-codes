"""
CRC encoder/decoder layers (generator-matrix based).

- CRCEncoder: appends CRC parity bits to the last axis.
- CRCDecoder: verifies CRC and removes parity bits.

Notes
-----
This implementation builds a CRC generator matrix G_crc (shape [k, crc_len])
so that parity = (u @ G_crc) mod 2, where u are the information bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from utils import int_mod_2  # expects integer tensor -> tensor mod 2


# -----------------------------
# CRC polynomial definitions
# -----------------------------
_CRC_SPECS: Dict[str, List[int]] = {
    "CRC24A":    [24, 23, 18, 17, 14, 11, 10,  7,  6,  5,  4,  3,  1, 0],
    "CRC24B":    [24, 23,  6,  5,  1, 0],
    "CRC24C":    [24, 23, 21, 20, 17, 15, 13, 12,  8,  4,  2,  1, 0],
    "CRC16":     [16, 12,  5, 0],
    "CRC11":     [11, 10,  9,  5, 0],
    "CRC6":      [ 6,  5, 0],
    # your extras:
    "CRC8":      [ 8,  4,  3,  1, 0],
    "CRC8DVBS2": [ 8,  7,  6,  4,  2, 0],
}


def _crc_polynomial_bits(crc_degree: str) -> Tuple[np.ndarray, int]:
    """
    Return CRC polynomial in MSB->LSB order (length crc_len+1), and crc_len.

    Example: CRC16 => length=17 bits.
    """
    if crc_degree not in _CRC_SPECS:
        raise ValueError(
            f"Invalid CRC Polynomial '{crc_degree}'. "
            f"Supported: {sorted(_CRC_SPECS.keys())}"
        )

    coeffs = _CRC_SPECS[crc_degree]
    crc_len = max(coeffs)

    pol = np.zeros(crc_len + 1, dtype=np.int32)
    pol[coeffs] = 1

    # Convert to MSB-first representation
    pol_msb = pol[::-1].copy()
    return pol_msb, crc_len


def _gen_crc_matrix(k: int, pol_msb: np.ndarray) -> np.ndarray:
    """
    Build dense CRC generator matrix G_crc of shape [k, crc_len].

    The method builds the remainders for unit vectors efficiently in O(k * crc_len).
    """
    crc_len = pol_msb.size - 1
    g = np.zeros((k, crc_len), dtype=np.int32)

    # Remainder register, MSB-first
    rem = np.zeros(crc_len, dtype=np.int32)
    rem[0] = 1

    # Construct rows from last unit-vector to first (as in original)
    for i in range(k):
        # shift left by one (append 0)
        rem_ext = np.concatenate([rem, [0]]).astype(np.int32)
        if rem_ext[0] == 1:
            rem_ext = np.bitwise_xor(rem_ext, pol_msb)
        rem = rem_ext[1:]  # drop MSB
        g[k - i - 1, :] = rem

    return g


# -----------------------------
# Layers
# -----------------------------
class CRCEncoder(Layer):
    """
    CRCEncoder(crc_degree, dtype=tf.float32)

    Appends CRC parity bits to the last axis.

    Parameters
    ----------
    crc_degree : str
        One of: {CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6, CRC8, CRC8DVBS2}
    dtype : tf.DType
        Output dtype. Internally uses float32 for matmul and int for mod-2.
    """

    def __init__(self, crc_degree: str, dtype: tf.DType = tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        if not isinstance(crc_degree, str):
            raise TypeError("crc_degree must be a str.")

        self._crc_degree = crc_degree
        self._crc_pol_msb, self._crc_len = _crc_polynomial_bits(crc_degree)

        self._k = None  # info length tracked after build
        self._n = None  # total length = k + crc_len
        self._g_mat_crc = None  # tf.Tensor [k, crc_len], float32

    @property
    def crc_degree(self) -> str:
        return self._crc_degree

    @property
    def crc_length(self) -> int:
        return self._crc_len

    @property
    def crc_pol(self) -> np.ndarray:
        return self._crc_pol_msb

    @property
    def k(self) -> int | None:
        return self._k

    @property
    def n(self) -> int | None:
        return self._n

    def build(self, input_shape):
        k = input_shape[-1]
        if k is None:
            raise ValueError("Last dimension (k) cannot be None for CRCEncoder.")

        k = int(k)
        g_np = _gen_crc_matrix(k, self._crc_pol_msb).astype(np.float32)
        self._g_mat_crc = tf.constant(g_np, dtype=tf.float32)

        self._k = k
        self._n = k + self._crc_len
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # rank >= 2
        tf.debugging.assert_greater(tf.rank(inputs), 1)

        # rebuild if k changed (e.g., dynamic shape)
        k_in = inputs.shape[-1]
        if (self._k is None) or (k_in is not None and int(k_in) != int(self._k)):
            self.build(inputs.shape)

        x = tf.cast(inputs, tf.float32)

        # parity = (x @ G) mod 2  -> shape [..., crc_len]
        parity = tf.linalg.matmul(x, self._g_mat_crc)

        # mod-2 using integer ops
        parity = tf.cast(parity, tf.int64)
        parity = int_mod_2(parity)
        parity = tf.cast(parity, self.dtype)

        return tf.concat([tf.cast(inputs, self.dtype), parity], axis=-1)


class CRCDecoder(Layer):
    """
    CRCDecoder(crc_encoder, dtype=tf.float32)

    Verifies CRC and removes parity bits.

    Returns
    -------
    x_info : tf.Tensor [..., k]
    crc_valid : tf.Tensor [..., 1] (bool)
    """

    def __init__(self, crc_encoder: CRCEncoder, dtype: tf.DType = tf.float32, **kwargs):
        if not isinstance(crc_encoder, CRCEncoder):
            raise TypeError("crc_encoder must be an instance of CRCEncoder.")
        self._encoder = crc_encoder

        # default dtype: match encoder if not given
        if dtype is None:
            dtype = crc_encoder.dtype

        super().__init__(dtype=dtype, **kwargs)

    @property
    def crc_degree(self) -> str:
        return self._encoder.crc_degree

    @property
    def encoder(self) -> CRCEncoder:
        return self._encoder

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        tf.debugging.assert_greater(tf.rank(inputs), 1)

        crc_len = self._encoder.crc_length
        tf.debugging.assert_greater_equal(tf.shape(inputs)[-1], crc_len)

        # Split info + received parity
        x_info = inputs[..., :-crc_len]
        parity_rx = inputs[..., -crc_len:]

        # Compute parity from info bits only
        x_crc = self._encoder(x_info)
        parity_hat = x_crc[..., -crc_len:]

        # Compare (bitwise): valid iff all parity bits match
        # Use int_mod_2 on (parity_hat + parity_rx) for XOR in {0,1}
        diff = tf.cast(parity_hat, tf.int64) + tf.cast(parity_rx, tf.int64)
        diff = int_mod_2(diff)  # XOR for {0,1}
        crc_valid = tf.reduce_all(tf.equal(diff, 0), axis=-1, keepdims=True)

        return tf.cast(x_info, self.dtype), crc_valid