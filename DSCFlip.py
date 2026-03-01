import numpy as np
import numbers
import tensorflow as tf 






class PolarSCDecoder(tf.keras.layers.Layer):
    """
    Successive Cancellation (SC) decoder for polar codes.

    Inputs
    ------
    inputs : tf.Tensor, shape [..., n]
        Channel LLRs or logits (you negate them below to convert logits->LLR).
    flip_bits : Optional[tf.Tensor], shape [..., n] or [batch, n]
        Binary mask indicating which bit decisions to flip (SC-Flip support).
        If provided, is only applied at leaf decisions.
    oracle_dec : bool
        If True, forces lower-branch VN update to use zeros (oracle-like behavior).

    Returns
    -------
    u_hat_n : tf.Tensor, shape [batch, n]
        Estimated u (including frozen positions).
    llr_out : tf.Tensor, shape [batch, n]
        Leaf LLRs after traversal (clipped).
    """

    def __init__(
        self,
        frozen_pos,
        n,
        output_dtype=tf.float32,
        use_fast_sc=False,
        use_flip_sc=False,
        llr_max=80.0,
        **kwargs,
    ):
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError("output_dtype must be one of {tf.float16, tf.float32, tf.float64}.")
        if output_dtype is not tf.float32:
            tf.print("Note: decoder uses tf.float32 for internal calculations.")

        super().__init__(dtype=output_dtype, **kwargs)
        self._output_dtype = output_dtype

        # ---- validate n
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        # ---- validate frozen_pos
        frozen_pos = np.asarray(frozen_pos)
        if frozen_pos.dtype.kind not in ("i", "u"):
            raise TypeError("frozen_pos must contain integers.")
        if frozen_pos.ndim != 1:
            raise ValueError("frozen_pos must be a 1-D array of positions.")
        if len(frozen_pos) > n:
            raise ValueError("len(frozen_pos) cannot exceed n.")
        if np.any(frozen_pos < 0) or np.any(frozen_pos >= n):
            raise ValueError("frozen_pos entries must be in [0, n-1].")

        frozen_pos = np.unique(frozen_pos)  # ensure no duplicates

        # ---- store code params
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            raise RuntimeError("Internal error: invalid info_pos generated.")

        self._llr_max = float(llr_max)

        # frozen indicator vector: 1 for frozen positions, 0 otherwise
        self._frozen_ind = np.zeros(self._n, dtype=np.int8)
        self._frozen_ind[self._frozen_pos] = 1

        self._use_fast_sc = bool(use_fast_sc)
        self._use_flip_sc = bool(use_flip_sc)

    # --------------------
    # Public properties
    # --------------------
    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def frozen_pos(self):
        return self._frozen_pos

    @property
    def info_pos(self):
        return self._info_pos

    @property
    def llr_max(self):
        return self._llr_max

    @property
    def output_dtype(self):
        return self._output_dtype

    # --------------------
    # Internal ops
    # --------------------
    def _clip_llr(self, x):
        return tf.clip_by_value(x, -self._llr_max, self._llr_max)

    def _cn_op_tf(self, x, y):
        """Check-node update (boxplus) for LLRs."""
        x_in = self._clip_llr(x)
        y_in = self._clip_llr(y)

        # stable form:
        # log(1+exp(x+y)) - log(exp(x)+exp(y))
        llr_out = tf.math.log1p(tf.math.exp(x_in + y_in))
        llr_out -= tf.math.log(tf.math.exp(x_in) + tf.math.exp(y_in))
        return llr_out

    def _vn_op_tf(self, x, y, u_hat):
        """Variable-node update for LLRs."""
        return tf.multiply((1.0 - 2.0 * u_hat), x) + y

    def _hard_decision(self, llr):
        """Hard decision: 0 if llr>=0 else 1, with tie-breaking to 1 (matches your code)."""
        u = 0.5 * (1.0 - tf.sign(llr))
        u = tf.where(tf.equal(u, 0.5), tf.ones_like(u), u)
        return u

    def _polar_decode_sc_tf(self, llr_ch, frozen_ind, flip_bits=None, oracle_dec=False):
        """
        Recursive SC decoding.
        frozen_ind: numpy array of {0,1} with length n (python-side).
        flip_bits: tf.Tensor with shape [batch, n] or None.
        """
        n = int(len(frozen_ind))

        # Fast SC prune: all frozen
        if n > 1 and self._use_fast_sc and int(np.sum(frozen_ind)) == n:
            u_hat = tf.zeros_like(llr_ch)
            return u_hat, llr_ch, u_hat

        if n > 1:
            half = n // 2

            llr1 = llr_ch[..., :half]
            llr2 = llr_ch[..., half:]
            frozen1 = frozen_ind[:half]
            frozen2 = frozen_ind[half:]

            flip1 = None
            flip2 = None
            if self._use_flip_sc and flip_bits is not None:
                flip1 = flip_bits[..., :half]
                flip2 = flip_bits[..., half:]

            # upper
            llr_up_in = self._cn_op_tf(llr1, llr2)
            u1, llr_up_out, u1_up = self._polar_decode_sc_tf(llr_up_in, frozen1, flip1, oracle_dec)

            # lower (VN uses either oracle zeros or previous decisions)
            u_for_vn = tf.zeros_like(u1_up) if oracle_dec else u1_up
            llr_lo_in = self._vn_op_tf(llr1, llr2, u_for_vn)
            u2, llr_lo_out, u2_up = self._polar_decode_sc_tf(llr_lo_in, frozen2, flip2, oracle_dec)

            # combine decisions
            u_hat = tf.concat([u1, u2], axis=-1)
            llr_out = tf.concat([llr_up_out, llr_lo_out], axis=-1)

            # re-encode partial sums upward:
            # u1_up <- u1_up xor u2_up
            u1_up_i = tf.cast(u1_up, tf.int8)
            u2_up_i = tf.cast(u2_up, tf.int8)
            u1_up_xor = tf.bitwise.bitwise_xor(u1_up_i, u2_up_i)
            u1_up = tf.cast(u1_up_xor, tf.float32)

            u_up = tf.concat([u1_up, u2_up], axis=-1)
            return u_hat, llr_out, u_up

        # ---- leaf
        # frozen_ind is length-1 numpy array; check its only value
        is_frozen = int(frozen_ind[0]) == 1
        llr_leaf = llr_ch  # shape [batch, 1]

        if is_frozen:
            u = tf.zeros_like(llr_leaf)
            return u, llr_leaf, u

        # info bit
        u = self._hard_decision(llr_leaf)

        if self._use_flip_sc and flip_bits is not None:
            # flip_bits should be shape [batch, 1] here
            flip_bits = tf.cast(flip_bits, u.dtype)
            u = tf.where(tf.equal(flip_bits, 1.0), 1.0 - u, u)
            llr_leaf = tf.where(tf.equal(flip_bits, 1.0), -llr_leaf, llr_leaf)

        return u, llr_leaf, u

    # --------------------
    # Keras methods
    # --------------------
    def build(self, input_shape):
        if input_shape[-1] != self._n:
            raise ValueError(f"Invalid input shape: last dim must be n={self._n}.")
        if len(input_shape) < 2:
            raise ValueError("Inputs must have at least 2 dimensions.")
        super().build(input_shape)

    def call(self, inputs, flip_bits=None, oracle_dec=False):
        tf.debugging.assert_type(inputs, self.dtype, "Invalid input dtype.")

        x = tf.cast(inputs, tf.float32)
        tf.debugging.assert_equal(tf.shape(x)[-1], self._n, "Last input dimension must be length n.")
        tf.debugging.assert_greater(tf.rank(x), 1)

        # Flatten batch dims to [batch, n]
        x = tf.reshape(x, [-1, self._n])

        # Convert logits -> LLR if your upstream provides logits
        llr_ch =  x

        fb = None
        if self._use_flip_sc and flip_bits is not None:
            fb = tf.cast(flip_bits, tf.float32)
            fb = tf.reshape(fb, [-1, self._n])
            tf.debugging.assert_equal(tf.shape(fb)[-1], self._n, "flip_bits last dim must be n.")
            # ensure binary-ish
            fb = tf.where(fb > 0.5, tf.ones_like(fb), tf.zeros_like(fb))

        u_hat_n, llr_out, _ = self._polar_decode_sc_tf(llr_ch, self._frozen_ind, fb, bool(oracle_dec))
        llr_out = self._clip_llr(llr_out)

        # return in internal flattened shape like your original (batch, n)
        return tf.cast(u_hat_n, self._output_dtype), tf.cast(llr_out, self._output_dtype)



class PolarDSCFlipDecoder(tf.keras.Model):
    """
    Polar Dynamic Successive Cancellation Flip (DSCF) Decoder.

    Parameters
    ----------
    frozen_pos : array-like of int
        Indices of frozen bit positions (0-based indexing).
        Length equals N - K.

    n : int
        Codeword length. Must be a power of two.

    omga : int
        Maximum flip order ω.
        ω = 1  → standard SC-Flip.
        ω > 1  → multi-bit Dynamic SC-Flip search.

    crc_decoder : callable, optional
        CRC checking function used as stopping criterion.
        Must return (decoded_bits, crc_pass_flag),
        where crc_pass_flag is a boolean tensor.

    alpha : float, default=0.3
        Scaling parameter used in the log-domain reliability metric.
        Controls smoothness of metric accumulation.

    num_trials : int, default=2
        Maximum number of flip attempts per codeword.

    output_dtype : tf.DType, default=tf.float32
        Output tensor data type.
        Internal computations use float32 for stability.

    use_fast_sc : bool, default=False
        Enables fast-SC pruning of frozen subtrees.

    use_flip_sc : bool, default=False
        Enables flip-mask support in the underlying SC decoder.

    Notes
    -----
    The decoder first performs standard SC decoding.
    If CRC fails, flip candidates are generated based on
    a reliability metric and tested sequentially.
    """

    def __init__(
        self,
        frozen_pos,
        n,
        omga,
        crc_decoder=None,
        alpha=0.3,
        num_trials=2,
        output_dtype=tf.float32,
        use_fast_sc=False,
        use_flip_sc=True,
        **kwargs,
    ):
        super().__init__(dtype=output_dtype, **kwargs)

        self.n = int(n)
        self.omga = int(omga)
        self.alpha = tf.constant(alpha, tf.float32)
        self.num_trials = int(num_trials)

        if crc_decoder is None:
            raise ValueError("crc_decoder must be provided.")
        self._crc_decoder = crc_decoder

        # decoder
        self.decoder = PolarSCDecoder(
            frozen_pos=frozen_pos,
            n=self.n,
            output_dtype=output_dtype,
            use_fast_sc=use_fast_sc,
            use_flip_sc=use_flip_sc,
        )

        # info positions (sorted)
        info_bits = np.setdiff1d(np.arange(self.n), np.asarray(frozen_pos, dtype=np.int32))
        self._info_bits = info_bits
        self._info_bits_tf = tf.constant(info_bits.astype(np.int32), dtype=tf.int32)

        self._output_dtype = output_dtype

    # ----------------------------
    # Fast metric: log-form (vectorized)
    # m(i) = |L_i| + (1/alpha) * sum_{j in info_bits, j <= i} log(1+exp(-alpha*|L_j|))
    #
    # If we order info bits increasingly, this becomes:
    #   v = |L| gathered at info bits         shape [B, K]
    #   s = log1p(exp(-alpha*v))              shape [B, K]
    #   prefix = cumsum(s)                    shape [B, K]
    #   metric = v + (1/alpha) * prefix       shape [B, K]
    #
    # This matches your "log" metric but WITHOUT per-i boolean masks.
    # ----------------------------
    @tf.function(reduce_retracing=True)
    def _init_metric(self, llr_leaf_full):
        llr_abs = tf.math.abs(llr_leaf_full)                          # [B, N]
        v = tf.gather(llr_abs, self._info_bits_tf, axis=1)            # [B, K]
        pe = tf.exp(-self.alpha * v)
        s = 1/(1+tf.exp(-self.alpha * v))                              # [B, K]
        prefix = tf.math.cumprod(s, axis=1)                                 # [B, K]
        metric = pe*prefix                                             # [B, K]
        return metric

    # ----------------------------
    # Build flip mask from indices (dense)
    # flip_idx: int32 tensor shape [B, m] (m flips per word)
    # returns flip_mask: float32 shape [B, N] with {0,1}
    # ----------------------------
    @tf.function(reduce_retracing=True)
    def _flip_mask_from_indices(self, flip_idx):
        flip_idx = tf.cast(flip_idx, tf.int32)                        # [B, m]
        onehot = tf.one_hot(flip_idx, depth=self.n, dtype=tf.float32) # [B, m, N]
        flip_mask = tf.reduce_sum(onehot, axis=1)                     # [B, N]
        # if duplicates happen, mod 2 is the correct behavior for flipping twice
        flip_mask = tf.math.floormod(flip_mask, 2.0)
        return flip_mask

    # ----------------------------
    # Main call
    # ----------------------------
    def call(self, x):
        # shape normalize: [B, N]
        llr_ch = tf.reshape(x, [-1, self.n])

        # --- initial SC
        u_hat_n, llr_leaf = self.decoder(llr_ch, flip_bits=None)
        u_hat = tf.gather(u_hat_n, self._info_bits_tf, axis=1)

        _, crc_ok = self._crc_decoder(u_hat)  # crc_ok: [B] boolean
        crc_ok = tf.cast(crc_ok, tf.bool)

        # If all pass, return
        if tf.reduce_all(crc_ok):
            llr_leaf = tf.cast(llr_leaf, self._output_dtype)
            u_out = tf.where(llr_leaf >= 0, tf.zeros_like(llr_leaf), tf.ones_like(llr_leaf))
            return tf.cast(u_out, self._output_dtype), llr_leaf

        # Work only on failed words
        fail_idx = tf.where(~crc_ok)[:, 0]  # [Bf]
        suc_idx  = tf.where(crc_ok)[:, 0]   # [Bs]

        llr_fail_in = tf.gather(llr_ch, fail_idx, axis=0)     # [Bf, N]
        llr_fail_leaf = tf.gather(llr_leaf, fail_idx, axis=0) # [Bf, N]

        # Store successful outputs from first pass (optional)
        llr_suc_list = []
        if tf.size(suc_idx) > 0:
            llr_suc_list.append(tf.gather(llr_leaf, suc_idx, axis=0))

        # --- compute metric on failed words (FAST)
        metric = self._init_metric(llr_fail_leaf)    # [Bf, K]

        # sort candidates per word
        sort_idx = tf.argsort(metric, axis=1, direction="DESCENDING")  # [Bf, K]
        metric_sorted = tf.gather(metric, sort_idx, batch_dims=1)     # [Bf, K]
        flip_sorted = tf.gather(self._info_bits_tf, sort_idx)         # [Bf, K]

        # keep only top-T trials
        T = tf.minimum(self.num_trials, tf.shape(flip_sorted)[1])
        flip_sorted = flip_sorted[:, :T]            # [Bf, T]
        metric_sorted = metric_sorted[:, :T]        # [Bf, T]

        # Iterate trials t=0..T-1
        # We keep a dynamic set of remaining failed words.
        for t in range(self.num_trials):
            # if no failed words left, break
            if tf.shape(llr_fail_in)[0] == 0:
                break
            if t >= T:
                break

            # choose flip index for this trial: one flip per word (ω=1 baseline)
            flip_idx_t = tf.expand_dims(flip_sorted[:, t], axis=1)  # [Bf, 1]
            flip_mask_t = self._flip_mask_from_indices(flip_idx_t)  # [Bf, N]

            u_hat_n_t, llr_leaf_t = self.decoder(llr_fail_in, flip_bits=flip_mask_t)
            u_hat_t = tf.gather(u_hat_n_t, self._info_bits_tf, axis=1)

            _, crc_ok_t = self._crc_decoder(u_hat_t)
            crc_ok_t = tf.cast(crc_ok_t, tf.bool)

            # split
            suc_t = tf.where(crc_ok_t)[:, 0]
            fail_t = tf.where(~crc_ok_t)[:, 0]

            if tf.size(suc_t) > 0:
                llr_suc_list.append(tf.cast(tf.gather(llr_leaf_t, suc_t, axis=0), tf.float32))

            # keep only remaining fails for next trial
            llr_fail_in = tf.gather(llr_fail_in, fail_t, axis=0)
            llr_fail_leaf = tf.gather(llr_leaf_t, fail_t, axis=0)

            # also keep candidate tables aligned with remaining fails
            flip_sorted = tf.gather(flip_sorted, fail_t, axis=0)
            metric_sorted = tf.gather(metric_sorted, fail_t, axis=0)
            
            if self.omga>1: 
                pass
        # After trials: combine all outputs
        if len(llr_suc_list) > 0:
            llr_all = tf.concat(llr_suc_list + [tf.cast(llr_fail_leaf, tf.float32)], axis=0)
        else:
            llr_all = tf.cast(llr_fail_leaf, tf.float32)

        llr_all = tf.cast(llr_all, self._output_dtype)
        u_out = tf.where(llr_all >= 0, tf.zeros_like(llr_all), tf.ones_like(llr_all))
        return tf.cast(u_out, self._output_dtype), tf.cast(llr_all, self._output_dtype)      


