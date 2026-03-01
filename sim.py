
import os
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from DSCFlip import PolarDSCFlipDecoder, PolarSCDecoder
from utils import generate_5g_ranking

# Optional: if you rely on local modules outside PYTHONPATH, prefer packaging instead of sys.path hacks.

from constructions import GA
from crc import CRCEncoder, CRCDecoder
from utils import generate_polar_transform_mat


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class SimConfig:
    n_log2: int = 8                 # N = 2^n_log2
    K_msg: int = 128                # message bits (without CRC)
    use_crc: bool = True
    crc_degree: str = "CRC16"       # CRC polynomial name (your CRCEncoder handles this)
    crc_len: int = 16               # number of CRC bits

    decoder: str = "SCF"            # "SCF" or "SC"
    omega: int = 1                  # DSCF/SCF maximum flip order (omega)
    num_trials: int = 10            # max trials (T)
    alpha: float = 0.3367           # metric hyperparameter
    construction: str = "5G"        # "5G", "GA", "Tal", "ours"

    snr_db: tuple = (-1, 0, 1)      # Es/N0 in dB (or Eb/N0 depending on your LLR convention)
    batches: tuple = (10_000, 100_000, 1_000_000)  # per SNR point

    min_block_errors: int = 50
    max_blocks: int = int(1e9)


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs(*names: str) -> None:
    for d in names:
        os.makedirs(d, exist_ok=True)


def snr_db_to_sigma(snr_db: np.ndarray) -> np.ndarray:
    """
    For BPSK with unit-energy symbols and LLR = 2y/sigma^2:
      sigma^2 = 1/(2 * 10^(snr_db/10))  (common when snr_db = Es/N0 in dB and Es=1)
    """
    snr_lin = 10 ** (snr_db / 10.0)
    return np.sqrt(1.0 / (2.0 * snr_lin))


def design_frozen_bits(cfg: SimConfig, N: int, K_total: int) -> np.ndarray:
    """
    Returns frozen bit positions (0-based) as a 1-D numpy array of ints.
    K_total includes CRC bits if enabled.
    """
    cons = cfg.construction

    if cons == "5G":
        frozen_pos, _info_pos = generate_5g_ranking(K_total, N)
        return np.array(frozen_pos, dtype=np.int32)

    if cons == "GA":
        # GA design SNR example (keep your logic)
        design_SNRdb = 3.6
        design_SNR = 10 ** (design_SNRdb / 10.0) * (K_total / N)
        init_mean = np.array([4.0 * design_SNR] * N)
        Z = GA(design_SNR, int(np.log2(N)), K_total / N).gaussian_approx(init_mean)
        return np.argsort(Z).astype(np.int32)

    raise ValueError(f"Unknown construction: {cons}")


def build_encoder_matrices(n_log2: int):
    """Polar transform matrix G_N (binary) for encoding."""
    g_mat = generate_polar_transform_mat(n_log2)  # expects shape [N, N]
    return g_mat


def build_model(cfg: SimConfig, frozen_bits: np.ndarray, N: int, crc_decoder):
    if cfg.decoder.upper() == "SCF":
        return PolarDSCFlipDecoder(
            frozen_pos=frozen_bits,
            n=N,
            omga=cfg.omega,
            crc_decoder=crc_decoder,
            alpha=cfg.alpha,
            num_trials=cfg.num_trials,
            use_flip_sc=True,
        )

    if cfg.decoder.upper() == "SC":
        return PolarSCDecoder(frozen_pos=frozen_bits, n=N, use_flip_sc=False)

    raise ValueError(f"Unknown decoder type: {cfg.decoder}")


def encode_batch(
    b: np.ndarray,
    N: int,
    info_bits: np.ndarray,
    g_mat: np.ndarray,
    crc_encoder=None,
) -> np.ndarray:
    """
    b: [B, K_msg] float32 {0,1}
    returns enc_msg: [B, N] float32 {0,1}
    """
    B = b.shape[0]
    u = np.zeros((B, N), dtype=np.float32)

    if crc_encoder is not None:
        u[:, info_bits] = crc_encoder(b)  # [B, K_msg+crc]
    else:
        u[:, info_bits] = b

    # binary encoding: x = uG mod 2
    x = (u @ g_mat) % 2
    return x.astype(np.float32)


# -----------------------------
# Main simulation
# -----------------------------
def run(cfg: SimConfig):
    ensure_dirs("figs", "results", "models", "hyperparams")

    # Code parameters
    N = 2 ** cfg.n_log2
    crc_len = cfg.crc_len if cfg.use_crc else 0
    K_total = cfg.K_msg + crc_len

    g_mat = build_encoder_matrices(cfg.n_log2)

    # CRC
    if cfg.use_crc:
        crc_encoder = CRCEncoder(cfg.crc_degree)
        crc_decoder = CRCDecoder(crc_encoder)
    else:
        crc_encoder = None
        crc_decoder = None

    # Frozen set / info set
    frozen_bits = design_frozen_bits(cfg, N=N, K_total=K_total)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int32)

    # Ensure correct count: number frozen = N - K_total
    frozen_bits = frozen_bits[: (N - K_total)]
    frozen_bits = np.unique(frozen_bits).astype(np.int32)

    info_bits = np.setdiff1d(np.arange(N), frozen_bits).astype(np.int32)

    print(f"N={N}, K_msg={cfg.K_msg}, CRC={cfg.use_crc}, K_total={K_total}")
    print(f"Frozen bits: {len(frozen_bits)}, Info bits: {len(info_bits)}")

    # Model
    model = build_model(cfg, frozen_bits=frozen_bits, N=N, crc_decoder=crc_decoder)

    # SNRs
    snr_db = np.array(cfg.snr_db, dtype=np.float32)
    sigma = snr_db_to_sigma(snr_db)

    FER = []

    for idx, sig in enumerate(sigma):
        batch_size = int(cfg.batches[min(idx, len(cfg.batches) - 1)])
        snr_point = float(snr_db[idx])

        print("\n" + "*" * 30 + f" SNR={snr_point:.2f} dB, alpha={cfg.alpha:.4f} " + "*" * 30)

        # Fixed all-zero message (common for symmetric channels)
        b = np.zeros((batch_size, cfg.K_msg), dtype=np.float32)
        x_bits = encode_batch(b, N=N, info_bits=info_bits, g_mat=g_mat, crc_encoder=crc_encoder)

        fer = 0
        ber = 0
        itr = 0

        # Precompute for speed
        sig2 = float(sig ** 2)

        while True:
            itr += 1

            # BPSK: 0 -> +1, 1 -> -1
            x_bpsk = 1.0 - 2.0 * x_bits
            y = x_bpsk + np.random.randn(batch_size, N).astype(np.float32) * sig

            llr = (2.0 / sig2) * y  # LLR for BPSK in AWGN
            llr = tf.convert_to_tensor(llr, dtype=tf.float32)

            u_hat_n, _llr_out = model(llr)
            u_hat_info = tf.gather(u_hat_n, info_bits, axis=1)

            if cfg.use_crc:
                u_hat, crc_ok = crc_decoder(u_hat_info)
                crc_ok = tf.cast(crc_ok, tf.bool)
                fer_inc = int(tf.reduce_sum(tf.cast(~crc_ok, tf.int32)).numpy())
                fer += fer_inc
            else:
                u_hat = u_hat_info
                # block error if any bit differs
                fer += int(np.sum(np.any(u_hat.numpy() != b, axis=1)))

            ber += int(np.sum(u_hat.numpy() != b))

            blocks = batch_size * itr
            fer_rate = fer / blocks
            print(f"ESN0={snr_point:.2f} dB | FER={fer_rate:1.4e} | BLER_count={fer:1.4e} | blocks={blocks:1.5e}")

            if fer >= cfg.min_block_errors:
                break
            if blocks >= cfg.max_blocks:
                break

        FER.append(fer_rate)

        # Save after each SNR point (robust)
        if cfg.decoder.upper() == "SCF":
            out_path = f"results/FER_{N}_{cfg.K_msg}_T{cfg.num_trials}_SCF_alpha_{cfg.alpha}_{cfg.construction}_crc_{cfg.use_crc}_log.csv"
        else:
            out_path = f"results/FER_{N}_{cfg.K_msg}_SC_alpha_{cfg.alpha}_{cfg.construction}_crc_{cfg.use_crc}.csv"

        np.savetxt(out_path, np.array(FER, dtype=np.float64), delimiter=",")
        

    return np.array(FER)


if __name__ == "__main__":
    cfg = SimConfig(
        n_log2=8,
        K_msg=128,
        use_crc=True,
        crc_degree="CRC16",
        crc_len=16,
        decoder="SCF",
        omega=1,
        num_trials=10,
        alpha=0.3367,
        construction="5G",
        snr_db=(-1, 0, 1),
        batches=(10_000, 100_000, 1_000_000),
        min_block_errors=50,
        max_blocks=int(1e10),
    )
    run(cfg)