# Dynamic SC-Flip Decoding for Polar Codes

TensorFlow-based implementation of **Successive Cancellation (SC)** and **Dynamic SC-Flip (DSC-Flip)** decoding for polar codes with CRC-aided stopping.

This repository provides:

* Polar code construction (5G / GA / RM variants)
* CRC encoder/decoder layers
* SC and DSC-Flip decoders
* AWGN simulation framework
* Plotting and visualization utilities

---

## Features

* ✅ SC and CRC-aided SC-Flip decoding
* ✅ Dynamic SC-Flip reliability metric (log-domain formulation)
* ✅ Multi-trial flip search
* ✅ 5G NR reliability sequence support
* ✅ Gaussian Approximation (GA) construction
* ✅ RM and hybrid constructions
* ✅ End-to-end FER simulation on AWGN

---

## Repository Structure

```
.
├── DSCFlip.py          # SC and DSC-Flip decoders
├── crc.py              # CRC encoder/decoder layers
├── constructions.py    # GA, RM, and hybrid constructions
├── utils.py            # Polar transform, ranking, plotting utilities
├── sim.py              # Simulation entry point
├── codes/
│   └── polar_5G.csv    # 5G reliability ranking table
├── results/            # Saved FER curves
├── figs/               # Generated plots
```

---

## Implemented Components

### 1. Polar Code Construction

Implemented in `constructions.py` 

* Gaussian Approximation (GA)
* RM-based design
* RM-Polar hybrid rule
* 5G NR ranking support

---

### 2. CRC Modules

Implemented in `crc.py` 

Generator-matrix-based CRC encoder and decoder.

Supported polynomials:

* CRC24A / CRC24B / CRC24C
* CRC16
* CRC11
* CRC6
* CRC8
* CRC8DVBS2

---

### 3. Decoders

Implemented in `DSCFlip.py` 

* `PolarSCDecoder`
* `PolarDSCFlipDecoder`

Features:

* Recursive SC decoding
* Optional fast-SC pruning
* Flip-mask injection
* Log-domain DSCF metric
* Multi-trial search
* CRC-aided early stopping

---

### 4. Simulation Framework

Implemented in `sim.py` 

* AWGN channel
* BPSK modulation
* FER evaluation
* Configurable SNR points
* Automatic results saving

Example configuration:

```python
cfg = SimConfig(
    n_log2=8,
    K_msg=128,
    use_crc=True,
    crc_degree="CRC16",
    decoder="SCF",
    omega=1,
    num_trials=10,
    alpha=0.3367,
    construction="5G",
    snr_db=(-1, 0, 1),
)
```

---

## Installation

```bash
pip install numpy tensorflow matplotlib pandas scipy
```

---

## Running a Simulation

```bash
python sim.py
```

FER results are saved in:

```
results/
```

Plots can be generated using utilities in `utils.py` .

---

## DSC-Flip Metric

The reliability metric is implemented in log-domain for numerical stability:

$$m(i) = e^{-\alpha |L_j|} \cdot \prod_{j \le i, j \in \mathcal{I}} \frac{1}{1 + e^{-\alpha |L_j|}}$$

where:

* ($L_i$) are SC leaf LLRs
* ($\alpha$) controls smoothness
* candidates are sorted and tested sequentially

---

## Dependencies

* Python 3.9+
* TensorFlow 2.x
* NumPy
* SciPy
* Matplotlib

---

## Notes

* Code assumes BPSK over AWGN.
* All-zero codeword simulation (standard symmetry assumption).
* Internal decoder computations use `tf.float32` for stability.
* Designed for research and reproducibility.

---
