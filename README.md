# MambaKAN: An Interpretable Framework for Alzheimer’s Disease Diagnosis via Selective State Space Modeling of Dynamic Functional Connectivity

Official code release for the paper:

> **MambaKAN: An Interpretable Framework for Alzheimer’s Disease Diagnosis via Selective State Space Modeling of Dynamic Functional Connectivity**  
> *Brain Sciences*, 2026.

---

> **Note on this release:**  
> This repository contains the **core model architecture and training logic** of MambaKAN.  
> Some implementation details (e.g. data preprocessing, experiment-specific configurations) may differ slightly from the exact version used in the paper experiments, as the public code has been cleaned and generalised for reuse.  
> If you spot any inconsistency or have questions, please open a [GitHub Issue](../../issues) — feedback is welcome.

---

## Overview

MambaKAN is a three-stage framework for classifying Alzheimer's disease (AD) and its prodromal stages from resting-state fMRI dynamic functional connectivity (dFC):

1. **VAE** — unsupervised pre-training to compress high-dimensional dFC windows into compact latent representations
2. **Mamba (S6)** — selective state-space temporal encoder that models the sequence of dFC windows; pure PyTorch, no CUDA extension required
3. **KAN (B-spline)** — interpretable classifier with learnable spline activations per connection

```
Input (N, W, F)
    │
    ▼
VAE encoder ──► latent z (N·W, D)  reshape → (N, W, D)
    │
    ▼
Mamba encoder ──► temporal context (N, D)   ← mean-pool over W outputs
    │
    ▼
KAN classifier ──► logits (N, C)            ← one prediction per subject
```

Where W = number of dFC sliding windows, F = upper-triangular dFC features, D = latent dim, C = number of classes.

> **Clarification on inference:** The model produces one subject-level prediction per forward pass (Mamba mean-pools its W output vectors into a single context vector before KAN, as in Eq. 10 of the paper).  A sentence in the paper's dataset-partitioning section that refers to "54 window-level predictions aggregated via probability averaging" is a textual inaccuracy in the published version; the actual implementation is as shown above.

---

## Repository Structure

```
├── models/
│   ├── vae.py              # Variational Autoencoder
│   ├── mamba_encoder.py    # Mamba (S6) encoder — CUDA-friendly parallel scan
│   ├── kan_classifier.py   # B-spline KAN classifier
│   └── proposed.py         # Combined ProposedModel (VAE + Mamba + KAN)
├── train_stage1.py         # Phase 1: VAE unsupervised pre-training
├── train_stage2.py         # Phase 2: end-to-end joint fine-tuning
├── demo.py                 # Quick sanity check (random data, no real fMRI needed)
├── analysis.py             # Interpretability visualizations (6 modules)
└── README.md
```

---

## Two-Phase Training

> **Note:** `train_stage1.py` and `train_stage2.py` use randomly generated data as placeholders.  
> Replace the `<<<  REPLACE WITH YOUR DATA LOADER  >>>` blocks with your own `DataLoader`.

### Phase 1 — VAE Pre-training

```bash
python train_stage1.py
# Saves: checkpoints/vae_stage1_best.pth  (best validation reconstruction loss)
```

### Phase 2 — End-to-End Joint Fine-tuning

```bash
python train_stage2.py --vae_ckpt checkpoints/vae_stage1_best.pth
# Saves: checkpoints/mambaKAN_stage2_best.pth  (best validation accuracy)
```

Training hyperparameters (from paper Section 2.5):

| Hyperparameter | Value |
|---|---|
| Phase 1 optimizer | Adam, lr = 1e-3 |
| Phase 2 VAE lr | 1e-5 |
| Phase 2 Mamba + KAN lr | 1e-3 |
| Warmup epochs | 15 (VAE frozen) |
| Total epochs per phase | 100 |
| Batch size | 32 |
| Joint loss weights | α = 0.1 (VAE), β = 1.0 (cls) |
| Dropout | 0.15 (within Mamba block) |

---

## Installation

```bash
pip install torch torchvision numpy scikit-learn scipy matplotlib
# Optional, for chord diagram visualization:
pip install pyecharts
```

Tested with Python 3.10, PyTorch 2.1.

---

## Model Architecture

### VAE (`models/vae.py`)

Standard VAE with configurable encoder/decoder depths. Encodes each dFC window independently.

```python
from models.vae import VAE, vae_loss

vae = VAE(input_dim=6670, hidden_dims=[2048, 1024, 512, 256], latent_dim=128)
mu, logvar, h = vae.encode(x)   # x: (batch, 6670)
z = vae.reparameterize(mu, logvar)
recon = vae.decode(z)
loss = vae_loss(recon, x, mu, logvar)
```

### Mamba Encoder (`models/mamba_encoder.py`)

Stacked Mamba blocks with selective SSM (S6). No CUDA extension — runs on any hardware.

```python
from models.mamba_encoder import MambaEncoder

encoder = MambaEncoder(d_model=128, n_layers=2, d_state=16, d_conv=4, expand=2)
ctx = encoder(z)   # z: (batch, W, 128) → ctx: (batch, 128)

# For interpretability: per-timestep selectivity scores
importance = encoder.get_temporal_importance(z)   # (batch, W)
```

### KAN Classifier (`models/kan_classifier.py`)

Two-layer KAN with B-spline activations (Cox–de Boor recursion). Each connection has its own learnable spline φ(x).

> **Parameter count note:** This implementation includes per-connection learnable scaling factors (`scale_base`, `scale_spline`) for training stability, giving the KAN head ~93 K parameters.  The paper reports ~43 K, which counted only `spline_weight` + `base_weight`.  The difference does not affect the model's behaviour or the paper's experimental results.

> **Grid update:** `KANLinear.update_grid(x)` / `KANClassifier.update_grid(x)` adapts the B-spline grid to span the current activation range (as described in the paper).  `train_stage2.py` calls this every `GRID_UPDATE_FREQ = 10` epochs.

```python
from models.kan_classifier import KANClassifier

kan = KANClassifier(in_features=128, hidden_dim=64, num_classes=4,
                    grid_size=5, spline_order=3)
logits = kan(ctx)   # ctx: (batch, 128) → logits: (batch, 4)

# Interpretability
importance = kan.get_input_importance()   # (128,)  — per-dim spline weight L1
x_vals, class_curves = kan.get_class_curves(dim=0, x_mean=ctx.mean(0))
# x_vals: (200,), class_curves: (4, 200)
```

### Full Model (`models/proposed.py`)

```python
from models.proposed import ProposedModel

model = ProposedModel(
    input_dim=6670,
    vae_hidden_dims=[2048, 1024, 512, 256],
    latent_dim=128,
    mamba_n_layers=2,
    mamba_d_state=16,
    mamba_d_conv=4,
    mamba_expand=2,
    kan_hidden_dim=64,
    kan_grid_size=5,
    kan_spline_order=3,
    num_classes=4,
)

# Forward pass
x = torch.randn(8, 54, 6670)   # (batch, windows, features)
logits, vae_loss = model(x, compute_vae_loss=True)

# Gradient attribution for brain region mapping
attr = model.get_gradient_attribution(x[:1], class_idx=0)   # (W, 6670)
```

---

## Interpretability Analysis (`analysis.py`)

Six analysis modules that produce publication-ready figures:

| # | Module | Output |
|---|--------|--------|
| 1 | Mamba temporal importance | Per-window Δ selectivity curves + heatmap |
| 2 | KAN activation curves | Ceteris-paribus class logit curves per latent dim |
| 3 | Brain region attribution | 116×116 gradient heatmap, top-ROI bar chart, chord diagram |
| 4 | t-SNE latent space | 2D scatter of Mamba context vectors |
| 5 | ROC curves | One-vs-rest AUC per class |
| 6 | Statistical significance | Pairwise t-test -log10(p) heatmap |

### Data Format

Prepare your data as a `.npz` file:

```python
import numpy as np

np.savez("test_data.npz",
    test_data=test_data,     # float32 (N, W, F): N subjects, W windows, F features
    test_labels=test_labels, # int64   (N,):      0-indexed class labels
    aal_labels=aal_labels,   # str     (116,):    AAL region names (optional)
)
```

### Usage

```bash
python analysis.py \
    --data_path test_data.npz \
    --ckpt_path best_model.pth \
    --num_classes 4 \
    --class_names CN EMCI LMCI AD \
    --top_k 10 \
    --out_dir output/analysis
```

Run a subset of analyses:

```bash
# Only Mamba temporal importance (1) and ROC curves (5)
python analysis.py --data_path test_data.npz --ckpt_path best_model.pth \
                   --analyses 1 5 --num_classes 4
```

---

## Citation

If you use this code, please cite our paper:

Gao, L.; Hu, Z. MambaKAN: An Interpretable Framework for Alzheimer's Disease Diagnosis via Selective State Space Modeling of Dynamic Functional Connectivity. *Brain Sci.* **2026**, *16*, 421. https://doi.org/10.3390/brainsci16040421

```bibtex
@article{gao2026mambaKAN,
  author  = {Gao, Libin and Hu, Zhongyi},
  title   = {MambaKAN: An Interpretable Framework for Alzheimer's Disease Diagnosis
             via Selective State Space Modeling of Dynamic Functional Connectivity},
  journal = {Brain Sciences},
  year    = {2026},
  volume  = {16},
  number  = {4},
  pages   = {421},
  doi     = {10.3390/brainsci16040421},
  url     = {https://doi.org/10.3390/brainsci16040421},
}
```

---

## License

This project is released under the MIT License.
