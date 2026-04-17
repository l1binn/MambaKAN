"""
MambaKAN Demo — runs the full pipeline on random data (no real fMRI data needed).

Usage:
    python demo.py

Expected output:
    Logits shape:        torch.Size([4, 4])
    Predicted classes:   [0, 2, 1, 3]  (random, will vary)
    VAE loss:            ...
    Temporal importance: torch.Size([4, 54])
    Gradient attribution:torch.Size([54, 6670])
    KAN input importance:torch.Size([128])
    Top-5 important latent dims: [...]
"""

import torch
from models.proposed import ProposedModel

# ── Dimensions (must match a trained checkpoint if loading one) ──────────────
BATCH      = 4       # number of subjects
N_WINDOWS  = 54      # dFC sliding windows per subject
INPUT_DIM  = 6670    # N(N-1)/2 for 116 AAL ROIs
LATENT_DIM = 128
NUM_CLASSES = 4      # CN / EMCI / LMCI / AD

# ── Build model ───────────────────────────────────────────────────────────────
model = ProposedModel(
    input_dim      = INPUT_DIM,
    vae_hidden_dims= [2048, 1024, 512, 256],
    latent_dim     = LATENT_DIM,
    mamba_n_layers = 2,
    mamba_d_state  = 16,
    mamba_d_conv   = 4,
    mamba_expand   = 2,
    kan_hidden_dim = 64,
    kan_grid_size  = 5,
    kan_spline_order = 3,
    num_classes    = NUM_CLASSES,
    dropout        = 0.0,   # disable dropout for deterministic demo output
)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print()

# ── Random input: (batch, windows, features) ─────────────────────────────────
torch.manual_seed(42)
x = torch.randn(BATCH, N_WINDOWS, INPUT_DIM)

# ── 1. Forward pass (classification) ─────────────────────────────────────────
with torch.no_grad():
    logits, _ = model(x, compute_vae_loss=False)

probs      = torch.softmax(logits, dim=-1)
predicted  = logits.argmax(dim=-1)
label_map  = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "AD"}

print("-- Classification --------------------------------------------------")
print(f"  Logits shape:      {logits.shape}")
print(f"  Predicted classes: {[label_map[c.item()] for c in predicted]}")
print(f"  Class probabilities (sample 0): "
      f"{dict(zip(label_map.values(), probs[0].tolist()))}")
print()

# ── 2. VAE loss (as in Phase-2 joint training) ────────────────────────────────
_, vae_loss = model(x, compute_vae_loss=True)
print("-- VAE Loss --------------------------------------------------------")
print(f"  VAE loss (MSE + KL): {vae_loss.item():.2f}")
print()

# ── 3. Mamba temporal importance ──────────────────────────────────────────────
with torch.no_grad():
    importance = model.get_temporal_importance(x)   # (batch, W)

top3_windows = importance[0].topk(3).indices.tolist()
print("-- Mamba Temporal Importance ---------------------------------------")
print(f"  Shape: {importance.shape}")
print(f"  Top-3 windows for sample 0: {top3_windows}")
print()

# ── 4. Gradient attribution ───────────────────────────────────────────────────
attr = model.get_gradient_attribution(x[:1], class_idx=0)   # (W, F)
top3_features = attr.abs().mean(dim=0).topk(3).indices.tolist()
print("-- Gradient Attribution (class 0 = CN) ----------------------------")
print(f"  Shape: {attr.shape}  (windows x ROI-pair features)")
print(f"  Top-3 most attributed features (sample 0): {top3_features}")
print()

# ── 5. KAN interpretability ───────────────────────────────────────────────────
with torch.no_grad():
    kan_importance = model.kan.get_input_importance()   # (latent_dim,)
    top5_dims = kan_importance.topk(5).indices.tolist()

    # Ceteris-paribus curve for the most important latent dimension
    ctx_mean = torch.zeros(LATENT_DIM)   # use zero as anchor (demo only)
    x_vals, curves = model.kan.get_class_curves(top5_dims[0], ctx_mean)

print("-- KAN Classifier Interpretability --------------------------------")
print(f"  KAN importance shape: {kan_importance.shape}")
print(f"  Top-5 most influential latent dims: {top5_dims}")
print(f"  Class activation curve shape for dim {top5_dims[0]}: {curves.shape}")
print(f"    (num_classes={curves.shape[0]}, n_points={curves.shape[1]})")
print()

print("Demo complete -- all modules ran successfully.")
