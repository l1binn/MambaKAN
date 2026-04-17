"""
Phase 1 — VAE Unsupervised Pre-training Demo.

The VAE is trained on per-window dFC vectors (without class labels) to learn a
compact, noise-robust latent representation.  The best checkpoint (lowest
validation reconstruction loss) is saved for Phase 2 initialisation.

Training details (Section 2.5 of the paper):
    Optimiser : Adam  (lr = 1e-3, beta1 = 0.9, beta2 = 0.999)
    Epochs    : 100
    Batch size: 32
    Loss      : L_VAE = MSE_recon + KL

Usage:
    python train_stage1.py

NOTE — Data placeholder:
    This demo uses randomly generated tensors.  Replace the two blocks
    marked  <<<  REPLACE WITH YOUR DATA LOADER  >>>  with your own
    torch.utils.data.Dataset / DataLoader that yields:
        x  : (batch, N_WINDOWS, INPUT_DIM)  float32 dFC feature vectors
    No labels are needed in Phase 1.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.vae import VAE, vae_loss

# ── Hyper-parameters ─────────────────────────────────────────────────────────
INPUT_DIM   = 6670          # N*(N-1)/2 for 116 AAL ROIs
LATENT_DIM  = 128
HIDDEN_DIMS = [2048, 1024, 512, 256]

N_EPOCHS    = 100
BATCH_SIZE  = 32
LR          = 1e-3
N_WINDOWS   = 54            # dFC windows per subject

CKPT_DIR  = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "vae_stage1_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data placeholder ─────────────────────────────────────────────────────────
# <<<  REPLACE WITH YOUR DATA LOADER (training set, no labels needed)  >>>
N_TRAIN = 120   # number of training subjects (dummy)
N_VAL   = 30    # number of validation subjects (dummy)

torch.manual_seed(0)
train_data = torch.randn(N_TRAIN, N_WINDOWS, INPUT_DIM)
val_data   = torch.randn(N_VAL,   N_WINDOWS, INPUT_DIM)
# <<<  END REPLACE  >>>

train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_data),   batch_size=BATCH_SIZE, shuffle=False)


# ── Model & optimiser ────────────────────────────────────────────────────────
os.makedirs(CKPT_DIR, exist_ok=True)

model = VAE(
    input_dim   = INPUT_DIM,
    hidden_dims = HIDDEN_DIMS,
    latent_dim  = LATENT_DIM,
).to(DEVICE)

optimiser = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

total_params = sum(p.numel() for p in model.parameters())
print(f"VAE parameters: {total_params:,}")
print(f"Training on   : {DEVICE}")
print(f"Train subjects: {N_TRAIN}  |  Val subjects: {N_VAL}")
print("-" * 55)


# ── Training loop ────────────────────────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(1, N_EPOCHS + 1):
    # ── train ──────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(DEVICE)
        b, w, f = x_batch.shape
        x_flat = x_batch.reshape(b * w, f)     # (b*W, input_dim)

        recon, mu, logvar, _, _ = model(x_flat)
        loss = vae_loss(recon, x_flat, mu, logvar)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    # ── validate ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(DEVICE)
            b, w, f = x_batch.shape
            x_flat = x_batch.reshape(b * w, f)

            recon, mu, logvar, _, _ = model(x_flat)
            loss = vae_loss(recon, x_flat, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)

    # ── checkpoint ─────────────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        flag = "  <-- saved"
    else:
        flag = ""

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}{flag}")

print("-" * 55)
print(f"Phase 1 complete.  Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {CKPT_PATH}")
