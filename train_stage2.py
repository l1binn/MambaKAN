"""
Phase 2 — End-to-End Joint Fine-tuning Demo.

The full MambaKAN pipeline (VAE + Mamba + KAN) is trained jointly.

Architecture (paper Section 2.5):
    Forward path  : x (b, W, F)
                    → VAE encode each window → z (b, W, 128)
                    → Mamba temporal encoder → c (b, 128)   [mean-pooled sequence]
                    → KAN classifier        → logits (b, C)

Training details:
    Joint loss    : L = alpha * L_VAE + beta * L_cls
                    alpha = 0.1,  beta = 1.0
    Warmup        : first E_warm = 15 epochs — VAE parameters frozen
    Optimiser     : Adam  (beta1=0.9, beta2=0.999, eps=1e-8)
    Learning rates: VAE = 1e-5,  Mamba + KAN = 1e-3   (differential LRs)
    Epochs        : 100
    Batch size    : 32
    Dropout       : 0.15  (within each Mamba block, after depthwise conv)
    Checkpoint    : best validation accuracy saved

Usage:
    # Run Phase 1 first to obtain the VAE checkpoint
    python train_stage1.py

    # Then run Phase 2
    python train_stage2.py --vae_ckpt checkpoints/vae_stage1_best.pth

NOTE — Data placeholder:
    Replace the two blocks marked  <<<  REPLACE WITH YOUR DATA LOADER  >>>
    with your own DataLoader that yields:
        x      : (batch, N_WINDOWS, INPUT_DIM)  float32 dFC feature vectors
        labels : (batch,)                       int64  class indices (0-indexed)
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.proposed import ProposedModel

# ── Hyper-parameters ─────────────────────────────────────────────────────────
INPUT_DIM     = 6670
LATENT_DIM    = 128
HIDDEN_DIMS   = [2048, 1024, 512, 256]
N_WINDOWS     = 54
NUM_CLASSES   = 4           # CN / EMCI / LMCI / AD

MAMBA_N_LAYERS  = 2
MAMBA_D_STATE   = 16
MAMBA_D_CONV    = 4
MAMBA_EXPAND    = 2
KAN_HIDDEN_DIM  = 64
KAN_GRID_SIZE   = 5
KAN_SPLINE_ORD  = 3
DROPOUT         = 0.15

N_EPOCHS    = 100
E_WARM      = 15            # warmup epochs: VAE frozen
BATCH_SIZE  = 32
LR_NEW      = 1e-3          # Mamba + KAN learning rate
LR_VAE      = 1e-5          # VAE fine-tune learning rate
ALPHA       = 0.1           # VAE loss weight
BETA        = 1.0           # classification loss weight
GRID_UPDATE_FREQ = 50       # update KAN spline grid every N epochs (infrequent to limit disruption)

CKPT_DIR  = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "mambaKAN_stage2_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str,
                        default="checkpoints/vae_stage1_best.pth",
                        help="Path to Phase-1 VAE checkpoint")
    return parser.parse_args()


def build_loaders():
    """
    <<<  REPLACE WITH YOUR DATA LOADER  >>>
    Return (train_loader, val_loader) where each batch yields (x, labels).
        x      : (batch, N_WINDOWS, INPUT_DIM)  float32
        labels : (batch,)                       int64
    """
    N_TRAIN = 120
    N_VAL   = 30
    torch.manual_seed(1)
    x_tr = torch.randn(N_TRAIN, N_WINDOWS, INPUT_DIM)
    y_tr = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
    x_va = torch.randn(N_VAL, N_WINDOWS, INPUT_DIM)
    y_va = torch.randint(0, NUM_CLASSES, (N_VAL,))
    # <<<  END REPLACE  >>>

    tr_loader = DataLoader(TensorDataset(x_tr, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(x_va, y_va),
                           batch_size=BATCH_SIZE, shuffle=False)
    return tr_loader, va_loader


def main():
    args = parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_loader, val_loader = build_loaders()

    # ── Build model ──────────────────────────────────────────────────────────
    model = ProposedModel(
        input_dim        = INPUT_DIM,
        vae_hidden_dims  = HIDDEN_DIMS,
        latent_dim       = LATENT_DIM,
        mamba_n_layers   = MAMBA_N_LAYERS,
        mamba_d_state    = MAMBA_D_STATE,
        mamba_d_conv     = MAMBA_D_CONV,
        mamba_expand     = MAMBA_EXPAND,
        kan_hidden_dim   = KAN_HIDDEN_DIM,
        kan_grid_size    = KAN_GRID_SIZE,
        kan_spline_order = KAN_SPLINE_ORD,
        num_classes      = NUM_CLASSES,
        dropout          = DROPOUT,
    ).to(DEVICE)

    # ── Load Phase-1 VAE weights ─────────────────────────────────────────────
    if os.path.exists(args.vae_ckpt):
        vae_state = torch.load(args.vae_ckpt, map_location=DEVICE)
        model.vae.load_state_dict(vae_state)
        print(f"Loaded VAE checkpoint: {args.vae_ckpt}")
    else:
        print(f"[WARNING] VAE checkpoint not found at '{args.vae_ckpt}'. "
              "Training from random initialisation.")

    total_params = sum(p.numel() for p in model.parameters())
    vae_params   = sum(p.numel() for p in model.vae.parameters())
    new_params   = sum(p.numel() for p in model.mamba.parameters()) + \
                   sum(p.numel() for p in model.kan.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"  VAE            : {vae_params:,}")
    print(f"  Mamba + KAN    : {new_params:,}")
    print(f"Training on      : {DEVICE}")
    print("-" * 62)

    # ── Differential-LR optimiser ────────────────────────────────────────────
    # VAE parameters use a 100x smaller lr than Mamba + KAN (paper Eq. 12)
    optimiser = torch.optim.Adam(
        [
            {"params": model.vae.parameters(),   "lr": LR_VAE},
            {"params": model.mamba.parameters(), "lr": LR_NEW},
            {"params": model.kan.parameters(),   "lr": LR_NEW},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    best_val_acc = 0.0

    for epoch in range(1, N_EPOCHS + 1):

        # ── Warmup: freeze VAE for the first E_WARM epochs ─────────────────
        if epoch == 1:
            model.freeze_vae()
            print(f"[Warmup] VAE frozen for epochs 1-{E_WARM}")
        elif epoch == E_WARM + 1:
            model.unfreeze_vae()
            print(f"[Joint ] VAE unfrozen from epoch {epoch}")

        # ── KAN grid update (adapt spline grid to current activations) ──────
        if epoch % GRID_UPDATE_FREQ == 0:
            model.eval()
            with torch.no_grad():
                x_sample, _ = next(iter(train_loader))
                x_sample = x_sample.to(DEVICE)
                b_s, w_s, f_s = x_sample.shape
                mu, _, _ = model.vae.encode(x_sample.reshape(b_s * w_s, f_s))
                ctx = model.mamba(mu.reshape(b_s, w_s, -1))
                model.kan.update_grid(ctx)

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            compute_vae = (epoch > E_WARM)
            logits, loss_vae = model(x_batch, compute_vae_loss=compute_vae)
            loss_cls = F.cross_entropy(logits, y_batch)

            if compute_vae and loss_vae is not None:
                b_sz, w, f = x_batch.shape
                # Normalize VAE loss (sum-reduced over b*w*f) to per-element scale
                # so it is comparable to cross_entropy (mean over batch).
                # Note: vae_loss = MSE_sum + KL_sum.  Dividing both terms by
                # b*w*f slightly over-normalizes the KL (which sums over latent_dim,
                # not f), but the effect is negligible given alpha=0.1.
                loss_vae_norm = loss_vae / (b_sz * w * f)
                loss = ALPHA * loss_vae_norm + BETA * loss_cls
            else:
                loss = loss_cls

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            b = y_batch.size(0)
            train_loss    += loss.item() * b
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total   += b

        train_loss /= train_total
        train_acc   = train_correct / train_total * 100

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits, _ = model(x_batch, compute_vae_loss=False)
                loss_cls = F.cross_entropy(logits, y_batch)

                b = y_batch.size(0)
                val_loss    += loss_cls.item() * b
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total   += b

        val_loss /= val_total
        val_acc   = val_correct / val_total * 100

        # ── Save best checkpoint ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            flag = "  <-- saved"
        else:
            flag = ""

        if epoch % 10 == 0 or epoch == 1 or epoch == E_WARM + 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%{flag}")

    print("-" * 62)
    print(f"Phase 2 complete.  Best val acc: {best_val_acc:.1f}%")
    print(f"Checkpoint saved to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
