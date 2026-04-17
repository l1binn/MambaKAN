"""
Proposed model: VAE + Mamba + KAN end-to-end framework
for Alzheimer's disease classification from fMRI dynamic functional connectivity.

Training strategy:
  Phase 1 — Unsupervised VAE pre-training (see stage1_train.py)
  Phase 2 — End-to-end fine-tuning with joint loss:
             L = alpha * L_VAE + beta * L_cls
             VAE uses a small lr to preserve pre-trained features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae import VAE, vae_loss
from models.mamba_encoder import MambaEncoder
from models.kan_classifier import KANClassifier


class ProposedModel(nn.Module):
    """
    VAE + Mamba Temporal Encoder + KAN Classifier.

    Forward input:  x  (batch, W, input_dim)
                       W = number of dFC windows (e.g. 54)
                       input_dim = upper-triangular dFC dim (e.g. 6670)
    Forward output:
        logits      (batch, num_classes)
        vae_loss    scalar or None  (only computed when compute_vae_loss=True)
    """

    def __init__(
        self,
        input_dim: int,
        vae_hidden_dims: list,
        latent_dim: int,
        mamba_n_layers: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        kan_hidden_dim: int,
        kan_grid_size: int,
        kan_spline_order: int,
        num_classes: int,
        dropout: float = 0.15,
        **kwargs,
    ):
        super().__init__()

        # ── Stage 1: VAE ────────────────────────────────────────────────
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=vae_hidden_dims,
            latent_dim=latent_dim,
        )

        # ── Stage 2a: Mamba temporal encoder ────────────────────────────
        self.mamba = MambaEncoder(
            d_model=latent_dim,
            n_layers=mamba_n_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
        )

        # ── Stage 2b: KAN classifier ─────────────────────────────────────
        self.kan = KANClassifier(
            in_features=latent_dim,
            hidden_dim=kan_hidden_dim,
            num_classes=num_classes,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, compute_vae_loss: bool = False):
        """
        Args:
            x: (batch, W, input_dim)
            compute_vae_loss: whether to compute and return VAE reconstruction loss
                              (set True during Phase-2 training, False at inference)
        Returns:
            logits:   (batch, num_classes)
            vae_loss: scalar tensor or None
        """
        b, w, f = x.shape

        # ── VAE: encode each window independently ────────────────────────
        x_flat = x.reshape(b * w, f)                   # (b*W, input_dim)
        mu, logvar, _ = self.vae.encode(x_flat)        # each (b*W, latent_dim)

        # VAE loss (reconstruction + KL) computed on reparameterized sample
        loss_vae = None
        if compute_vae_loss:
            z_sample = self.vae.reparameterize(mu, logvar)
            recon = self.vae.decode(z_sample)
            loss_vae = vae_loss(recon, x_flat, mu, logvar)

        # Use mean (deterministic) for downstream sequence modeling
        z = mu.reshape(b, w, -1)                       # (b, W, latent_dim)

        # ── Mamba: temporal sequence modeling ────────────────────────────
        ctx = self.mamba(z)                             # (b, latent_dim)
        # Note: Mamba mean-pools its W output vectors into a single context
        # vector c ∈ R^{latent_dim} before KAN classification (Eq. 10 in the
        # paper).  Classification is therefore subject-level, not window-level.
        # A sentence in the paper's dataset-partitioning section that mentions
        # "54 window-level predictions aggregated via probability averaging"
        # is a textual inaccuracy; the actual inference path is as coded here.

        # ── KAN: interpretable classification ────────────────────────────
        logits = self.kan(ctx)                          # (b, num_classes)

        return logits, loss_vae

    # ------------------------------------------------------------------
    def get_temporal_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-timestep selectivity scores (from Mamba Δ values).

        Args:
            x: (batch, W, input_dim)
        Returns:
            importance: (batch, W)
        """
        b, w, f = x.shape
        x_flat = x.reshape(b * w, f)
        with torch.no_grad():
            mu, _, _ = self.vae.encode(x_flat)
        z = mu.reshape(b, w, -1).detach()
        return self.mamba.get_temporal_importance(z)

    # ------------------------------------------------------------------
    def get_gradient_attribution(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Compute gradient of class logit w.r.t. input dFC vectors.
        Used for brain-region mapping.

        Args:
            x: (1, W, input_dim)   — single sample
            class_idx: target class
        Returns:
            attr: (W, input_dim)  — gradient attribution per window per feature
        """
        x = x.detach().requires_grad_(True)
        logits, _ = self.forward(x, compute_vae_loss=False)
        score = logits[0, class_idx]
        score.backward()
        return x.grad[0].detach()   # (W, input_dim)

    # ------------------------------------------------------------------
    def freeze_vae(self):
        """Freeze VAE parameters (for two-stage training)."""
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def unfreeze_vae(self):
        """Unfreeze VAE parameters (for end-to-end fine-tuning)."""
        for p in self.vae.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_config(cls, cfg: dict, num_classes: int):
        """Instantiate from a parsed YAML config dict."""
        return cls(
            input_dim=cfg["vae"]["input_dim"],
            vae_hidden_dims=cfg["vae"]["hidden_dims"],
            latent_dim=cfg["vae"]["latent_dim"],
            mamba_n_layers=cfg["mamba"]["n_layers"],
            mamba_d_state=cfg["mamba"]["d_state"],
            mamba_d_conv=cfg["mamba"]["d_conv"],
            mamba_expand=cfg["mamba"]["expand"],
            kan_hidden_dim=cfg["kan"]["hidden_dim"],
            kan_grid_size=cfg["kan"]["grid_size"],
            kan_spline_order=cfg["kan"]["spline_order"],
            dropout=cfg.get("model", {}).get("dropout", 0.15),
            num_classes=num_classes,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    b, W, input_dim, latent_dim, num_cls = 4, 54, 6670, 128, 4

    model = ProposedModel(
        input_dim=input_dim,
        vae_hidden_dims=[2048, 1024, 512, 256],
        latent_dim=latent_dim,
        mamba_n_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        kan_hidden_dim=64,
        kan_grid_size=5,
        kan_spline_order=3,
        num_classes=num_cls,
    )

    x = torch.randn(b, W, input_dim)
    logits, loss_vae = model(x, compute_vae_loss=True)
    print("Logits:", logits.shape)      # (4, 4)
    print("VAE loss:", loss_vae.item())

    imp = model.get_temporal_importance(x)
    print("Temporal importance:", imp.shape)   # (4, 54)

    attr = model.get_gradient_attribution(x[:1], class_idx=0)
    print("Attribution:", attr.shape)          # (54, 6670)

    # Parameter groups for differential learning rates
    vae_params = list(model.vae.parameters())
    new_params = list(model.mamba.parameters()) + list(model.kan.parameters())
    print(f"VAE params: {sum(p.numel() for p in vae_params):,}")
    print(f"Mamba+KAN params: {sum(p.numel() for p in new_params):,}")
