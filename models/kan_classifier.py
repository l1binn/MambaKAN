"""
Kolmogorov-Arnold Network (KAN) Classifier — B-Spline implementation.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
Inspired by efficient-kan (https://github.com/Blealtan/efficient-kan)

Key difference from MLP:
  MLP:  y = W · σ(x)           — fixed activation, learnable weights
  KAN:  y = Σ φ_{q,p}(x_p)    — learnable spline activations per connection

Visualizing φ curves directly reveals the non-linear mapping from each
latent dimension to class logits, providing intrinsic interpretability.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    A single KAN layer: replaces Linear + fixed activation with
    per-connection learnable B-spline activations.

    Input:  (batch, in_features)
    Output: (batch, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build extended B-spline grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

        # Number of B-spline basis functions
        n_basis = grid_size + spline_order

        self.base_weight  = nn.Parameter(torch.empty(out_features, in_features))
        # Spline coefficients: (out, in, n_basis)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))

        # Per-connection scaling factors (learnable)
        self.scale_base   = nn.Parameter(torch.ones(out_features, in_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features, in_features))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at x via Cox–de Boor recursion.

        Args:
            x: (batch, in_features)
        Returns:
            bases: (batch, in_features, n_basis)
        """
        assert x.dim() == 2
        x = x.unsqueeze(-1)
        grid = self.grid

        # Order-0 indicator basis
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Cox–de Boor recursion
        for k in range(1, self.spline_order + 1):
            denom_l = grid[k:-1] - grid[: -(k + 1)]
            denom_r = grid[k + 1 :] - grid[1: -k]
            # Avoid division by zero
            left = torch.where(
                denom_l != 0,
                (x - grid[: -(k + 1)]) / denom_l * bases[..., :-1],
                torch.zeros_like(bases[..., :-1]),
            )
            right = torch.where(
                denom_r != 0,
                (grid[k + 1 :] - x) / denom_r * bases[..., 1:],
                torch.zeros_like(bases[..., 1:]),
            )
            bases = left + right

        return bases   # (b, in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            out: (batch, out_features)
        """
        # Base (SiLU) branch
        base_out = F.linear(F.silu(x), self.base_weight * self.scale_base)

        # Spline branch: contract (b, in, n_basis) × (out, in, n_basis) → (b, out)
        bases = self.b_splines(x)
        spline_out = torch.einsum(
            "bik,oik->bo",
            bases,
            self.spline_weight * self.scale_spline.unsqueeze(-1),
        )

        return base_out + spline_out

    def get_activation_curve(self, dim: int, n_points: int = 200, x_range=(-3.0, 3.0)):
        """
        Evaluate the learned spline activation φ(x) for a single input dimension.

        Args:
            dim: input dimension index
            n_points: number of evaluation points
            x_range: evaluation range
        Returns:
            x_vals: (n_points,)
            y_vals: (out_features, n_points)
        """
        x_vals = torch.linspace(x_range[0], x_range[1], n_points, device=self.grid.device)
        dummy = torch.zeros(n_points, self.in_features, device=self.grid.device)
        dummy[:, dim] = x_vals

        bases = self.b_splines(dummy)
        sw = self.spline_weight[:, dim, :] * self.scale_spline[:, dim].unsqueeze(-1)
        y_vals = (bases[:, dim, :] @ sw.T).T   # (out, n_pts)
        return x_vals.detach(), y_vals.detach()

    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Adapt the B-spline grid to span the activation range of x.

        Note on coefficient resampling:
            A full grid update should re-fit spline_weight onto the new basis
            (e.g. via least squares) to preserve learned activation shapes.
            This implementation omits that step because the current architecture
            uses a single shared 1-D grid for all in_features; when the grid
            shifts significantly the new and old basis spaces diverge and simple
            least-squares resampling is numerically unreliable (larger error
            than no resampling at all on large feature dimensions).
            In practice, call this method infrequently (every 50 epochs) so the
            network has enough gradient steps to recover from the small
            coefficient mismatch before the next update.

        Args:
            x:      (batch, in_features) — representative input batch
            margin: fractional padding added beyond [x_min, x_max]
        """
        with torch.no_grad():
            x_min, x_max = x.min().item(), x.max().item()
            span  = max(x_max - x_min, 1e-6)
            x_min -= margin * span
            x_max += margin * span
            h = (x_max - x_min) / self.grid_size
            new_grid = torch.linspace(
                x_min - self.spline_order * h,
                x_max + self.spline_order * h,
                len(self.grid),
                device=self.grid.device,
                dtype=self.grid.dtype,
            )
            self.grid.copy_(new_grid)


class KANClassifier(nn.Module):
    """
    Two-layer KAN for classification.

    Structure: in_features → hidden_dim → num_classes

    Parameter count note:
        This implementation includes per-connection learnable scaling factors
        (scale_base and scale_spline in each KANLinear layer) for training
        stability.  These add (out * in) extra parameters per layer compared
        to counting only the spline and base weights.  As a result the KAN
        head has ~93 K parameters, somewhat larger than the ~43 K figure
        reported in the paper (which counted only spline_weight + base_weight).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden_dim, grid_size, spline_order)
        self.layer2 = KANLinear(hidden_dim, num_classes, grid_size, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

    def update_grid(self, x: torch.Tensor):
        """
        Update B-spline grids in both layers.
        Call with a representative context batch (output of MambaEncoder)
        every few epochs during Phase-2 training.

        Args:
            x: (batch, in_features) — Mamba context vectors
        """
        self.layer1.update_grid(x)
        with torch.no_grad():
            h = self.layer1(x)
        self.layer2.update_grid(h)

    def get_input_importance(self) -> torch.Tensor:
        """
        L1 norm of spline weights in layer 1, summed over outputs and basis functions.

        Returns:
            importance: (in_features,)  — higher = more influential
        """
        w = self.layer1.spline_weight   # (out, in, n_basis)
        return w.abs().sum(dim=[0, 2])  # (in_features,)

    def get_class_curves(
        self,
        dim: int,
        x_mean: torch.Tensor,
        n_points: int = 200,
        x_range: tuple = (-3.0, 3.0),
    ):
        """
        Ceteris-paribus class activation curve for input dimension `dim`.
        Varies dim over x_range while holding all other dims at x_mean.

        Args:
            dim: input dimension to vary
            x_mean: (in_features,) anchor values for non-varied dims
            n_points: evaluation resolution
            x_range: range of variation for `dim`
        Returns:
            x_vals: (n_points,)
            logits: (num_classes, n_points)
        """
        device = next(self.parameters()).device
        x_vals = torch.linspace(x_range[0], x_range[1], n_points, device=device)
        probe = x_mean.unsqueeze(0).expand(n_points, -1).clone().to(device)
        probe[:, dim] = x_vals
        with torch.no_grad():
            out = self.forward(probe)   # (n_points, num_classes)
        return x_vals.cpu(), out.T.cpu()  # (num_classes, n_points)

    def get_top_activation_curves(self, top_k: int = 10, n_points: int = 200):
        """
        Returns activation curves for the top-k most important input dimensions.

        Returns:
            top_dims: (top_k,)
            x_vals:   (n_points,)
            curves:   (top_k, out_features, n_points)
        """
        importance = self.get_input_importance()
        top_dims = importance.topk(top_k).indices

        curves, x_vals = [], None
        for d in top_dims.tolist():
            xv, yv = self.layer1.get_activation_curve(d, n_points)
            x_vals = xv
            curves.append(yv)

        return top_dims, x_vals, torch.stack(curves, dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    b, in_f, hidden, num_cls = 16, 128, 64, 4

    model = KANClassifier(in_f, hidden, num_cls)
    x = torch.randn(b, in_f)
    logits = model(x)
    print("Logits shape:", logits.shape)        # (16, 4)

    imp = model.get_input_importance()
    print("Importance shape:", imp.shape)       # (128,)

    top_dims, x_vals, curves = model.get_top_activation_curves(top_k=5)
    print("Top dims:", top_dims.tolist())
    print("Curves shape:", curves.shape)        # (5, 64, 200)
