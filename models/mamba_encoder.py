"""
Mamba (S6) Temporal Encoder — pure PyTorch implementation.
No mamba-ssm CUDA extension required.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space model (S6).

    Input/Output: (batch, seq_len, d_model)

    The selective mechanism computes input-dependent Δ, B, C matrices,
    enabling the model to focus on task-relevant time steps.
    Dropout (p=0.15) is applied after the depthwise convolution, within the block.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection: split into two branches (x and z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True
        )

        # Dropout applied after depthwise conv (within block)
        self.dropout = nn.Dropout(dropout)

        # Projections for SSM parameters (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Fixed SSM A matrix (log-parameterized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Skip connection scalar
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self._last_delta = None

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        """
        Args:
            x: (batch, seq_len, d_model)
            return_delta: if True, also return delta values for interpretability
        Returns:
            out: (batch, seq_len, d_model)
            delta (optional): (batch, seq_len, d_inner) — per-step selectivity
        """
        residual = x
        x = self.norm(x)
        b, l, _ = x.shape

        # Split into SSM branch (x_part) and gating branch (z)
        xz = self.in_proj(x)                          # (b, l, 2*d_inner)
        x_part, z = xz.chunk(2, dim=-1)               # each (b, l, d_inner)

        # Causal conv1d (local context) + dropout within block
        x_part = x_part.transpose(1, 2)               # (b, d_inner, l)
        x_part = self.conv1d(x_part)[:, :, :l]        # trim causal padding
        x_part = x_part.transpose(1, 2)               # (b, l, d_inner)
        x_part = self.dropout(F.silu(x_part))          # dropout after conv, within block

        y, delta = self._ssm(x_part)
        y = y * F.silu(z)

        self._last_delta = delta.detach()

        out = self.out_proj(y) + residual
        if return_delta:
            return out, delta
        return out

    def _ssm(self, x: torch.Tensor):
        """
        Args:
            x: (b, l, d_inner)
        Returns:
            y: (b, l, d_inner)
            delta: (b, l, d_inner)
        """
        b, l, d = x.shape
        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # Compute input-dependent Δ, B, C
        x_dbl = self.x_proj(x)                        # (b, l, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))           # (b, l, d_inner) — selectivity

        y = self._selective_scan(x, delta, A, B, C, self.D)
        return y, delta

    @staticmethod
    def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Hillis-Steele parallel prefix scan solving h_k = a_k * h_{k-1} + b_k
        with h_{-1} = 0.  O(log L) sequential depth, fully vectorized on CUDA.

        Equivalent to the sequential loop::

            h = zeros(...)
            for k in range(L):
                h = a[:, k] * h + b[:, k]
                out[:, k] = h

        but replaces L sequential kernel launches with ceil(log2 L) vectorized
        passes via the associative composition operator:
            (a2, b2) ∘ (a1, b1) = (a2·a1,  a2·b1 + b2)
        with identity (1, 0).

        Args:
            a, b: (batch, L, d_inner, n)  — scan over dim=1
        Returns:
            h:    (batch, L, d_inner, n)
        """
        h_a, h_b = a, b
        L = a.shape[1]
        d = 1
        while d < L:
            pad_a = a.new_ones( a.shape[0], d, *a.shape[2:])
            pad_b = b.new_zeros(b.shape[0], d, *b.shape[2:])
            shifted_a = torch.cat([pad_a, h_a[:, :-d]], dim=1)
            shifted_b = torch.cat([pad_b, h_b[:, :-d]], dim=1)
            # Evaluate both using the OLD h_a before updating it
            h_b = h_a * shifted_b + h_b
            h_a = h_a * shifted_a
            d *= 2
        return h_b

    @staticmethod
    def _selective_scan(u, delta, A, B, C, D):
        """
        Discretized SSM scan via parallel prefix scan (hardware-aware,
        O(log L) sequential depth — efficient on CUDA without Python time loops).

        Args:
            u:     (b, l, d_in)
            delta: (b, l, d_in)   — input-dependent discretization step
            A:     (d_in, n)      — state matrix (negative, log-parameterized)
            B:     (b, l, n)
            C:     (b, l, n)
            D:     (d_in,)        — skip connection
        Returns:
            y: (b, l, d_in)
        """
        b, l, d_in = u.shape
        n = A.shape[1]

        # Discretize: Abar = exp(Δ·A),  Bbar·u = Δ·B·u
        deltaA   = torch.exp(delta.unsqueeze(-1) * A[None, None])           # (b, l, d, n)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)   # (b, l, d, n)

        # Parallel scan: h[:,k] = deltaA[:,k] * h[:,k-1] + deltaB_u[:,k]
        h = MambaBlock._parallel_scan(deltaA, deltaB_u)                      # (b, l, d, n)

        y = (h * C.unsqueeze(2)).sum(-1)                                     # (b, l, d)
        return y + u * D[None, None]


class MambaEncoder(nn.Module):
    """
    Stacked Mamba blocks for encoding a temporal sequence of dFC latent vectors.

    Input:  (batch, seq_len, d_model)   — sequence of VAE latent vectors
    Output: (batch, d_model)            — mean-pooled temporal context
    """

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_deltas: bool = False):
        """
        Args:
            x: (batch, seq_len, d_model)
            return_deltas: if True, return delta tensors from all layers
        Returns:
            ctx: (batch, d_model)   — mean-pooled temporal context
            deltas (optional): list of (batch, seq_len, d_inner) per layer
        """
        deltas = []
        for layer in self.layers:
            if return_deltas:
                x, delta = layer(x, return_delta=True)
                deltas.append(delta)
            else:
                x = layer(x)

        x = self.norm(x)
        ctx = x.mean(dim=1)   # mean pooling over time

        if return_deltas:
            return ctx, deltas
        return ctx

    def get_temporal_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-timestep importance scores averaged over all layers and d_inner.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            importance: (batch, seq_len)  — higher = more selective attention
        """
        _, deltas = self.forward(x, return_deltas=True)
        stacked = torch.stack(deltas, dim=0)           # (n_layers, b, l, d_inner)
        return stacked.mean(dim=0).mean(dim=-1)        # (b, l)


if __name__ == "__main__":
    torch.manual_seed(0)
    b, seq_len, d_model = 8, 54, 128
    x = torch.randn(b, seq_len, d_model)

    encoder = MambaEncoder(d_model=d_model, n_layers=2, d_state=16)
    ctx = encoder(x)
    print("Context shape:", ctx.shape)        # (8, 128)

    imp = encoder.get_temporal_importance(x)
    print("Importance shape:", imp.shape)     # (8, 54)
    print("Importance range:", imp.min().item(), "~", imp.max().item())
