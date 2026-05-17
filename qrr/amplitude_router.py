"""
Amplitude Router — complex amplitude evolution between branches.

In QRR, each branch k carries a complex amplitude α^(k) ∈ ℂ.
The amplitude encodes not just magnitude (probability) but also phase,
which enables interference: branches with opposite phase cancel;
branches with aligned phase reinforce.

Amplitude update rule:
  α̃^(k)_{t+1} = Σ_j M_kj(x_t, c_t) · α^(j)_t

where M is a complex-valued K×K routing matrix.

Phase alignment between two branches:
  cos(θ_k - θ_j) > 0  →  constructive interference
  cos(θ_k - θ_j) < 0  →  destructive interference

This is the key mechanism that distinguishes QRR from a standard
Mixture of Experts, which has no phase and therefore no interference.

Curriculum:
  Stage 1 (early training): imaginary parts zeroed → real-valued MoE.
  Stage 2 (later training): complex phases activated → full QRR.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math


class AmplitudeRouter(nn.Module):
    """
    Context-conditioned complex amplitude router.

    Args:
        k_branches:    Number of branches K.
        context_dim:   Dimensionality of context vector.
        use_complex:   If False (Stage 1), imaginary parts are zeroed.
                       Set to True for Stage 2 (complex phase training).
        init_scale:    Scale of initial routing weights.
    """

    def __init__(
        self,
        k_branches: int = 4,
        context_dim: int = 768,
        use_complex: bool = False,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = k_branches
        self.use_complex = use_complex

        # Project context → K×K complex routing matrix (real + imag)
        # Output: 2 * K * K values (real part + imaginary part)
        self.router_real = nn.Linear(context_dim, k_branches * k_branches)
        self.router_imag = nn.Linear(context_dim, k_branches * k_branches)

        # Initialize close to identity (branches start independent)
        nn.init.eye_(self.router_real.weight.view(k_branches * k_branches, -1)[:k_branches * k_branches, :k_branches * k_branches] if context_dim >= k_branches * k_branches else self.router_real.weight)
        nn.init.zeros_(self.router_real.bias)
        nn.init.zeros_(self.router_imag.weight)
        nn.init.zeros_(self.router_imag.bias)

        self._scale = init_scale

    def forward(
        self,
        amplitudes: torch.Tensor,   # (batch, K, 2) — [real, imag]
        context: torch.Tensor,      # (batch, context_dim)
    ) -> torch.Tensor:
        """
        Route complex amplitudes through context-conditioned M.

        Returns: updated amplitudes, shape (batch, K, 2)
        """
        batch = context.size(0)

        # Build routing matrix M ∈ ℂ^{K×K}
        M_real = self.router_real(context).view(batch, self.K, self.K)
        if self.use_complex:
            M_imag = self.router_imag(context).view(batch, self.K, self.K)
        else:
            M_imag = torch.zeros_like(M_real)

        # Normalize M rows to prevent amplitude explosion
        M_norm = (M_real ** 2 + M_imag ** 2).sum(dim=-1, keepdim=True).sqrt() + 1e-9
        M_real = M_real / M_norm
        M_imag = M_imag / M_norm

        # Extract input real and imaginary parts
        a_real = amplitudes[..., 0]  # (batch, K)
        a_imag = amplitudes[..., 1]  # (batch, K)

        # Complex matrix-vector multiply:
        # (M_real + i·M_imag)(a_real + i·a_imag)
        # = (M_real·a_real - M_imag·a_imag) + i(M_real·a_imag + M_imag·a_real)
        out_real = (torch.bmm(M_real, a_real.unsqueeze(-1)).squeeze(-1)
                    - torch.bmm(M_imag, a_imag.unsqueeze(-1)).squeeze(-1))
        out_imag = (torch.bmm(M_real, a_imag.unsqueeze(-1)).squeeze(-1)
                    + torch.bmm(M_imag, a_real.unsqueeze(-1)).squeeze(-1))

        return torch.stack([out_real, out_imag], dim=-1)  # (batch, K, 2)

    def phase_interference_map(
        self, amplitudes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute K×K matrix of cosine phase differences between all branch pairs.
        Values > 0: constructive interference.
        Values < 0: destructive interference.

        Returns: (batch, K, K) cosine similarity of phases.
        """
        a_real = amplitudes[..., 0]  # (batch, K)
        a_imag = amplitudes[..., 1]  # (batch, K)
        # Phase angle θ_k = atan2(imag, real)
        phases = torch.atan2(a_imag, a_real)  # (batch, K)
        # cos(θ_k - θ_j) via outer difference
        diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # (batch, K, K)
        return torch.cos(diff)

    def activate_complex_phases(self) -> None:
        """Switch from Stage 1 (real) to Stage 2 (complex). Call after Stage 1 convergence."""
        self.use_complex = True
