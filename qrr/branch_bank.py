"""
Branch Bank — Core module of QRR.

Maintains K competing latent hypotheses (branches) with complex amplitudes.
Each branch represents a coherent interpretation of the input context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BranchBank(nn.Module):
    """
    Maintains K latent branches with complex amplitudes.

    Args:
        d_model: Hidden dimension of the transformer backbone.
        K: Number of branches (default 8).
        use_complex_phase: Whether to use full complex amplitudes.
                           If False, uses real amplitudes only (curriculum phase 1).
        sparsity_k: Max number of active branches (top-k sparsity constraint).
    """

    def __init__(
        self,
        d_model: int,
        K: int = 8,
        use_complex_phase: bool = False,
        sparsity_k: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.use_complex_phase = use_complex_phase
        self.sparsity_k = sparsity_k or K

        # Branch initialization projections: map transformer hidden to K branches
        self.branch_projections = nn.Linear(d_model, K * d_model)

        # Amplitude head: produces (magnitude, phase) per branch
        self.amplitude_head_r = nn.Linear(d_model, K)  # magnitude (r)
        if use_complex_phase:
            self.amplitude_head_phi = nn.Linear(d_model, K)  # phase (φ)
        else:
            self.register_parameter('amplitude_head_phi', None)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Transformer hidden states (B, L, d_model)

        Returns:
            H: Branch hidden states (B, K, d_model)
            alpha: Complex amplitudes (B, K) — as (real, imag) or real-only
        """
        # Use [CLS]-equivalent or mean pooling for amplitude routing
        x_pooled = x.mean(dim=1)  # (B, d_model)

        # Project to K branches
        H = self.branch_projections(x_pooled)  # (B, K * d_model)
        H = H.view(-1, self.K, self.d_model)   # (B, K, d_model)

        # Compute amplitudes
        r = F.softplus(self.amplitude_head_r(x_pooled))  # (B, K), magnitudes > 0

        if self.use_complex_phase and self.amplitude_head_phi is not None:
            phi = self.amplitude_head_phi(x_pooled)  # (B, K), phases
            # Complex amplitude: α = r * e^(iφ)
            alpha_real = r * torch.cos(phi)  # (B, K)
            alpha_imag = r * torch.sin(phi)  # (B, K)
            alpha = torch.complex(alpha_real, alpha_imag)  # (B, K) complex
        else:
            alpha = r  # (B, K) real

        # Normalize: Σ_k |α_k|² = 1
        alpha = self._normalize_amplitudes(alpha)

        # Apply top-k sparsity if needed
        if self.sparsity_k < self.K:
            alpha = self._apply_sparsity(alpha)

        return H, alpha

    def _normalize_amplitudes(
        self, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Normalize so that Σ_k |α_k|² = 1."""
        if torch.is_complex(alpha):
            norm = torch.sqrt((alpha.abs() ** 2).sum(dim=-1, keepdim=True))
        else:
            norm = torch.sqrt((alpha ** 2).sum(dim=-1, keepdim=True))
        return alpha / (norm + 1e-8)

    def _apply_sparsity(
        self, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Zero out all but top-k branches by magnitude."""
        magnitudes = alpha.abs() if torch.is_complex(alpha) else alpha.abs()
        _, top_indices = magnitudes.topk(self.sparsity_k, dim=-1)
        mask = torch.zeros_like(magnitudes)
        mask.scatter_(-1, top_indices, 1.0)
        if torch.is_complex(alpha):
            mask = mask.to(alpha.real.dtype)
            alpha = torch.complex(alpha.real * mask, alpha.imag * mask)
        else:
            alpha = alpha * mask
        return self._normalize_amplitudes(alpha)


def compute_chi(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute collapse index χ ∈ [0, 1].

    χ = 1 - Σ|α|⁴ / (Σ|α|²)²

    χ = 0: one branch dominates (collapsed)
    χ = 1: maximum superposition (uniform)

    Args:
        alpha: (B, K) real or complex amplitudes (normalized)

    Returns:
        chi: (B,) collapse index
    """
    probs = alpha.abs() ** 2 if torch.is_complex(alpha) else alpha ** 2
    sum_p2 = (probs ** 2).sum(dim=-1)          # Σ|α|⁴
    sum_p = probs.sum(dim=-1)                   # Σ|α|²
    chi = 1.0 - sum_p2 / (sum_p ** 2 + 1e-8)
    return chi
