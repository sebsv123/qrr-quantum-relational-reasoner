"""
Branch Bank — Core QRR module.

Maintains K latent hypotheses (branches) with complex amplitudes.
Each branch is a hidden state vector h^(k) ∈ R^d paired with a
complex amplitude α^(k) ∈ C.

The probability of each branch is |α^(k)|^2 (Born rule analog).
The collapse index χ = 1 - max_k |α^(k)|^2 / sum_j |α^(j)|^2
measures residual ambiguity across branches.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import NamedTuple


class BranchState(NamedTuple):
    """Immutable snapshot of the branch bank at a single time step."""
    hidden: torch.Tensor          # shape: (batch, K, d)
    amplitudes: torch.Tensor      # shape: (batch, K, 2)  — [real, imag]
    probabilities: torch.Tensor   # shape: (batch, K)     — |α|^2 normalized
    chi: torch.Tensor             # shape: (batch,)       — collapse index ∈ [0,1]


class BranchBank(nn.Module):
    """
    Core branch bank for QRR.

    Args:
        hidden_dim: Dimensionality of each branch hidden state (must match base model).
        k_branches:  Number of competing hypotheses to maintain.
        init_uniform: If True, initialize amplitudes uniformly (equal superposition).
                      If False, initialize with small random perturbations.
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        init_uniform: bool = True,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches

        # Learnable branch projection: maps base hidden state to K branch states
        self.branch_proj = nn.Linear(hidden_dim, k_branches * hidden_dim, bias=False)

        # Learnable amplitude initializer: maps base hidden state to K complex amplitudes
        self.amp_proj = nn.Linear(hidden_dim, k_branches * 2)  # real + imag per branch

        self.init_uniform = init_uniform

    def forward(self, h_base: torch.Tensor) -> BranchState:
        """
        Initialize or update branch bank from base transformer hidden state.

        Args:
            h_base: shape (batch, d) — pooled hidden state from base transformer

        Returns:
            BranchState with K branches initialized from h_base
        """
        batch = h_base.size(0)

        # Project to K branch hidden states
        hidden = self.branch_proj(h_base)              # (batch, K*d)
        hidden = hidden.view(batch, self.K, self.d)    # (batch, K, d)

        # Compute complex amplitudes
        if self.init_uniform:
            # Equal superposition: all branches start with equal weight
            amp_real = torch.full((batch, self.K), 1.0 / (self.K ** 0.5), device=h_base.device)
            amp_imag = torch.zeros_like(amp_real)
        else:
            raw = self.amp_proj(h_base).view(batch, self.K, 2)  # (batch, K, 2)
            amp_real = raw[..., 0]
            amp_imag = raw[..., 1]

        amplitudes = torch.stack([amp_real, amp_imag], dim=-1)  # (batch, K, 2)

        # Compute probabilities: p^(k) = |α^(k)|^2 / sum_j |α^(j)|^2
        norm_sq = amp_real ** 2 + amp_imag ** 2               # (batch, K)
        probabilities = norm_sq / (norm_sq.sum(dim=-1, keepdim=True) + 1e-9)

        # Collapse index: χ = 1 - max_k p^(k)
        chi = 1.0 - probabilities.max(dim=-1).values          # (batch,)

        return BranchState(
            hidden=hidden,
            amplitudes=amplitudes,
            probabilities=probabilities,
            chi=chi,
        )

    def branch_diversity(self, state: BranchState) -> torch.Tensor:
        """
        Compute mean pairwise cosine distance between branch hidden states.
        High diversity = branches have diverged; low = branches collapsed.

        Returns: scalar tensor (mean over batch)
        """
        h = state.hidden  # (batch, K, d)
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-9)
        # Cosine similarity matrix: (batch, K, K)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
        # Mean off-diagonal distance
        mask = ~torch.eye(self.K, dtype=torch.bool, device=h.device).unsqueeze(0)
        diversity = (1.0 - sim[mask.expand_as(sim)]).mean()
        return diversity
