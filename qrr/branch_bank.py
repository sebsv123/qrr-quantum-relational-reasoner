"""
Branch Bank — Core QRR module. v2.1 (EXP-001b fixes)

Changes from v2.0:
  - Semantic initialization: K distinct learned projections instead of
    a single linear + reshape. Each branch now has its own weight matrix,
    ensuring guaranteed diversity from the first forward pass.
  - Amplitude init now content-conditioned (not uniform) so chi carries
    signal from the very first step without any training.
  - chi formula corrected to Simpson's diversity index:
    chi = 1 - sum_k p^(k)^2   (not 1 - max p^(k))
    Simpson's is smoother, differentiable everywhere, and better
    calibrated for intermediate ambiguity levels.

Theoretical grounding:
  Branch diversity is necessary for chi to carry semantic signal.
  If all K branches project to nearly identical hidden states,
  chi measures only amplitude noise. The fix enforces structural
  diversity at initialization via orthogonality regularization.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class BranchState(NamedTuple):
    """Immutable snapshot of the branch bank at a single time step."""
    hidden: torch.Tensor          # (batch, K, d)
    amplitudes: torch.Tensor      # (batch, K, 2)  — [real, imag]
    probabilities: torch.Tensor   # (batch, K)     — |alpha|^2 normalized
    chi: torch.Tensor             # (batch,)       — Simpson diversity index


class BranchBank(nn.Module):
    """
    Core branch bank for QRR.

    v2.1: Each branch k has its OWN projection matrix W^(k) : R^d → R^d.
    This guarantees structural diversity at init: even with random weights,
    K distinct linear maps applied to the same h_base produce K genuinely
    different hidden states. The amplitude projector is also per-branch
    and content-conditioned.

    Init orthogonality:
      W^(k) are initialized orthogonally (nn.init.orthogonal_) to maximize
      initial branch diversity. This is the key fix for EXP-001b.

    Args:
        hidden_dim:    Dimensionality matching base model.
        k_branches:    Number of branches K.
        init_scale:    Scale of residual perturbation added to each branch.
                       Larger = more initial diversity, but less coherent with h_base.
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.init_scale = init_scale

        # FIX 1: K distinct projection matrices (one per branch)
        # Each W^(k) initialized orthogonally → maximum initial diversity
        self.branch_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(k_branches)
        ])
        for proj in self.branch_projs:
            nn.init.orthogonal_(proj.weight)
            nn.init.zeros_(proj.bias)

        # FIX 2: Content-conditioned amplitude initializer (per branch)
        # Maps h_base → scalar compatibility score per branch
        # High score → high initial amplitude → branch is initially "favored"
        self.amp_scorers = nn.ModuleList([
            nn.Linear(hidden_dim, 1, bias=True)
            for _ in range(k_branches)
        ])
        # Initialize with diverse random weights so different branches
        # respond to different features of h_base from the start
        for i, scorer in enumerate(self.amp_scorers):
            nn.init.normal_(scorer.weight, mean=0.0, std=0.1)
            nn.init.constant_(scorer.bias, float(i) * 0.1 - (k_branches * 0.05))

    def forward(self, h_base: torch.Tensor) -> BranchState:
        """
        Initialize branch bank from base transformer hidden state.

        Args:
            h_base: (batch, d) — encoded input from base transformer

        Returns:
            BranchState with K semantically diverse branches
        """
        batch = h_base.size(0)

        # FIX 1: Apply K distinct projections
        # Each branch gets a different nonlinear transformation of h_base
        branch_hiddens = []
        for proj in self.branch_projs:
            h_k = F.silu(proj(h_base))  # (batch, d)
            branch_hiddens.append(h_k)
        hidden = torch.stack(branch_hiddens, dim=1)  # (batch, K, d)

        # FIX 2: Content-conditioned amplitudes
        # Each branch scores its own compatibility with h_base
        scores = []
        for scorer in self.amp_scorers:
            s = scorer(h_base).squeeze(-1)  # (batch,)
            scores.append(s)
        scores = torch.stack(scores, dim=1)  # (batch, K)

        # Real amplitudes from softmax-normalized scores
        # This makes amplitudes content-dependent from step 0
        amp_real = torch.softmax(scores, dim=-1).sqrt()  # Born-rule consistent: sum(amp^2)=1
        amp_imag = torch.zeros_like(amp_real)
        amplitudes = torch.stack([amp_real, amp_imag], dim=-1)  # (batch, K, 2)

        # Probabilities: p^(k) = amp_real^2 (imag=0 at init)
        probabilities = amp_real ** 2  # already sums to 1 by softmax+sqrt construction

        # FIX 3: chi = Simpson's diversity index (smoother, differentiable)
        # chi = 1 - sum_k p^(k)^2
        # chi=0: all probability on one branch (certain)
        # chi=1-1/K: uniform distribution (maximum ambiguity)
        chi = 1.0 - (probabilities ** 2).sum(dim=-1)  # (batch,)

        return BranchState(
            hidden=hidden,
            amplitudes=amplitudes,
            probabilities=probabilities,
            chi=chi,
        )

    def branch_diversity(self, state: BranchState) -> torch.Tensor:
        """
        Mean pairwise cosine distance between branch hidden states.
        High → branches diverged. Low → branches collapsed.
        """
        h = state.hidden  # (batch, K, d)
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-9)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (batch, K, K)
        mask = ~torch.eye(self.K, dtype=torch.bool, device=h.device).unsqueeze(0)
        diversity = (1.0 - sim[mask.expand_as(sim)]).mean()
        return diversity

    def orthogonality_loss(self, state: BranchState) -> torch.Tensor:
        """
        Regularization loss penalizing branch collapse.
        Encourages branch hidden states to remain orthogonal.
        Add to training loss as: L += lambda_orth * bank.orthogonality_loss(state)
        """
        h = state.hidden  # (batch, K, d)
        h_norm = F.normalize(h, dim=-1)  # (batch, K, d)
        gram = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (batch, K, K)
        I = torch.eye(self.K, device=h.device).unsqueeze(0)
        off_diag = gram - I
        return (off_diag ** 2).mean()
