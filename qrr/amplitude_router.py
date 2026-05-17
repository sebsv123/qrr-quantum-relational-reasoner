"""
Amplitude Router — complex routing of branch amplitudes.

Analog of quantum phase shifts and amplitude redistribution.
Routes amplitude flow between branches based on semantic compatibility
between the current token and each branch's hypothesis.

Key idea: branches that are semantically compatible with the current
token receive constructive interference (amplitude boost);
incompatible branches receive destructive interference (amplitude reduction).

This is the QRR analog of:
  |ψ_new⟩ = M(x_t, c_t) |ψ⟩
where M is a context-dependent complex mixing matrix.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from qrr.branch_bank import BranchState


class AmplitudeRouter(nn.Module):
    """
    Routes complex amplitudes between branches based on
    semantic compatibility with the current token.

    Computes a compatibility score s^(k)(x_t) ∈ [-1, 1] for each branch,
    then applies constructive/destructive interference:
      α̃^(k) = α^(k) · exp(i · π · s^(k))

    Args:
        hidden_dim:  Branch hidden state dimensionality.
        token_dim:   Token embedding dimensionality (usually = hidden_dim).
        k_branches:  Number of branches.
        phase_scale: Maximum phase shift in radians (default π).
    """

    def __init__(
        self,
        hidden_dim: int,
        token_dim: int | None = None,
        k_branches: int = 4,
        phase_scale: float = math.pi,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.phase_scale = phase_scale
        token_dim = token_dim or hidden_dim

        # Compatibility scorer: token × branch_hidden → score ∈ [-1, 1]
        # For each branch k, scores how compatible the current token is
        self.compat_net = nn.Sequential(
            nn.Linear(token_dim + hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # output ∈ (-1, 1)
        )

        # Amplitude redistribution: soft cross-branch flow
        self.flow_net = nn.Sequential(
            nn.Linear(token_dim, k_branches * k_branches),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        state: BranchState,
        token_emb: torch.Tensor,  # (batch, token_dim) — current token embedding
    ) -> BranchState:
        """
        Apply amplitude routing based on current token.

        Args:
            state:     Current BranchState
            token_emb: Embedding of current token (batch, token_dim)

        Returns:
            Updated BranchState with phase-shifted amplitudes
        """
        batch = token_emb.size(0)
        h = state.hidden        # (batch, K, d)
        alpha = state.amplitudes  # (batch, K, 2)

        # 1. Compute compatibility score for each branch
        # Expand token to match branches
        token_exp = token_emb.unsqueeze(1).expand(-1, self.K, -1)  # (batch, K, token_dim)
        compat_input = torch.cat([token_exp, h], dim=-1)           # (batch, K, token_dim+d)
        compat_input_flat = compat_input.view(batch * self.K, -1)
        scores = self.compat_net(compat_input_flat).view(batch, self.K)  # (batch, K) ∈ (-1,1)

        # 2. Phase shift: φ^(k) = phase_scale · s^(k)
        # Complex multiplication: α · e^{iφ} = (a+ib)(cos φ + i sin φ)
        phi = self.phase_scale * scores  # (batch, K)
        cos_phi = phi.cos()              # (batch, K)
        sin_phi = phi.sin()              # (batch, K)

        a = alpha[..., 0]  # real part (batch, K)
        b = alpha[..., 1]  # imag part (batch, K)

        # (a + ib)(cos φ + i sin φ) = (a cos φ - b sin φ) + i(a sin φ + b cos φ)
        a_new = a * cos_phi - b * sin_phi
        b_new = a * sin_phi + b * cos_phi

        # 3. Cross-branch amplitude flow (redistribution)
        # Some amplitude flows from one branch to another based on token
        flow = self.flow_net(token_emb).view(batch, self.K, self.K)  # (batch, K, K)
        a_new = torch.bmm(flow, a_new.unsqueeze(-1)).squeeze(-1)
        b_new = torch.bmm(flow, b_new.unsqueeze(-1)).squeeze(-1)

        alpha_new = torch.stack([a_new, b_new], dim=-1)  # (batch, K, 2)

        # 4. Update probabilities and χ
        norm_sq = a_new ** 2 + b_new ** 2
        probs_new = norm_sq / (norm_sq.sum(dim=-1, keepdim=True) + 1e-9)
        chi_new = 1.0 - probs_new.max(dim=-1).values

        from qrr.branch_bank import BranchState
        return BranchState(
            hidden=state.hidden,  # hidden states unchanged (mixer handles that)
            amplitudes=alpha_new,
            probabilities=probs_new,
            chi=chi_new,
        )

    def interference_pattern(
        self, state: BranchState, token_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the compatibility scores (phase shifts / π) for visualization.
        Positive = constructive interference, negative = destructive.
        Shape: (batch, K)
        """
        batch = token_emb.size(0)
        h = state.hidden
        token_exp = token_emb.unsqueeze(1).expand(-1, self.K, -1)
        compat_input = torch.cat([token_exp, h], dim=-1).view(batch * self.K, -1)
        return self.compat_net(compat_input).view(batch, self.K)
