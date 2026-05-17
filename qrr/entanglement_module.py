"""
Entanglement Module — context-branch correlation across time.

In quantum entanglement, measuring one subsystem instantaneously
updates the state of another. In QRR, this translates to:
when a new token or observation arrives, it selectively correlates
with specific branches, updating their hidden states and amplitudes
based on contextual compatibility.

Mechanism:
  1. Compute compatibility score s_k = ⟨h^(k), c_t⟩ / √d for each branch k.
  2. Convert scores to update gates g_k = softmax(s_k / τ).
  3. Update each branch hidden state:
     h'^(k) = h^(k) + g_k · W_ent · c_t
  4. Update amplitudes by modulating their magnitude:
     |α'^(k)| = |α^(k)| · (1 + ε · g_k)
     (phase is preserved — only magnitude is updated by new evidence)

This models the intuition: a new observation that is compatible
with branch k amplifies it; incompatible observations suppress it.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math


class EntanglementModule(nn.Module):
    """
    Selective context-branch correlation update.

    Args:
        hidden_dim:    Branch hidden state dimensionality.
        k_branches:    Number of branches.
        context_dim:   Context vector dimensionality.
        temperature:   Softmax temperature τ for compatibility gates.
                       Lower τ → sharper selection (one branch dominates).
                       Higher τ → softer (all branches updated equally).
        amplitude_lr:  Scale ε for amplitude modulation (default 0.1).
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        context_dim: int | None = None,
        temperature: float = 1.0,
        amplitude_lr: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.tau = temperature
        self.eps = amplitude_lr
        context_dim = context_dim or hidden_dim

        # Project context into branch hidden space for update
        self.W_ent = nn.Linear(context_dim, hidden_dim, bias=False)
        # Learnable temperature (can be trained)
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(
        self,
        hidden: torch.Tensor,       # (batch, K, d)
        amplitudes: torch.Tensor,   # (batch, K, 2)
        context: torch.Tensor,      # (batch, context_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update branch states via context-branch entanglement.

        Returns:
            updated_hidden:     (batch, K, d)
            updated_amplitudes: (batch, K, 2)
        """
        tau = self.log_tau.exp().clamp(min=0.1, max=10.0)

        # Step 1: Compatibility scores s_k = ⟨h^(k), c_t⟩ / √d
        c_proj = self.W_ent(context)                     # (batch, d)
        c_expanded = c_proj.unsqueeze(1)                 # (batch, 1, d)
        scores = (hidden * c_expanded).sum(dim=-1) / math.sqrt(self.d)  # (batch, K)

        # Step 2: Compatibility gates
        gates = torch.softmax(scores / tau, dim=-1)      # (batch, K)  sums to 1

        # Step 3: Update hidden states — each branch gets a gated context injection
        delta_h = gates.unsqueeze(-1) * c_expanded       # (batch, K, d)
        updated_hidden = hidden + delta_h                # (batch, K, d)

        # Step 4: Modulate amplitude magnitudes (preserve phase)
        a_real = amplitudes[..., 0]                       # (batch, K)
        a_imag = amplitudes[..., 1]                       # (batch, K)
        magnitude = (a_real ** 2 + a_imag ** 2).sqrt() + 1e-9  # (batch, K)
        phase_real = a_real / magnitude
        phase_imag = a_imag / magnitude
        # New magnitude: scale by (1 + ε·g_k), then re-normalize to preserve ||α||=1
        new_magnitude = magnitude * (1.0 + self.eps * gates)
        # Re-normalize so sum of |α|² stays 1
        new_magnitude = new_magnitude / (new_magnitude.norm(dim=-1, keepdim=True) + 1e-9)
        updated_real = new_magnitude * phase_real
        updated_imag = new_magnitude * phase_imag
        updated_amplitudes = torch.stack([updated_real, updated_imag], dim=-1)

        return updated_hidden, updated_amplitudes

    def compatibility_scores(
        self, hidden: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Return raw compatibility scores (pre-softmax) for interpretability.
        Higher score → branch is more compatible with current observation.
        """
        c_proj = self.W_ent(context).unsqueeze(1)        # (batch, 1, d)
        return (hidden * c_proj).sum(dim=-1) / math.sqrt(self.d)  # (batch, K)
