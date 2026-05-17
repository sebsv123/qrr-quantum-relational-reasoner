"""
Amplitude Router — context-conditioned complex amplitude update.

After UnitaryMixer evolves the branch bank, the AmplitudeRouter updates
the complex amplitudes based on new evidence from the current token
and accumulated context.

This is the module responsible for constructive and destructive interference:
  - Branches consistent with new evidence get amplified
  - Branches inconsistent with new evidence get suppressed

The router operates in the complex domain:
  α^(k)_{t+1} = router(h^(k)_t, x_t, c_t) * α^(k)_t
where router output is a complex gain in polar form (magnitude ∈ [0,1], phase ∈ [-π, π]).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math


class AmplitudeRouter(nn.Module):
    """
    Updates complex amplitudes based on branch-context compatibility.

    For each branch k, computes a complex gain g^(k) = r^(k) * e^{iθ^(k)}:
      r^(k) ∈ [0,1]  — magnitude: how much this branch is supported by evidence
      θ^(k) ∈ [-π,π] — phase:     interference angle with other branches

    Args:
        hidden_dim:   Dimensionality of branch hidden states.
        k_branches:   Number of branches K.
        context_dim:  Dimensionality of conditioning context.
        use_phase:    If True, compute full complex gain with phase (Stage 2).
                      If False, use real-valued gain only (Stage 1 curriculum).
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int,
        context_dim: int | None = None,
        use_phase: bool = False,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.context_dim = context_dim or hidden_dim
        self.use_phase = use_phase

        # Compatibility scorer: how well does branch k explain current evidence?
        # Input: [h^(k), context] → scalar compatibility score
        self.compatibility = nn.Sequential(
            nn.Linear(hidden_dim + self.context_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if use_phase:
            # Phase predictor: context → phase per branch
            self.phase_net = nn.Sequential(
                nn.Linear(hidden_dim + self.context_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1),
            )

    def forward(
        self,
        hidden: torch.Tensor,      # (batch, K, d)
        amplitudes: torch.Tensor,  # (batch, K, 2) — [real, imag]
        context: torch.Tensor,     # (batch, context_dim)
    ) -> torch.Tensor:
        """
        Compute updated complex amplitudes.

        Args:
            hidden:     Current branch hidden states (batch, K, d)
            amplitudes: Current complex amplitudes (batch, K, 2)
            context:    Current token context from base transformer (batch, context_dim)

        Returns:
            new_amplitudes: (batch, K, 2) — updated [real, imag]
        """
        batch = hidden.size(0)

        # Expand context to match K branches
        ctx = context.unsqueeze(1).expand(-1, self.K, -1)  # (batch, K, context_dim)

        # Compute compatibility score per branch
        gate_input = torch.cat([hidden, ctx], dim=-1)         # (batch, K, d+ctx)
        compat = self.compatibility(gate_input).squeeze(-1)   # (batch, K)
        magnitude = torch.sigmoid(compat)                     # r^(k) ∈ (0, 1)

        if self.use_phase:
            # Stage 2: full complex gain with phase
            raw_phase = self.phase_net(gate_input).squeeze(-1)  # (batch, K)
            phase = raw_phase * math.pi                          # θ^(k) ∈ (-π, π)

            # Complex multiplication: (α_r + iα_i) * r*(cosθ + i*sinθ)
            amp_r = amplitudes[..., 0]   # (batch, K)
            amp_i = amplitudes[..., 1]   # (batch, K)
            gain_r = magnitude * torch.cos(phase)
            gain_i = magnitude * torch.sin(phase)

            new_amp_r = amp_r * gain_r - amp_i * gain_i
            new_amp_i = amp_r * gain_i + amp_i * gain_r
        else:
            # Stage 1: real-valued gain only (simpler, more stable to train)
            new_amp_r = amplitudes[..., 0] * magnitude
            new_amp_i = amplitudes[..., 1] * magnitude

        new_amplitudes = torch.stack([new_amp_r, new_amp_i], dim=-1)  # (batch, K, 2)

        # Renormalize to prevent amplitude collapse
        norm_sq = (new_amp_r ** 2 + new_amp_i ** 2).sum(dim=-1, keepdim=True) + 1e-9
        scale = (1.0 / norm_sq.sqrt()).unsqueeze(-1)  # broadcast over K and 2
        # Only scale amplitude magnitudes, not hidden states
        norms = (new_amp_r ** 2 + new_amp_i ** 2 + 1e-9).sqrt()  # (batch, K)
        total_norm = norms.sum(dim=-1, keepdim=True)               # (batch, 1)
        new_amp_r = new_amp_r / total_norm
        new_amp_i = new_amp_i / total_norm
        new_amplitudes = torch.stack([new_amp_r, new_amp_i], dim=-1)

        return new_amplitudes

    def interference_pattern(
        self,
        amplitudes_before: torch.Tensor,
        amplitudes_after: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Diagnostic: measure how much constructive vs destructive interference occurred.

        Returns dict with:
          'constructive': fraction of branches that gained amplitude
          'destructive':  fraction of branches that lost amplitude
          'net_change':   mean amplitude magnitude change
        """
        mag_before = (amplitudes_before[..., 0] ** 2 + amplitudes_before[..., 1] ** 2).sqrt()
        mag_after  = (amplitudes_after[..., 0]  ** 2 + amplitudes_after[..., 1]  ** 2).sqrt()
        delta = mag_after - mag_before
        constructive = (delta > 0).float().mean()
        destructive  = (delta < 0).float().mean()
        return {
            "constructive": constructive,
            "destructive":  destructive,
            "net_change":   delta.mean(),
        }
