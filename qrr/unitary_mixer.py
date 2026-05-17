"""
Unitary Mixer — coherent evolution operator for QRR branch bank.

Mixes K branch hidden states using an approximately unitary transformation,
preserving total amplitude norm across steps (analogous to unitary evolution
in quantum mechanics).

Two parametrizations:
  'cayley'  : U = (I - A)(I + A)^{-1}, A skew-symmetric → exactly orthogonal
  'linear'  : unconstrained linear mix (faster, less stable — for ablations)

Default is 'cayley' for training stability.
See EXP-002 for empirical comparison of norm drift.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitaryMixer(nn.Module):
    """
    Mixes K branch hidden states coherently.

    For each position in the branch bank, computes a context-conditioned
    mixing matrix M(x_t, c_t) ∈ R^{K x K} and applies it to the branch
    hidden states and amplitudes.

    Args:
        hidden_dim:    Dimensionality of branch hidden states.
        k_branches:    Number of branches K.
        context_dim:   Dimensionality of conditioning context (from base transformer).
                       If None, uses hidden_dim.
        parametrization: 'cayley' (orthogonal, default) or 'linear' (ablation).
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int,
        context_dim: int | None = None,
        parametrization: str = "cayley",
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.context_dim = context_dim or hidden_dim
        self.parametrization = parametrization

        if parametrization not in ("cayley", "linear"):
            raise ValueError(f"Unknown parametrization '{parametrization}'.")

        # Context → K*K mixing weights
        # For Cayley: output is upper-triangular A (skew-sym), then U = (I-A)(I+A)^-1
        # For linear: output is raw K*K matrix
        self.mixer_net = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim // 2),
            nn.SiLU(),
            nn.Linear(self.context_dim // 2, k_branches * k_branches),
        )

        # Branch hidden state update: context-gated residual
        self.branch_update = nn.Linear(hidden_dim + self.context_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _build_mixing_matrix(self, context: torch.Tensor) -> torch.Tensor:
        """
        Build K×K mixing matrix from context vector.

        Args:
            context: (batch, context_dim)
        Returns:
            M: (batch, K, K) — mixing matrix
        """
        batch = context.size(0)
        raw = self.mixer_net(context).view(batch, self.K, self.K)  # (batch, K, K)

        if self.parametrization == "cayley":
            # Build skew-symmetric A = raw - raw^T
            A = raw - raw.transpose(1, 2)  # (batch, K, K), skew-symmetric
            # Cayley transform: U = (I - A)(I + A)^{-1}
            I = torch.eye(self.K, device=context.device).unsqueeze(0).expand(batch, -1, -1)
            try:
                M = torch.linalg.solve(I + A, I - A)  # (batch, K, K)
            except RuntimeError:
                # Fallback to softmax if matrix is singular
                M = torch.softmax(raw, dim=-1)
        else:
            # Linear ablation: row-normalize with softmax
            M = torch.softmax(raw, dim=-1)

        return M

    def forward(
        self,
        hidden: torch.Tensor,      # (batch, K, d)
        amplitudes: torch.Tensor,  # (batch, K, 2) — [real, imag]
        context: torch.Tensor,     # (batch, context_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coherent mixing to branches.

        Returns:
            new_hidden:     (batch, K, d)
            new_amplitudes: (batch, K, 2)
        """
        batch = hidden.size(0)
        M = self._build_mixing_matrix(context)  # (batch, K, K)

        # Mix amplitudes: new_α^(k) = sum_j M_{kj} * α^(j)
        amp_real = amplitudes[..., 0]  # (batch, K)
        amp_imag = amplitudes[..., 1]  # (batch, K)
        new_amp_real = torch.bmm(M, amp_real.unsqueeze(-1)).squeeze(-1)  # (batch, K)
        new_amp_imag = torch.bmm(M, amp_imag.unsqueeze(-1)).squeeze(-1)  # (batch, K)
        new_amplitudes = torch.stack([new_amp_real, new_amp_imag], dim=-1)  # (batch, K, 2)

        # Mix hidden states: new_h^(k) = sum_j M_{kj} * h^(j)  [amplitude-weighted mean]
        # Shape: (batch, K, d) = (batch, K, K) x (batch, K, d)
        new_hidden_mixed = torch.bmm(M, hidden)  # (batch, K, d)

        # Context-gated residual update per branch
        ctx_expanded = context.unsqueeze(1).expand(-1, self.K, -1)  # (batch, K, context_dim)
        gate_input = torch.cat([new_hidden_mixed, ctx_expanded], dim=-1)  # (batch, K, d+ctx)
        delta = self.branch_update(gate_input)  # (batch, K, d)
        new_hidden = self.layer_norm(new_hidden_mixed + delta)

        return new_hidden, new_amplitudes

    def norm_drift(self, amp_initial: torch.Tensor, amp_final: torch.Tensor) -> torch.Tensor:
        """
        Compute amplitude norm drift for EXP-002.
        Returns ||alpha_final|| / ||alpha_initial|| - 1 (0 = perfect preservation).
        """
        norm_i = (amp_initial[..., 0] ** 2 + amp_initial[..., 1] ** 2).sum(-1).sqrt()
        norm_f = (amp_final[..., 0] ** 2 + amp_final[..., 1] ** 2).sum(-1).sqrt()
        return ((norm_f / (norm_i + 1e-9)) - 1.0).abs().mean()
