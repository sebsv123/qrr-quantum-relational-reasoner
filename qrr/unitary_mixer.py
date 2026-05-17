"""
Unitary Mixer — coherent evolution of branch hidden states.

Inspired by quantum unitary evolution: U|ψ⟩.
Mixes K branch hidden states while approximately preserving
the total amplitude norm (analogous to unitary transformation).

Two parametrizations:
  'cayley'  : U = (I - A)(I + A)^{-1}, A skew-symmetric → exact unitary
  'learned' : free linear mixer with orthogonality regularization loss

The Cayley parametrization guarantees ‖α‖ is preserved exactly.
The learned variant is faster but requires λ_orth penalty in training.

Reference: Cayley transform for unitary matrices:
  Helfrich et al. (2018) "Orthogonal Recurrent Neural Networks"
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from qrr.branch_bank import BranchState


class UnitaryMixer(nn.Module):
    """
    Mixes K branch hidden states via an approximately unitary transformation.
    Conditioned on the current token embedding and context vector.

    Args:
        hidden_dim:  Dimensionality d of each branch hidden state.
        k_branches:  Number of branches K.
        context_dim: Dimensionality of the context conditioning vector (default = hidden_dim).
        parametrization: 'cayley' (exact unitary) or 'learned' (free + regularized).
        num_heads:   Number of attention heads for context conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        context_dim: int | None = None,
        parametrization: str = "cayley",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.parametrization = parametrization
        context_dim = context_dim or hidden_dim

        # Context → mixing matrix generator
        # Outputs K×K mixing coefficients conditioned on current input
        self.mixer_gen = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, k_branches * k_branches),
        )

        # Per-branch transformation conditioned on context
        self.branch_transform = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Residual gate: how much to mix vs. keep original
        self.gate = nn.Sequential(
            nn.Linear(context_dim, k_branches),
            nn.Sigmoid(),
        )

    def _cayley_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Construct a unitary matrix via Cayley transform.
        A = raw - raw^T (skew-symmetric)
        U = (I - A)(I + A)^{-1}

        Args:
            raw: (batch, K, K) raw mixing coefficients
        Returns:
            U: (batch, K, K) unitary mixing matrix
        """
        # Make skew-symmetric: A = (raw - raw^T) / 2
        A = (raw - raw.transpose(-1, -2)) / 2.0
        I = torch.eye(self.K, device=raw.device).unsqueeze(0)
        # Cayley: U = (I - A)(I + A)^{-1}
        U = torch.linalg.solve(I + A, I - A)
        return U

    def forward(
        self,
        state: BranchState,
        context: torch.Tensor,  # (batch, d) — pooled context from base transformer
    ) -> BranchState:
        """
        Apply coherent evolution to all branches.

        Args:
            state:   Current BranchState
            context: Context vector from base transformer (batch, d)

        Returns:
            Updated BranchState with mixed hidden states and updated amplitudes
        """
        batch = context.size(0)
        h = state.hidden       # (batch, K, d)
        alpha = state.amplitudes  # (batch, K, 2)

        # 1. Generate K×K mixing matrix conditioned on context
        raw_mix = self.mixer_gen(context).view(batch, self.K, self.K)  # (batch, K, K)

        if self.parametrization == "cayley":
            U = self._cayley_matrix(raw_mix)  # (batch, K, K) unitary
        else:
            # Learned: normalize rows for approximate orthogonality
            U = F.normalize(raw_mix, dim=-1)

        # 2. Mix hidden states: h̃^(k) = Σ_j U_{kj} h^(j)
        h_mixed = torch.bmm(U, h)  # (batch, K, d)

        # 3. Apply per-branch self-attention (branches attend to each other)
        h_attended, _ = self.branch_transform(h_mixed, h_mixed, h_mixed)

        # 4. Residual gate: blend original and transformed
        gate = self.gate(context).unsqueeze(-1)  # (batch, K, 1)
        h_new = gate * h_attended + (1 - gate) * h  # (batch, K, d)

        # 5. Mix amplitudes with same U (amplitude evolution mirrors hidden evolution)
        # α_real and α_imag mixed separately: ã^(k) = Σ_j U_{kj} α^(j)
        alpha_real = alpha[..., 0]  # (batch, K)
        alpha_imag = alpha[..., 1]  # (batch, K)
        alpha_real_new = torch.bmm(U, alpha_real.unsqueeze(-1)).squeeze(-1)
        alpha_imag_new = torch.bmm(U, alpha_imag.unsqueeze(-1)).squeeze(-1)
        alpha_new = torch.stack([alpha_real_new, alpha_imag_new], dim=-1)  # (batch, K, 2)

        # 6. Recompute probabilities and χ
        norm_sq = alpha_real_new ** 2 + alpha_imag_new ** 2
        probs_new = norm_sq / (norm_sq.sum(dim=-1, keepdim=True) + 1e-9)
        chi_new = 1.0 - probs_new.max(dim=-1).values

        from qrr.branch_bank import BranchState
        return BranchState(
            hidden=h_new,
            amplitudes=alpha_new,
            probabilities=probs_new,
            chi=chi_new,
        )

    def orthogonality_loss(self, context: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss for 'learned' parametrization.
        Penalizes deviation from orthogonality: ‖U^T U - I‖_F^2
        """
        batch = context.size(0)
        raw = self.mixer_gen(context).view(batch, self.K, self.K)
        I = torch.eye(self.K, device=context.device).unsqueeze(0)
        orth_err = torch.bmm(raw.transpose(-1, -2), raw) - I
        return orth_err.pow(2).mean()
