"""
Unitary Mixer — coherent evolution of branch hidden states.

In quantum mechanics, time evolution of a closed system is unitary:
  |ψ(t+1)⟩ = U |ψ(t)⟩,  where U†U = I

Here, U_kj(x_t, c_t) mixes branch hidden states while approximately
preserving the total amplitude norm — analogous to unitary evolution.

Two parametrizations:
  'cayley'  : U = (I - A)(I + A)^{-1}, A skew-symmetric → exact orthogonal matrix.
              Slower but norm-preserving by construction.
  'learned' : Unconstrained linear mixing with soft orthogonality regularizer.
              Faster, differentiable, norm approximately preserved via L_coh loss.

The context vector c_t (from the input) modulates U, making evolution
input-dependent — branches evolve differently based on what is observed.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitaryMixer(nn.Module):
    """
    Context-conditioned approximate unitary mixer for branch hidden states.

    h̃^(k)_{t+1} = Σ_j U_kj(x_t, c_t) · h^(j)_t

    Args:
        hidden_dim:   Dimensionality of branch hidden states.
        k_branches:   Number of branches K.
        context_dim:  Dimensionality of context vector c_t.
        parametrize:  'cayley' (exact orthogonal) or 'learned' (soft).
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        context_dim: int | None = None,
        parametrize: str = "learned",
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches
        self.parametrize = parametrize
        context_dim = context_dim or hidden_dim

        # Context → mixing matrix (K×K weights)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, k_branches * k_branches),
            nn.Tanh(),
        )

        if parametrize == "cayley":
            # Skew-symmetric generator A: K×K, A^T = -A
            # U = (I - A)(I + A)^{-1}  is orthogonal
            self.A_base = nn.Parameter(torch.zeros(k_branches, k_branches))

        # Per-branch hidden-state projection (keeps branch in same subspace)
        self.branch_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _mixing_matrix(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute K×K mixing matrix from context vector.
        Returns: (batch, K, K)
        """
        batch = context.size(0)
        raw = self.context_proj(context).view(batch, self.K, self.K)

        if self.parametrize == "cayley":
            # Skew-symmetric A from base + context modulation
            A = self.A_base.unsqueeze(0) + (raw - raw.transpose(1, 2)) * 0.1
            A = A - A.transpose(1, 2)  # enforce skew-symmetry
            I = torch.eye(self.K, device=context.device).unsqueeze(0)
            # Cayley transform: U = (I - A)(I + A)^{-1}
            U = torch.linalg.solve(I + A, I - A)
        else:
            # Soft orthogonal: normalize columns
            U = F.normalize(raw, dim=1)

        return U  # (batch, K, K)

    def forward(
        self,
        hidden: torch.Tensor,    # (batch, K, d)
        context: torch.Tensor,   # (batch, context_dim)
    ) -> torch.Tensor:
        """
        Mix branch hidden states via context-conditioned U.

        Returns: mixed hidden states, shape (batch, K, d)
        """
        U = self._mixing_matrix(context)  # (batch, K, K)
        # h̃^(k) = Σ_j U_kj h^(j)  →  batched matmul
        # hidden: (batch, K, d), U: (batch, K, K)
        mixed = torch.bmm(U, hidden)  # (batch, K, d)
        # Per-branch projection keeps each branch in its learned subspace
        mixed = self.branch_proj(mixed)
        return mixed

    def orthogonality_loss(self, context: torch.Tensor) -> torch.Tensor:
        """
        Regularizer: penalize deviation from orthogonality.
        ||U^T U - I||_F²  should be 0 for a true unitary.
        Only needed for 'learned' parametrize; Cayley is exact.
        """
        U = self._mixing_matrix(context)
        I = torch.eye(self.K, device=context.device).unsqueeze(0)
        deviation = torch.bmm(U.transpose(1, 2), U) - I
        return deviation.pow(2).mean()
