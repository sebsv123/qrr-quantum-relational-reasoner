"""
Decoherence Module — Controlled collapse of branch superposition.

Reduces the full branch density matrix to a classical mixture
and samples a single branch for output generation.

Collapse is triggered only when:
  (a) An action requires commitment (action_required=True)
  (b) Residual ambiguity χ < τ_χ (evidence is clear)
  (c) Mutual information I(R; O) > τ_I
  (d) Computational budget is exceeded
"""

import torch
import torch.nn as nn
from typing import Tuple


class DecoherenceModule(nn.Module):
    """
    Args:
        tau_chi: χ threshold below which collapse is triggered (default 0.3).
        tau_I: Mutual information threshold for collapse (default 0.7).
        training_collapse_prob: During training, probability of
                                forcing a collapse for supervision.
    """

    def __init__(
        self,
        tau_chi: float = 0.3,
        tau_I: float = 0.7,
        training_collapse_prob: float = 0.5,
    ):
        super().__init__()
        self.tau_chi = tau_chi
        self.tau_I = tau_I
        self.training_collapse_prob = training_collapse_prob

    def forward(
        self,
        H: torch.Tensor,         # (B, K, d_model)
        alpha: torch.Tensor,     # (B, K) complex or real
        chi: torch.Tensor,       # (B,) collapse index
        mutual_info: torch.Tensor = None,  # (B,) optional I(R;O)
        action_required: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_out: (B, d_model) — output hidden state
            k_star: (B,) — selected branch index
            p: (B, K) — branch probabilities |α_k|²
        """
        probs = alpha.abs() ** 2 if torch.is_complex(alpha) else alpha ** 2
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Determine whether to collapse
        collapse_flag = self._should_collapse(
            chi, mutual_info, action_required
        )  # (B,)

        B, K, d = H.shape
        h_out = torch.zeros(B, d, device=H.device, dtype=H.dtype)
        k_star = torch.zeros(B, dtype=torch.long, device=H.device)

        for b in range(B):
            if collapse_flag[b]:
                # Sample k* from branch probability distribution
                k_s = torch.multinomial(probs[b], num_samples=1).squeeze()
                h_out[b] = H[b, k_s]
                k_star[b] = k_s
            else:
                # Weighted mixture — no collapse yet
                h_out[b] = (probs[b].unsqueeze(-1) * H[b]).sum(dim=0)
                k_star[b] = probs[b].argmax()

        return h_out, k_star, probs

    def _should_collapse(
        self,
        chi: torch.Tensor,
        mutual_info: torch.Tensor,
        action_required: bool,
    ) -> torch.Tensor:
        """Returns boolean tensor (B,) indicating collapse decision."""
        B = chi.shape[0]
        flag = torch.zeros(B, dtype=torch.bool, device=chi.device)

        if action_required:
            flag = torch.ones(B, dtype=torch.bool, device=chi.device)
            return flag

        # Low ambiguity → collapse
        flag = flag | (chi < self.tau_chi)

        # High mutual information → collapse
        if mutual_info is not None:
            flag = flag | (mutual_info > self.tau_I)

        # During training: stochastic forced collapse for supervision
        if self.training:
            random_mask = torch.rand(B, device=chi.device) < self.training_collapse_prob
            flag = flag | random_mask

        return flag
