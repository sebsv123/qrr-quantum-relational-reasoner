"""
Decoherence Module — controlled branch collapse.

Triggered when χ < threshold. Merges or selects branches
to produce a final hidden state for the output head.

Two collapse strategies:
  'weighted_sum' : final_h = sum_k p^(k) * h^(k)  (soft, differentiable)
  'argmax'       : final_h = h^(argmax_k p^(k))    (hard, not differentiable)

During training: use 'weighted_sum'.
During inference (agent actions): use 'argmax' or temperature-scaled sampling.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from qrr.branch_bank import BranchState


class DecoherenceModule(nn.Module):
    """
    Collapses the branch bank into a single hidden state.

    Args:
        strategy: 'weighted_sum' (default, differentiable) or 'argmax' (hard).
        temperature: Softmax temperature applied to probabilities before collapse.
                     temperature=1.0 uses raw probabilities.
                     temperature→0 approaches argmax.
    """

    def __init__(
        self,
        strategy: str = "weighted_sum",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if strategy not in ("weighted_sum", "argmax"):
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'weighted_sum' or 'argmax'.")
        self.strategy = strategy
        self.temperature = temperature

    def forward(self, state: BranchState) -> torch.Tensor:
        """
        Collapse branch bank to a single hidden state.

        Args:
            state: BranchState from BranchBank

        Returns:
            h_collapsed: shape (batch, d)
        """
        probs = state.probabilities  # (batch, K)
        hidden = state.hidden        # (batch, K, d)

        if self.strategy == "argmax":
            idx = probs.argmax(dim=-1, keepdim=True)        # (batch, 1)
            idx = idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1))  # (batch, 1, d)
            h_collapsed = hidden.gather(1, idx).squeeze(1)  # (batch, d)

        else:  # weighted_sum
            if self.temperature != 1.0:
                probs = torch.softmax(probs / self.temperature, dim=-1)
            # (batch, K, 1) * (batch, K, d) → (batch, d)
            h_collapsed = (probs.unsqueeze(-1) * hidden).sum(dim=1)

        return h_collapsed

    def interference_loss(self, state: BranchState) -> torch.Tensor:
        """
        Auxiliary loss that penalizes branches with near-identical hidden states.
        Encourages branch diversity (constructive interference requires distinct branches).

        Returns: scalar loss term (add to total loss with weight λ_div)
        """
        h = state.hidden  # (batch, K, d)
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-9)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (batch, K, K)
        # Penalize high off-diagonal similarity
        eye = torch.eye(h.size(1), device=h.device).unsqueeze(0)
        off_diag = sim * (1 - eye)
        return off_diag.abs().mean()
