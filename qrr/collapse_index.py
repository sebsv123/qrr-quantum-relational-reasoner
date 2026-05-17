"""
Collapse Index (χ) — QRR's core uncertainty scalar.

χ ∈ [0, 1] measures residual ambiguity across K branches:
  χ = 1 - max_k p^(k)

Interpretation:
  χ → 0 : one branch dominates → model is confident → safe to act
  χ → 1 : branches are equally weighted → high ambiguity → defer
  χ = (K-1)/K : maximum ambiguity (uniform distribution)

The DecoherenceModule uses χ to decide when to collapse branches.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CollapseIndex(nn.Module):
    """
    Computes and tracks the collapse index χ over an inference sequence.

    Args:
        threshold: χ below this value triggers decoherence (default 0.3).
                   Tune this per task: lower = act sooner, higher = more patience.
        smoothing:  EMA smoothing factor for χ trace (0 = no smoothing).
    """

    def __init__(self, threshold: float = 0.3, smoothing: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.smoothing = smoothing
        self._trace: list[torch.Tensor] = []
        self._smoothed: torch.Tensor | None = None

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute χ from branch probability distribution.

        Args:
            probabilities: shape (batch, K) — normalized branch probabilities

        Returns:
            chi: shape (batch,) — collapse index ∈ [0, 1]
        """
        chi = 1.0 - probabilities.max(dim=-1).values

        if self.smoothing > 0.0 and self._smoothed is not None:
            chi = self.smoothing * self._smoothed + (1 - self.smoothing) * chi
        self._smoothed = chi.detach()
        self._trace.append(chi.detach())

        return chi

    def should_collapse(self, chi: torch.Tensor) -> torch.Tensor:
        """Returns boolean mask: True where χ < threshold (ready to act)."""
        return chi < self.threshold

    def get_trace(self) -> list[torch.Tensor]:
        """Return full χ trace since last reset."""
        return self._trace

    def reset(self) -> None:
        """Clear trace for new inference sequence."""
        self._trace = []
        self._smoothed = None

    @property
    def max_ambiguity(self) -> float:
        """Theoretical maximum χ for K branches: (K-1)/K."""
        # Requires K — placeholder; set externally or via BranchBank
        return 1.0
