"""
Observer Module — internalized selective observer.

In quantum mechanics, measurement (observation) forces a system
out of superposition into a definite state. In QRR, the Observer
models this: it decides *when* to measure (collapse) and *what*
to do with the collapsed state.

The Observer is NOT always active. It monitors χ over time and
triggers decoherence only when the collapse criterion is met:
  - χ < χ_threshold   (model is confident enough)
  - OR a forced_action signal is received (external deadline)
  - OR max_steps steps have elapsed without collapse

Upon triggering, the Observer:
  1. Requests decoherence (via DecoherenceModule)
  2. Produces a summary vector for the output head
  3. Records the step at which collapse occurred (interpretability)

This models the relational interpretation of QM (Rovelli 1996):
observation is a relation between systems, not an absolute event.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class ObservationRecord:
    """Tracks when and why the Observer triggered collapse."""
    step: int
    chi_at_collapse: float
    reason: str  # 'threshold' | 'forced' | 'timeout'
    branch_weights: list[float] = field(default_factory=list)


class ObserverModule(nn.Module):
    """
    Internalized selective observer that gates decoherence.

    Args:
        hidden_dim:      Branch hidden state dim.
        chi_threshold:   Collapse when χ falls below this (default 0.3).
        max_steps:       Force collapse after this many steps (default 20).
        patience:        Steps χ must stay below threshold before collapsing.
                         Prevents premature collapse on noisy χ (default 1).
    """

    def __init__(
        self,
        hidden_dim: int,
        chi_threshold: float = 0.3,
        max_steps: int = 20,
        patience: int = 1,
    ) -> None:
        super().__init__()
        self.chi_threshold = chi_threshold
        self.max_steps = max_steps
        self.patience = patience

        # Learnable output projection after collapse
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self._consecutive_below: int = 0
        self._step: int = 0
        self.history: list[ObservationRecord] = []

    def should_observe(
        self,
        chi: torch.Tensor,      # (batch,)
        forced: bool = False,
    ) -> torch.Tensor:
        """
        Returns bool mask: True where Observer should trigger collapse.
        Uses patience to avoid collapsing on transient χ dips.
        """
        self._step += 1
        below = (chi < self.chi_threshold)

        if forced or self._step >= self.max_steps:
            return torch.ones_like(below)

        if below.all():
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        return below & (self._consecutive_below >= self.patience)

    def observe(
        self,
        collapsed_h: torch.Tensor,   # (batch, d) — from DecoherenceModule
        chi: torch.Tensor,
        forced: bool = False,
    ) -> torch.Tensor:
        """
        Post-collapse processing: project and normalize the collapsed hidden state.
        Records observation event.

        Returns: output hidden state (batch, d)
        """
        reason = "forced" if forced else ("timeout" if self._step >= self.max_steps else "threshold")
        record = ObservationRecord(
            step=self._step,
            chi_at_collapse=chi.mean().item(),
            reason=reason,
        )
        self.history.append(record)
        return self.norm(self.output_proj(collapsed_h))

    def reset(self) -> None:
        """Reset for new sequence."""
        self._consecutive_below = 0
        self._step = 0
        self.history.clear()
