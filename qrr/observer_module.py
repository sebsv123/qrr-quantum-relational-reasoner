"""
Observer Module — internalized selective observer.

In relational quantum mechanics (Rovelli 1996), there is no universal
observer: measurement outcomes are relative to the observing system.

In QRR, the Observer Module plays this role:
  - It monitors the χ trace across time steps
  - It decides WHEN to trigger decoherence (not just IF)
  - It can request additional context before collapsing
  - It maintains a confidence history and detects premature collapse

The observer is the module that answers: "Is now the right time to commit?"

This is distinct from the DecoherenceModule (which performs the collapse):
the Observer decides; the Decoherence executes.

Reference:
  Rovelli, C. (1996). Relational quantum mechanics.
  International Journal of Theoretical Physics, 35(8), 1637-1678.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from collections import deque


class ObserverModule(nn.Module):
    """
    Monitors χ trajectory and decides when to trigger decoherence.

    Decision policy options:
      'threshold'    : collapse when χ < threshold (simple)
      'patience'     : collapse only after χ < threshold for N consecutive steps
      'gradient'     : collapse when dχ/dt < 0 and χ < threshold (χ is decreasing)
      'learned'      : small MLP maps χ history → collapse probability

    Args:
        chi_threshold:  Base χ threshold for collapse decision.
        policy:         Decision policy (see above).
        patience:       Steps to wait before confirming collapse (for 'patience' policy).
        history_len:    χ history window for 'learned' and 'gradient' policies.
    """

    def __init__(
        self,
        chi_threshold: float = 0.3,
        policy: str = "patience",
        patience: int = 3,
        history_len: int = 10,
    ) -> None:
        super().__init__()
        self.threshold = chi_threshold
        self.policy = policy
        self.patience = patience
        self.history_len = history_len

        self._chi_history: deque = deque(maxlen=history_len)
        self._below_threshold_count: int = 0

        # Learned policy: maps χ history → P(collapse)
        if policy == "learned":
            self.collapse_predictor = nn.Sequential(
                nn.Linear(history_len, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def observe(self, chi: torch.Tensor) -> dict:
        """
        Process current χ value and return observation report.

        Args:
            chi: (batch,) collapse index at current step

        Returns:
            dict with:
              'should_collapse': bool tensor (batch,)
              'confidence':      float tensor (batch,) — 1 - χ
              'trend':           'decreasing' | 'stable' | 'increasing'
              'steps_below':     int — consecutive steps below threshold
        """
        chi_mean = chi.mean().item()
        self._chi_history.append(chi_mean)
        confidence = 1.0 - chi

        if self.policy == "threshold":
            should_collapse = chi < self.threshold
            self._below_threshold_count = 0

        elif self.policy == "patience":
            if chi_mean < self.threshold:
                self._below_threshold_count += 1
            else:
                self._below_threshold_count = 0
            should_collapse = (
                (chi < self.threshold) &
                torch.tensor(self._below_threshold_count >= self.patience)
            )

        elif self.policy == "gradient":
            if len(self._chi_history) >= 2:
                d_chi = self._chi_history[-1] - self._chi_history[-2]
                collapsing = d_chi < 0 and chi_mean < self.threshold
            else:
                collapsing = False
            should_collapse = chi < self.threshold if collapsing else torch.zeros_like(chi, dtype=torch.bool)

        elif self.policy == "learned":
            if len(self._chi_history) == self.history_len:
                hist = torch.tensor(list(self._chi_history), dtype=torch.float32)
                p_collapse = self.collapse_predictor(hist.unsqueeze(0)).squeeze()
                should_collapse = p_collapse > 0.5
                should_collapse = should_collapse.expand(chi.size(0))
            else:
                should_collapse = torch.zeros(chi.size(0), dtype=torch.bool)

        else:
            should_collapse = chi < self.threshold

        # Trend analysis
        if len(self._chi_history) >= 3:
            recent = list(self._chi_history)[-3:]
            if recent[-1] < recent[0] - 0.05:
                trend = "decreasing"   # converging toward collapse
            elif recent[-1] > recent[0] + 0.05:
                trend = "increasing"   # diverging, more ambiguous
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return {
            "should_collapse": should_collapse,
            "confidence": confidence,
            "trend": trend,
            "steps_below": self._below_threshold_count,
            "chi_history": list(self._chi_history),
        }

    def reset(self) -> None:
        """Reset observer state for new inference sequence."""
        self._chi_history.clear()
        self._below_threshold_count = 0
