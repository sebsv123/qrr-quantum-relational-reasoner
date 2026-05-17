"""
Training Curriculum for QRR.

Stage 1 — Real Amplitudes:
  Phase scale = 0 → amplitude router applies no phase shifts.
  This trains the BranchBank and UnitaryMixer to maintain
  diverse branches using only real-valued amplitude weights.
  Analogous to classical probability mixture.

Stage 2 — Complex Phase Introduction:
  Phase scale ramps from 0 to π over N epochs.
  Amplitude router now applies genuine complex phase shifts,
  enabling constructive and destructive interference.
  This is the key step that distinguishes QRR from a simple ensemble.

Stage 3 (optional) — Dynamic Phase Scale:
  Phase scale itself becomes a learned parameter,
  conditioned on the input distribution.

Rationale:
  Training directly with complex amplitudes is unstable (loss landscape
  has saddle points where destructive interference kills gradients).
  Warm-starting with real amplitudes gives the branch bank time to
  develop diverse hidden states before phase shifts are introduced.
"""

from __future__ import annotations
import math
from qrr.qrr_model import QRRModel


class QRRCurriculum:
    """
    Manages the training curriculum for QRR.

    Args:
        model:               QRRModel instance.
        total_epochs:        Total training epochs.
        stage2_start_frac:   Fraction of epochs before Stage 2 begins (default 0.5).
        stage2_ramp_frac:    Fraction of epochs over which phase ramps from 0 to π.
    """

    def __init__(
        self,
        model: QRRModel,
        total_epochs: int,
        stage2_start_frac: float = 0.5,
        stage2_ramp_frac: float = 0.2,
    ) -> None:
        self.model = model
        self.total_epochs = total_epochs
        self.stage2_start = int(total_epochs * stage2_start_frac)
        self.stage2_ramp_end = self.stage2_start + int(total_epochs * stage2_ramp_frac)

    def step(self, epoch: int) -> dict:
        """
        Update model curriculum state for given epoch.
        Returns a dict describing the current curriculum stage.
        """
        if epoch < self.stage2_start:
            # Stage 1: real amplitudes
            self.model.amplitude_router.phase_scale = 0.0
            stage = 1
            phase_scale = 0.0
            description = "Real amplitudes only — training branch diversity"

        elif epoch < self.stage2_ramp_end:
            # Stage 2 ramp: linearly increase phase scale from 0 to π
            progress = (epoch - self.stage2_start) / max(
                self.stage2_ramp_end - self.stage2_start, 1
            )
            phase_scale = progress * math.pi
            self.model.amplitude_router.phase_scale = phase_scale
            stage = 2
            description = f"Phase ramp: {phase_scale:.3f} rad ({progress*100:.0f}% of π)"

        else:
            # Stage 2 full: complete complex amplitudes
            self.model.amplitude_router.phase_scale = math.pi
            phase_scale = math.pi
            stage = 2
            description = "Full complex amplitudes — interference active"

        return {
            "stage": stage,
            "epoch": epoch,
            "phase_scale": phase_scale,
            "description": description,
        }

    def log(self, epoch: int) -> str:
        state = self.step(epoch)
        return (
            f"[Curriculum] Epoch {epoch}/{self.total_epochs} "
            f"Stage {state['stage']} — {state['description']}"
        )
