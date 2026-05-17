"""
Curriculum Scheduler for QRR training.

Stage 1 (real amplitudes): Train branch bank, mixer, entanglement, and
  decoherence with real-valued amplitudes only. The model learns to
  maintain and select between branches without phase information.
  Equivalent to a structured Mixture of Experts with coherent evolution.

Stage 2 (complex phases): Activate imaginary parts of the AmplitudeRouter.
  The model now has full complex amplitudes. Branches can interfere
  constructively (aligned phases) or destructively (opposite phases).
  This is where QRR becomes strictly more expressive than MoE.

Transition criterion:
  Switch to Stage 2 when Stage 1 training shows:
  1. L_coh < 0.05 (model can maintain branches on ambiguous inputs)
  2. L_dec < 0.05 (model can collapse cleanly on clear inputs)
  3. Δχ > 0.10 on validation set (χ has some discriminative signal)
"""

from __future__ import annotations
import torch
from qrr.qrr_model import QRRModel


class CurriculumScheduler:
    """
    Manages the Stage 1 → Stage 2 transition.

    Args:
        model:          QRRModel instance.
        transition_step: Force transition at this step regardless of criteria.
        l_coh_threshold: Max L_coh to allow transition (default 0.05).
        l_dec_threshold: Max L_dec to allow transition (default 0.05).
        delta_chi_min:   Min Δχ on validation to allow transition (default 0.10).
    """

    def __init__(
        self,
        model: QRRModel,
        transition_step: int = 5000,
        l_coh_threshold: float = 0.05,
        l_dec_threshold: float = 0.05,
        delta_chi_min: float = 0.10,
    ) -> None:
        self.model = model
        self.transition_step = transition_step
        self.l_coh_thr = l_coh_threshold
        self.l_dec_thr = l_dec_threshold
        self.delta_chi_min = delta_chi_min
        self.stage = 1
        self._step = 0
        self._transition_log: list[dict] = []

    def step(
        self,
        l_coh: float,
        l_dec: float,
        delta_chi: float,
    ) -> bool:
        """
        Call each training step. Returns True if transition just occurred.

        Args:
            l_coh:     Current coherence loss.
            l_dec:     Current decoherence loss.
            delta_chi: Current Δχ on validation set.
        """
        self._step += 1
        if self.stage == 2:
            return False

        criteria_met = (
            l_coh < self.l_coh_thr
            and l_dec < self.l_dec_thr
            and delta_chi > self.delta_chi_min
        )
        forced = self._step >= self.transition_step

        if criteria_met or forced:
            self._activate_stage2(forced=forced, l_coh=l_coh, l_dec=l_dec, delta_chi=delta_chi)
            return True

        return False

    def _activate_stage2(self, forced: bool, **metrics) -> None:
        """Activate complex phases in the AmplitudeRouter."""
        self.model.router.activate_complex_phases()
        self.stage = 2
        record = {"step": self._step, "forced": forced, **metrics}
        self._transition_log.append(record)
        reason = "forced" if forced else "criteria met"
        print(f"\n[Curriculum] Stage 1 → Stage 2 at step {self._step} ({reason})")
        print(f"  l_coh={metrics.get('l_coh', 0):.4f}  l_dec={metrics.get('l_dec', 0):.4f}  Δχ={metrics.get('delta_chi', 0):.4f}")
        print("  Complex phases activated. Interference is now live.\n")

    @property
    def in_stage1(self) -> bool:
        return self.stage == 1

    @property
    def in_stage2(self) -> bool:
        return self.stage == 2
