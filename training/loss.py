"""
QRR Multi-Objective Loss.

L = L_token + λ_coh * L_coh + λ_dec * L_dec + λ_cal * L_cal + λ_div * L_div

Terms:
  L_token : Standard cross-entropy on next-token prediction.
  L_coh   : Coherence — penalizes branches that collapse too early (χ should stay
             high on genuinely ambiguous inputs). Measured as -χ on ambiguous inputs.
  L_dec   : Decoherence — penalizes branches that fail to collapse when the answer
             is clear (χ should drop on unambiguous inputs). Measured as χ on clear inputs.
  L_cal   : Calibration — ECE between branch confidence (1-χ) and actual accuracy.
  L_div   : Diversity — penalizes branches with near-identical hidden states
             (from DecoherenceModule.interference_loss).

Default λ values are warm-up suggestions; tune per task via grid search.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from qrr.branch_bank import BranchState


class QRRLoss(nn.Module):
    """
    Multi-objective QRR training loss.

    Args:
        lambda_coh: Weight for coherence term (default 0.1).
        lambda_dec: Weight for decoherence term (default 0.1).
        lambda_cal: Weight for calibration term (default 0.05).
        lambda_div: Weight for diversity term (default 0.1).
    """

    def __init__(
        self,
        lambda_coh: float = 0.1,
        lambda_dec: float = 0.1,
        lambda_cal: float = 0.05,
        lambda_div: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_coh = lambda_coh
        self.lambda_dec = lambda_dec
        self.lambda_cal = lambda_cal
        self.lambda_div = lambda_div
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,          # (batch, vocab)
        targets: torch.Tensor,          # (batch,)
        state: BranchState,
        is_ambiguous: torch.Tensor,     # (batch,) bool — supervision signal
        decoherence_module=None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss terms.

        Returns:
            dict with keys: 'total', 'token', 'coh', 'dec', 'cal', 'div'
        """
        # L_token: standard language modeling loss
        l_token = self.ce(logits, targets)

        chi = state.chi  # (batch,)

        # L_coh: on ambiguous inputs, χ should be high → penalize low χ
        if is_ambiguous.any():
            l_coh = -chi[is_ambiguous].mean()
        else:
            l_coh = torch.tensor(0.0, device=logits.device)

        # L_dec: on clear inputs, χ should be low → penalize high χ
        if (~is_ambiguous).any():
            l_dec = chi[~is_ambiguous].mean()
        else:
            l_dec = torch.tensor(0.0, device=logits.device)

        # L_cal: calibration — |confidence - accuracy| averaged over batch
        # confidence = 1 - χ, accuracy = (argmax(logits) == targets).float()
        confidence = 1.0 - chi
        accuracy = (logits.argmax(-1) == targets).float()
        l_cal = (confidence - accuracy).abs().mean()

        # L_div: branch diversity (via interference loss)
        if decoherence_module is not None:
            l_div = decoherence_module.interference_loss(state)
        else:
            l_div = torch.tensor(0.0, device=logits.device)

        total = (
            l_token
            + self.lambda_coh * l_coh
            + self.lambda_dec * l_dec
            + self.lambda_cal * l_cal
            + self.lambda_div * l_div
        )

        return {
            "total": total,
            "token": l_token,
            "coh": l_coh,
            "dec": l_dec,
            "cal": l_cal,
            "div": l_div,
        }
