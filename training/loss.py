"""
QRR Multi-Objective Loss Function.

L = L_token + λ1*L_coh + λ2*L_dec + λ3*L_cal + λ4*L_ent

Each term targets a specific aspect of QRR behavior.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def qrr_loss(
    logits: torch.Tensor,           # (B, L, vocab_size)
    labels: torch.Tensor,           # (B, L)
    branch_probs: torch.Tensor,     # (B, K)
    chi: torch.Tensor,              # (B,)
    evidence_strength: Optional[torch.Tensor] = None,  # (B,) ∈ [0,1]
    lambda_coh: float = 0.1,
    lambda_dec: float = 0.1,
    lambda_cal: float = 0.05,
    lambda_div: float = 0.05,
) -> dict:
    """
    Compute QRR multi-objective loss.

    Args:
        logits: Model output logits.
        labels: Ground truth token ids.
        branch_probs: |α_k|² per branch.
        chi: Collapse index.
        evidence_strength: If available, how strong is the evidence
                           for collapsing (0=ambiguous, 1=clear).

    Returns:
        dict with 'total', 'token', 'coherence', 'decoherence',
                  'calibration', 'diversity' losses.
    """

    # 1. Token-level cross-entropy (standard LM objective)
    B, L, V = logits.shape
    L_token = F.cross_entropy(
        logits.view(B * L, V),
        labels.view(B * L),
        ignore_index=-100,
    )

    # 2. Coherence loss: penalize premature collapse when evidence is weak
    # High chi (superposition) is good when evidence_strength is low
    if evidence_strength is not None:
        # We want χ to be high when evidence is weak, low when evidence is strong
        target_chi = 1.0 - evidence_strength  # (B,)
        L_coh = F.mse_loss(chi, target_chi)
    else:
        L_coh = torch.tensor(0.0, device=logits.device)

    # 3. Decoherence loss: penalize delayed collapse when evidence is strong
    # Similar to coherence but in opposite direction
    if evidence_strength is not None:
        collapse_needed = (evidence_strength > 0.7).float()  # (B,)
        L_dec = (collapse_needed * chi).mean()  # penalize high χ when evidence clear
    else:
        L_dec = torch.tensor(0.0, device=logits.device)

    # 4. Calibration loss: align branch_probs with actual error distribution
    # Approximate: high-confidence (low-entropy) branches should have lower error
    branch_entropy = -(branch_probs * (branch_probs + 1e-8).log()).sum(dim=-1)  # (B,)
    # Normalize to [0,1]
    max_entropy = torch.log(torch.tensor(branch_probs.size(-1), dtype=torch.float))
    branch_entropy_norm = branch_entropy / (max_entropy + 1e-8)
    # Calibration target: entropy should correlate with token loss magnitude
    # (this is approximate — proper calibration needs held-out data)
    L_cal = F.mse_loss(branch_entropy_norm, chi.detach())

    # 5. Diversity loss: prevent all branches from collapsing to same representation
    # Maximize entropy of branch distribution
    L_div = -branch_entropy.mean()  # maximize entropy = minimize negative entropy

    # Total loss
    total = (
        L_token
        + lambda_coh * L_coh
        + lambda_dec * L_dec
        + lambda_cal * L_cal
        + lambda_div * L_div
    )

    return {
        "total": total,
        "token": L_token,
        "coherence": L_coh,
        "decoherence": L_dec,
        "calibration": L_cal,
        "diversity": L_div,
    }
