"""
QRR Model — Full architecture wrapping a transformer backbone.

This module integrates all QRR components:
  BranchBank → AmplitudeRouter → UnitaryMixer → EntanglementModule
  → ObserverModule → DecoherenceModule → OutputHead

Designed to wrap any HuggingFace-compatible causal LM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .branch_bank import BranchBank, compute_chi
from .decoherence_module import DecoherenceModule


class QRRModel(nn.Module):
    """
    Quantum Relational Reasoner wrapper.

    Wraps a pretrained transformer and adds:
    - Branch Bank: K competing latent hypotheses
    - Decoherence: controlled collapse to single output
    - Collapse Index χ: residual ambiguity measure

    Args:
        backbone: A pretrained HuggingFace CausalLM model.
        d_model: Hidden dimension of backbone.
        K: Number of branches.
        use_complex_phase: Enable full complex amplitudes.
        tau_chi: Collapse threshold for χ.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        K: int = 8,
        use_complex_phase: bool = False,
        tau_chi: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.K = K
        self.d_model = d_model

        self.branch_bank = BranchBank(
            d_model=d_model,
            K=K,
            use_complex_phase=use_complex_phase,
        )
        self.decoherence = DecoherenceModule(tau_chi=tau_chi)

        # Branch-to-output projection (maps selected branch back to vocab space)
        self.branch_output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_required: bool = False,
        return_branch_map: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Returns dict with:
          - logits: (B, L, vocab_size)
          - chi: (B,) collapse index
          - k_star: (B,) selected branch
          - branch_probs: (B, K) branch probabilities
          - branch_map (optional): top-3 branch descriptions
        """
        # 1. Run transformer backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, d_model)

        # 2. Branch Bank: create K hypotheses
        H, alpha = self.branch_bank(hidden_states)  # (B,K,d), (B,K)

        # 3. Compute collapse index χ
        chi = compute_chi(alpha)  # (B,)

        # 4. Decoherence: select/mix branches
        h_selected, k_star, branch_probs = self.decoherence(
            H=H,
            alpha=alpha,
            chi=chi,
            action_required=action_required,
        )

        # 5. Project selected branch back and combine with backbone output
        h_branch = self.branch_output_proj(h_selected)  # (B, d_model)

        # Residual addition to last hidden state (mean pooled)
        h_combined = hidden_states + h_branch.unsqueeze(1)  # (B, L, d_model)

        # 6. Compute logits using backbone's LM head
        logits = self.backbone.lm_head(h_combined)  # (B, L, vocab_size)

        result = {
            "logits": logits,
            "chi": chi,
            "k_star": k_star,
            "branch_probs": branch_probs,
        }

        if return_branch_map:
            result["branch_map"] = self._get_branch_map(branch_probs, k_star)

        return result

    def _get_branch_map(
        self,
        branch_probs: torch.Tensor,
        k_star: torch.Tensor,
    ) -> list:
        """Return top-3 branches with their probabilities."""
        top3_probs, top3_idx = branch_probs.topk(min(3, self.K), dim=-1)
        maps = []
        for b in range(branch_probs.size(0)):
            entry = {
                "selected": k_star[b].item(),
                "top_branches": [
                    {"branch": top3_idx[b, i].item(), "prob": top3_probs[b, i].item()}
                    for i in range(top3_idx.size(1))
                ],
            }
            maps.append(entry)
        return maps
