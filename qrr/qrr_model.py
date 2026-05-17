"""
QRR Model — full pipeline wrapping a HuggingFace causal LM base.

Architecture flow per token:
  1. Base transformer encodes token → context vector c_t ∈ R^d
  2. BranchBank initializes or holds K branch states {h^(k), α^(k)}
  3. EntanglementModule: new evidence c_t selectively updates branches
  4. UnitaryMixer: coherent evolution of branch hidden states
  5. AmplitudeRouter: complex amplitude routing between branches
  6. CollapseIndex: compute χ = 1 - max_k p^(k)
  7. ObserverModule: if χ < threshold → trigger DecoherenceModule
  8. DecoherenceModule: collapse branches → single hidden state
  9. Output head: produce logits over vocabulary
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from qrr.branch_bank import BranchBank, BranchState
from qrr.unitary_mixer import UnitaryMixer
from qrr.amplitude_router import AmplitudeRouter
from qrr.entanglement_module import EntanglementModule
from qrr.decoherence_module import DecoherenceModule
from qrr.observer_module import ObserverModule
from qrr.collapse_index import CollapseIndex


class QRRModel(nn.Module):
    """
    Full Quantum Relational Reasoner.

    Args:
        base_model_name:  HuggingFace model ID (default 'gpt2').
        k_branches:       Number of competing hypotheses K (default 4).
        chi_threshold:    Collapse trigger threshold (default 0.3).
        use_complex:      Enable complex phase routing (Stage 2, default False).
        freeze_base:      Freeze base transformer weights (default True).
        mixer_parametrize: 'cayley' or 'learned' for UnitaryMixer.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        k_branches: int = 4,
        chi_threshold: float = 0.3,
        use_complex: bool = False,
        freeze_base: bool = True,
        mixer_parametrize: str = "learned",
    ) -> None:
        super().__init__()
        self.K = k_branches
        self.chi_threshold = chi_threshold

        # --- Base transformer (encoder) ---
        self.base = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        hidden_dim = self.base.config.hidden_size
        self.hidden_dim = hidden_dim

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        # --- QRR modules ---
        self.branch_bank = BranchBank(hidden_dim, k_branches)
        self.entanglement = EntanglementModule(hidden_dim, k_branches)
        self.mixer = UnitaryMixer(hidden_dim, k_branches, parametrize=mixer_parametrize)
        self.router = AmplitudeRouter(k_branches, hidden_dim, use_complex=use_complex)
        self.collapse_idx = CollapseIndex(threshold=chi_threshold)
        self.decoherence = DecoherenceModule(strategy="weighted_sum")
        self.observer = ObserverModule(hidden_dim, chi_threshold=chi_threshold)

        # --- Output head ---
        vocab_size = self.base.config.vocab_size
        self.output_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Core forward pass (single step)
    # ------------------------------------------------------------------

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run base transformer → pooled hidden state (batch, d)."""
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over sequence dimension
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else torch.ones(*outputs.last_hidden_state.shape[:2], 1, device=outputs.last_hidden_state.device)
        h = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return h  # (batch, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        forced_collapse: bool = False,
    ) -> dict:
        """
        Full QRR forward pass.

        Returns dict with:
          logits, chi, branch_state, collapsed_h, did_collapse
        """
        # 1. Encode input
        c_t = self.encode(input_ids, attention_mask)   # (batch, d)

        # 2. Initialize branch bank from context
        state: BranchState = self.branch_bank(c_t)

        # 3. Entanglement: update branches with new evidence
        hidden, amplitudes = self.entanglement(
            state.hidden, state.amplitudes, c_t
        )

        # 4. Coherent evolution: unitary mixing of hidden states
        hidden = self.mixer(hidden, c_t)

        # 5. Amplitude routing: complex amplitude evolution
        amplitudes = self.router(amplitudes, c_t)

        # 6. Recompute probabilities and χ
        norm_sq = (amplitudes[..., 0] ** 2 + amplitudes[..., 1] ** 2)  # (batch, K)
        probabilities = norm_sq / (norm_sq.sum(-1, keepdim=True) + 1e-9)
        chi = self.collapse_idx(probabilities)                           # (batch,)

        # Rebuild BranchState with updated values
        state = BranchState(
            hidden=hidden,
            amplitudes=amplitudes,
            probabilities=probabilities,
            chi=chi,
        )

        # 7–8. Observer decides whether to collapse
        should_obs = self.observer.should_observe(chi, forced=forced_collapse)
        did_collapse = should_obs.any().item()

        # Always collapse for output (training requires gradient through decoherence)
        collapsed_h = self.decoherence(state)          # (batch, d)
        if did_collapse:
            collapsed_h = self.observer.observe(collapsed_h, chi, forced=forced_collapse)

        # 9. Output logits
        logits = self.output_head(collapsed_h)         # (batch, vocab)

        return {
            "logits": logits,
            "chi": chi,
            "branch_state": state,
            "collapsed_h": collapsed_h,
            "did_collapse": did_collapse,
        }

    # ------------------------------------------------------------------
    # Convenience: forward_with_branches (for EXP-001)
    # ------------------------------------------------------------------

    def forward_with_branches(self, text: str) -> dict:
        """
        Tokenize text, run forward, return χ + branch diversity.
        Convenience method for experiment scripts and examples.
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            out = self.forward(enc["input_ids"], enc.get("attention_mask"))
        return {
            "chi": out["chi"].mean().item(),
            "probabilities": out["branch_state"].probabilities.squeeze(0).tolist(),
            "branch_diversity": self.branch_bank.branch_diversity(out["branch_state"]).item(),
            "did_collapse": out["did_collapse"],
        }

    # ------------------------------------------------------------------
    # Demo
    # ------------------------------------------------------------------

    def demo(self) -> None:
        """Quick sanity check — print χ for a few example inputs."""
        examples = [
            ("AMBIGUOUS", "The bank was steep."),
            ("AMBIGUOUS", "I saw the man with the telescope."),
            ("AMBIGUOUS", "Book the flight or the hotel first."),
            ("CLEAR",     "Water boils at 100 degrees Celsius."),
            ("CLEAR",     "The Eiffel Tower is in Paris."),
            ("CLEAR",     "Python is a programming language."),
        ]
        print("\nQRR Demo — χ (collapse index) per input")
        print("=" * 55)
        for label, text in examples:
            result = self.forward_with_branches(text)
            chi = result["chi"]
            bar = "█" * int(chi * 20) + "░" * (20 - int(chi * 20))
            print(f"[{label:9s}] [{bar}] χ={chi:.3f}  '{text[:40]}'")
        print("\nExpected: ambiguous inputs → higher χ")
        print("(Before fine-tuning this signal may be weak — that's EXP-001)")
