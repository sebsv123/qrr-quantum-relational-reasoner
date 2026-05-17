"""
QRR Model — full Quantum Relational Reasoner.

Wraps a HuggingFace causal language model (default: GPT-2) with the
QRR branch bank pipeline:

  token → [Base Transformer] → h_base
         → [BranchBank]     → {h^(k), α^(k)}
         → [UnitaryMixer]   → coherent evolution
         → [AmplitudeRouter]→ evidence-gated amplitude update
         → [DecoherenceModule] → collapse when χ < threshold
         → [LM Head]        → logits + χ

Usage:
    model = QRRModel(base_model_name="gpt2", k_branches=4)
    output = model.forward_with_branches("The bank was steep.")
    print(output['chi'])  # collapse index
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from qrr.branch_bank import BranchBank, BranchState
from qrr.unitary_mixer import UnitaryMixer
from qrr.amplitude_router import AmplitudeRouter
from qrr.decoherence_module import DecoherenceModule
from qrr.collapse_index import CollapseIndex


class QRRModel(nn.Module):
    """
    Full QRR model.

    Args:
        base_model_name:  HuggingFace model name (default: 'gpt2').
        k_branches:       Number of competing hypotheses (default: 4).
        chi_threshold:    Collapse trigger threshold (default: 0.3).
        freeze_base:      If True, freeze base transformer weights (default: True).
                          Set False for full fine-tuning.
        use_phase:        If True, enable complex phase in AmplitudeRouter (Stage 2).
                          Default False (Stage 1 — real amplitudes only).
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        k_branches: int = 4,
        chi_threshold: float = 0.3,
        freeze_base: bool = True,
        use_phase: bool = False,
    ) -> None:
        super().__init__()
        self.K = k_branches
        self.chi_threshold = chi_threshold

        # Load base transformer (encoder only — we use hidden states, not logits)
        config = AutoConfig.from_pretrained(base_model_name)
        self.base = AutoModel.from_pretrained(base_model_name)
        self.hidden_dim = config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # QRR modules
        self.branch_bank    = BranchBank(self.hidden_dim, k_branches)
        self.unitary_mixer  = UnitaryMixer(self.hidden_dim, k_branches)
        self.amp_router     = AmplitudeRouter(self.hidden_dim, k_branches, use_phase=use_phase)
        self.decoherence    = DecoherenceModule(strategy="weighted_sum")
        self.collapse_index = CollapseIndex(threshold=chi_threshold)

        # Output head: collapsed hidden state → vocab logits
        self.lm_head = nn.Linear(self.hidden_dim, config.vocab_size, bias=False)

    def encode(self, text: str | list[str], device: str = "cpu") -> torch.Tensor:
        """
        Tokenize and encode text through base transformer.

        Returns:
            h_base: (batch, hidden_dim) — mean-pooled last hidden state
        """
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.base(**tokens)
        # Mean pool over sequence length
        h = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        h_base = (h * mask).sum(1) / mask.sum(1)  # (batch, hidden_dim)
        return h_base

    def forward_with_branches(
        self,
        text: str | list[str],
        device: str = "cpu",
    ) -> dict:
        """
        Full QRR forward pass with branch inspection.

        Returns dict with:
          'logits':          (batch, vocab_size)
          'chi':             scalar — mean collapse index over batch
          'branch_state':    BranchState namedtuple
          'branch_diversity': scalar — mean pairwise cosine distance
          'collapsed_hidden': (batch, hidden_dim)
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else device

        # 1. Encode through base transformer
        h_base = self.encode(text, str(device))

        # 2. Initialize branch bank
        state: BranchState = self.branch_bank(h_base)

        # 3. Coherent evolution via unitary mixer
        new_hidden, new_amplitudes = self.unitary_mixer(
            state.hidden, state.amplitudes, h_base
        )

        # 4. Evidence-gated amplitude update
        new_amplitudes = self.amp_router(new_hidden, new_amplitudes, h_base)

        # 5. Recompute probabilities and χ
        amp_r = new_amplitudes[..., 0]
        amp_i = new_amplitudes[..., 1]
        norm_sq = amp_r ** 2 + amp_i ** 2
        probs = norm_sq / (norm_sq.sum(-1, keepdim=True) + 1e-9)
        chi = self.collapse_index(probs)  # (batch,)

        # Rebuild state with updated tensors
        state = BranchState(
            hidden=new_hidden,
            amplitudes=new_amplitudes,
            probabilities=probs,
            chi=chi,
        )

        # 6. Collapse branches → single hidden state
        h_collapsed = self.decoherence(state)   # (batch, hidden_dim)

        # 7. LM head
        logits = self.lm_head(h_collapsed)      # (batch, vocab_size)

        return {
            "logits":            logits,
            "chi":               chi.mean().item(),
            "chi_per_sample":    chi,
            "branch_state":      state,
            "branch_diversity":  self.branch_bank.branch_diversity(state).item(),
            "collapsed_hidden":  h_collapsed,
        }

    def compute_chi(self, text: str | list[str]) -> float:
        """
        Convenience: return just the collapse index χ for a text input.
        High χ → ambiguous. Low χ → clear.
        """
        with torch.no_grad():
            out = self.forward_with_branches(text)
        return out["chi"]
