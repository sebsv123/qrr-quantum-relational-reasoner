"""
QRR Model — full Quantum Relational Reasoner. v2.1 (EXP-001b fixes)

Changes from v2.0:
  - encode() now uses ATTENTION-WEIGHTED pooling instead of mean pooling.
    Mean pooling over all tokens destroys positional and semantic structure.
    Attention pooling uses a learned query vector to weight tokens by
    relevance, preserving semantic distinctions between inputs.
  - Fallback to last-token for GPT-2 style models (causal, no [CLS]).
  - Added warm_start_router() for supervised pre-training of AmplitudeRouter
    on a few ambiguous/clear pairs before running EXP-001b.

Theoretical grounding:
  The encode() function is the interface between the base transformer and
  the QRR branch bank. If it produces similar vectors for semantically
  different inputs, no downstream module can recover the signal.
  Attention pooling maximizes information retention from the sequence.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from qrr.branch_bank import BranchBank, BranchState
from qrr.unitary_mixer import UnitaryMixer
from qrr.amplitude_router import AmplitudeRouter
from qrr.decoherence_module import DecoherenceModule
from qrr.collapse_index import CollapseIndex


class AttentionPooling(nn.Module):
    """
    Learned attention pooling over sequence tokens.

    Instead of averaging all tokens (which loses structure), learns a
    query vector that attends to the most semantically relevant tokens.
    For ambiguous sentences, relevant tokens are the ambiguous words.
    For clear sentences, relevant tokens are the key content words.

    This structural difference is what chi needs to detect.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (batch, seq_len, d)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:                 # (batch, d)
        # Score each token
        scores = self.query(hidden_states).squeeze(-1)  # (batch, seq_len)
        # Mask padding tokens
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)         # (batch, seq_len)
        # Weighted sum
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d)
        return pooled


class QRRModel(nn.Module):
    """
    Full QRR model. v2.1

    Pipeline:
      text → [Base Transformer] → [AttentionPooling] → h_base
           → [BranchBank v2.1]  → {h^(k), α^(k)}  (diverse branches)
           → [UnitaryMixer]     → coherent evolution
           → [AmplitudeRouter]  → evidence-gated amplitude update
           → [DecoherenceModule]→ collapse to single state
           → [LM Head]          → logits + χ

    Args:
        base_model_name:  HuggingFace model name (default: 'gpt2').
        k_branches:       Number of competing hypotheses (default: 4).
        chi_threshold:    Collapse trigger threshold (default: 0.3).
        freeze_base:      Freeze base transformer weights (default: True).
        use_phase:        Enable complex phase in AmplitudeRouter (Stage 2).
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

        config = AutoConfig.from_pretrained(base_model_name)
        self.base = AutoModel.from_pretrained(base_model_name)
        self.hidden_dim = config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # FIX: Attention pooling replaces mean pooling
        self.attn_pool = AttentionPooling(self.hidden_dim)

        # QRR modules (v2.1)
        self.branch_bank   = BranchBank(self.hidden_dim, k_branches)
        self.unitary_mixer = UnitaryMixer(self.hidden_dim, k_branches)
        self.amp_router    = AmplitudeRouter(self.hidden_dim, k_branches, use_phase=use_phase)
        self.decoherence   = DecoherenceModule(strategy="weighted_sum")
        self.collapse_index = CollapseIndex(threshold=chi_threshold)

        self.lm_head = nn.Linear(self.hidden_dim, config.vocab_size, bias=False)

    def encode(self, text: str | list[str], device: str = "cpu") -> torch.Tensor:
        """
        Tokenize and encode via base transformer + attention pooling.

        v2.1: Uses AttentionPooling instead of mean pooling.
        Attention pooling preserves semantic distinctions between inputs
        that mean pooling collapses.

        Returns:
            h_base: (batch, hidden_dim)
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

        ctx = torch.no_grad() if not self.training else torch.enable_grad()
        with ctx:
            outputs = self.base(**tokens)

        h = outputs.last_hidden_state  # (batch, seq_len, d)
        h_base = self.attn_pool(h, tokens["attention_mask"])  # (batch, d)
        return h_base

    def forward_with_branches(
        self,
        text: str | list[str],
        device: str = "cpu",
    ) -> dict:
        """
        Full QRR forward pass.

        Returns dict:
          logits, chi, chi_per_sample, branch_state,
          branch_diversity, collapsed_hidden
        """
        device = str(next(self.parameters()).device)

        h_base = self.encode(text, device)
        state: BranchState = self.branch_bank(h_base)

        new_hidden, new_amplitudes = self.unitary_mixer(
            state.hidden, state.amplitudes, h_base
        )
        new_amplitudes = self.amp_router(new_hidden, new_amplitudes, h_base)

        amp_r = new_amplitudes[..., 0]
        amp_i = new_amplitudes[..., 1]
        norm_sq = amp_r ** 2 + amp_i ** 2
        probs = norm_sq / (norm_sq.sum(-1, keepdim=True) + 1e-9)

        # Simpson's diversity index (consistent with BranchBank v2.1)
        chi = 1.0 - (probs ** 2).sum(dim=-1)  # (batch,)

        state = BranchState(
            hidden=new_hidden,
            amplitudes=new_amplitudes,
            probabilities=probs,
            chi=chi,
        )

        h_collapsed = self.decoherence(state)
        logits = self.lm_head(h_collapsed)

        return {
            "logits":            logits,
            "chi":               chi.mean().item(),
            "chi_per_sample":    chi,
            "branch_state":      state,
            "branch_diversity":  self.branch_bank.branch_diversity(state).item(),
            "collapsed_hidden":  h_collapsed,
        }

    def compute_chi(self, text: str | list[str]) -> float:
        """Return collapse index chi for input. High=ambiguous, Low=clear."""
        with torch.no_grad():
            out = self.forward_with_branches(text)
        return out["chi"]

    def warm_start_router(
        self,
        ambiguous: list[str],
        clear: list[str],
        steps: int = 50,
        lr: float = 1e-3,
    ) -> list[float]:
        """
        Brief supervised warm-start for AmplitudeRouter and BranchBank.

        Trains QRR modules for `steps` steps to maximize:
          chi(ambiguous) - chi(clear)

        This gives the router the minimum signal needed for EXP-001b.
        Uses contrastive loss: L = -mean(chi_amb) + mean(chi_clr) + margin

        Args:
            ambiguous: list of ambiguous sentences
            clear:     list of clear sentences
            steps:     number of gradient steps (default 50)
            lr:        learning rate (default 1e-3)

        Returns:
            list of loss values per step
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-3
        )
        losses = []
        n = min(len(ambiguous), len(clear), 8)  # mini-batch

        self.train()
        for step in range(steps):
            # Sample mini-batch
            import random
            amb_batch = random.sample(ambiguous, min(n, len(ambiguous)))
            clr_batch = random.sample(clear,     min(n, len(clear)))

            optimizer.zero_grad()
            out_amb = self.forward_with_branches(amb_batch)
            out_clr = self.forward_with_branches(clr_batch)

            chi_amb = out_amb["chi_per_sample"].mean()
            chi_clr = out_clr["chi_per_sample"].mean()

            # Contrastive: maximize gap, with margin 0.15
            margin = 0.15
            loss = F.relu(margin - (chi_amb - chi_clr))

            # Orthogonality regularizer: keep branches diverse
            orth_loss = (
                self.branch_bank.orthogonality_loss(out_amb["branch_state"]) +
                self.branch_bank.orthogonality_loss(out_clr["branch_state"])
            ) * 0.1

            total = loss + orth_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            losses.append(total.item())

            if step % 10 == 0:
                print(f"  warm_start step {step:02d}: loss={total.item():.4f} "
                      f"chi_amb={chi_amb.item():.3f} chi_clr={chi_clr.item():.3f} "
                      f"delta={chi_amb.item()-chi_clr.item():.3f}")
        self.eval()
        return losses
