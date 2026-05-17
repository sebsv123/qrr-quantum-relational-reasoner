"""
QRR Model — full Quantum Relational Reasoner.

Wraps a HuggingFace causal language model as the base transformer,
then applies the full QRR pipeline on top:

  token → [Base Transformer] → h_t
       → [BranchBank]        → {(h^(k), α^(k))} K branches
       → [UnitaryMixer]      → coherent branch evolution
       → [EntanglementModule]→ context-branch correlation
       → [AmplitudeRouter]   → phase shifts & interference
       → [ObserverModule]    → monitors χ, decides when to collapse
       → [DecoherenceModule] → collapses when observer signals
       → [LM Head]           → token logits

The model can operate in two modes:
  'training':  always run full pipeline, return logits + all aux tensors
  'inference': run pipeline, collapse when ObserverModule signals
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
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
    Full Quantum Relational Reasoner model.

    Args:
        base_model_name:  HuggingFace model name (e.g. 'gpt2', 'gpt2-medium').
        k_branches:       Number of competing hypotheses (default 4).
        chi_threshold:    Collapse threshold for ObserverModule (default 0.3).
        observer_policy:  'threshold' | 'patience' | 'gradient' | 'learned'
        freeze_base:      If True, freeze base transformer weights (default True).
        mixer_param:      'cayley' | 'learned' for UnitaryMixer.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        k_branches: int = 4,
        chi_threshold: float = 0.3,
        observer_policy: str = "patience",
        freeze_base: bool = True,
        mixer_param: str = "cayley",
    ) -> None:
        super().__init__()

        # Base transformer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        # Infer hidden dim from base model config
        self.hidden_dim = self.base_model.config.hidden_size
        self.K = k_branches

        # QRR modules
        self.branch_bank = BranchBank(
            hidden_dim=self.hidden_dim,
            k_branches=k_branches,
            init_uniform=True,
        )
        self.unitary_mixer = UnitaryMixer(
            hidden_dim=self.hidden_dim,
            k_branches=k_branches,
            parametrization=mixer_param,
        )
        self.entanglement = EntanglementModule(
            hidden_dim=self.hidden_dim,
            k_branches=k_branches,
        )
        self.amplitude_router = AmplitudeRouter(
            hidden_dim=self.hidden_dim,
            k_branches=k_branches,
        )
        self.decoherence = DecoherenceModule(
            strategy="weighted_sum",
            temperature=1.0,
        )
        self.observer = ObserverModule(
            chi_threshold=chi_threshold,
            policy=observer_policy,
            patience=3,
        )
        self.collapse_index = CollapseIndex(threshold=chi_threshold)

    def _get_base_hidden(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run base transformer and extract:
          - pooled hidden state (batch, d)
          - full token sequence (batch, seq_len, d)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        # Last hidden layer: (batch, seq_len, d)
        token_sequence = outputs.hidden_states[-1]
        # Pool: mean over sequence (or use CLS-like last token for causal)
        h_pooled = token_sequence[:, -1, :]  # (batch, d) — last token as context
        return h_pooled, token_sequence

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_branch_states: bool = False,
    ) -> dict:
        """
        Full QRR forward pass.

        Returns dict with:
          'logits':         (batch, vocab) — final token logits
          'chi':            (batch,) — collapse index at final step
          'branch_state':   BranchState (if return_branch_states=True)
          'collapsed':      bool — whether decoherence was triggered
          'observation':    observer report dict
        """
        # 1. Base transformer
        h_pooled, token_sequence = self._get_base_hidden(input_ids)

        # 2. Initialize branch bank
        state = self.branch_bank(h_pooled)

        # 3. Unitary mixing (coherent evolution)
        state = self.unitary_mixer(state, context=h_pooled)

        # 4. Entanglement (context-branch correlation)
        state, ctx_memory = self.entanglement(state, token_sequence)

        # 5. Amplitude routing (phase shifts based on last token)
        last_token_emb = token_sequence[:, -1, :]  # (batch, d)
        state = self.amplitude_router(state, token_emb=last_token_emb)

        # 6. Observer: should we collapse now?
        observation = self.observer.observe(state.chi)

        # 7. Decoherence: collapse branches → single hidden state
        h_collapsed = self.decoherence(state)  # (batch, d)

        # 8. LM head (reuse base model's lm_head)
        logits = self.base_model.lm_head(h_collapsed)  # (batch, vocab)

        result = {
            "logits": logits,
            "chi": state.chi,
            "collapsed": observation["should_collapse"],
            "observation": observation,
        }
        if return_branch_states:
            result["branch_state"] = state
            result["context_memory"] = ctx_memory

        return result

    @torch.no_grad()
    def forward_with_branches(self, prompt: str) -> dict:
        """
        Convenience method for analysis: runs forward pass and returns
        branch-level statistics for a single string prompt.

        Returns dict with 'chi', 'branch_diversity', 'probabilities', 'logits'
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"]
        out = self.forward(input_ids, return_branch_states=True)
        state = out["branch_state"]
        diversity = self.branch_bank.branch_diversity(state)
        return {
            "chi": out["chi"].mean().item(),
            "branch_diversity": diversity.item(),
            "probabilities": state.probabilities.squeeze(0).tolist(),
            "logits": out["logits"],
            "observation": out["observation"],
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        return_branch_states: bool = False,
    ) -> dict:
        """
        Autoregressive generation with QRR pipeline.
        Tracks χ trace across all generation steps.

        Returns:
            'text':                Generated text string
            'chi_trace':           List of χ values per step
            'final_branch_weights': Branch probabilities at last step
            'branch_states':       List of BranchState (if requested)
        """
        self.eval()
        self.observer.reset()
        self.collapse_index.reset()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"]

        chi_trace = []
        branch_states = []
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            out = self.forward(
                generated_ids,
                return_branch_states=return_branch_states,
            )
            chi_trace.append(out["chi"].mean().item())
            if return_branch_states:
                branch_states.append(out["branch_state"])

            # Greedy next token
            next_token = out["logits"].argmax(dim=-1, keepdim=True)  # (batch, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids.size(1):],
            skip_special_tokens=True,
        )

        final_state = self.forward(generated_ids, return_branch_states=True)

        result = {
            "text": generated_text,
            "chi_trace": chi_trace,
            "final_branch_weights": final_state["branch_state"].probabilities.squeeze(0).tolist(),
        }
        if return_branch_states:
            result["branch_states"] = branch_states

        return result

    @classmethod
    def from_pretrained_qrr(cls, path: str, **kwargs) -> "QRRModel":
        """Load a saved QRR checkpoint."""
        import os
        model = cls(**kwargs)
        state_dict = torch.load(os.path.join(path, "qrr_modules.pt"), map_location="cpu")
        # Load only QRR modules (not base model)
        qrr_keys = {k: v for k, v in state_dict.items() if not k.startswith("base_model.")}
        model.load_state_dict(qrr_keys, strict=False)
        return model

    def save_qrr_modules(self, path: str) -> None:
        """Save only the QRR module weights (base model excluded)."""
        import os
        os.makedirs(path, exist_ok=True)
        qrr_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("base_model.")
        }
        torch.save(qrr_state, os.path.join(path, "qrr_modules.pt"))
        print(f"QRR modules saved to {path}/qrr_modules.pt")
