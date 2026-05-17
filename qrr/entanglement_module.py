"""
Entanglement Module — context-branch correlation across time.

Analog of quantum entanglement: the state of each branch becomes
correlated with the observation history (context), not just the
current token.

In standard transformers, context is a flat attention over all tokens.
In QRR, each branch maintains its own context vector, and branches
become "entangled" when they share observations — their hidden states
become correlated in a structured way.

Implementation:
  Each branch k maintains a context memory c^(k) ∈ R^d.
  At each step, c^(k) is updated via cross-attention between
  branch k's hidden state and the full token sequence.
  Branches that have observed the same tokens develop similar
  context vectors (positive entanglement).
  Branches that diverged in interpretation develop orthogonal
  context vectors (negative entanglement / superorthogonality).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from qrr.branch_bank import BranchState


class EntanglementModule(nn.Module):
    """
    Maintains per-branch context memory and computes
    context-branch correlation (entanglement).

    Args:
        hidden_dim:   Branch hidden state dimensionality.
        k_branches:   Number of branches.
        context_len:  Maximum context sequence length.
        num_heads:    Attention heads for cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        k_branches: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d = hidden_dim
        self.K = k_branches

        # Cross-attention: each branch attends to the token sequence
        # Query = branch hidden state, Key/Value = token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Context update gate (GRU-style)
        self.context_gate = nn.GRUCell(hidden_dim, hidden_dim)

        # Entanglement scorer: measures correlation between branch pairs
        # Used for the entanglement loss term
        self.entangle_proj = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(
        self,
        state: BranchState,
        token_sequence: torch.Tensor,  # (batch, seq_len, d)
        context_memory: torch.Tensor | None = None,  # (batch, K, d)
    ) -> tuple[BranchState, torch.Tensor]:
        """
        Update branch hidden states via cross-attention with token sequence.
        Returns updated BranchState and updated context memory.

        Args:
            state:          Current BranchState (batch, K, d)
            token_sequence: Full token embeddings (batch, seq_len, d)
            context_memory: Previous context memory per branch (batch, K, d)
                            If None, initialized to zeros.

        Returns:
            (updated_state, new_context_memory)
        """
        batch, K, d = state.hidden.shape
        seq_len = token_sequence.size(1)

        if context_memory is None:
            context_memory = torch.zeros(batch, K, d, device=state.hidden.device)

        # Process each branch independently via cross-attention
        h_updated = []
        ctx_updated = []

        for k in range(K):
            h_k = state.hidden[:, k:k+1, :]  # (batch, 1, d) — query

            # Cross-attend to full token sequence
            h_k_attn, _ = self.cross_attn(
                query=h_k,
                key=token_sequence,
                value=token_sequence,
            )  # (batch, 1, d)

            h_k_attn = h_k_attn.squeeze(1)  # (batch, d)

            # Update context memory via GRU gate
            ctx_k = self.context_gate(h_k_attn, context_memory[:, k, :])  # (batch, d)

            h_updated.append(h_k_attn)
            ctx_updated.append(ctx_k)

        h_new = torch.stack(h_updated, dim=1)    # (batch, K, d)
        ctx_new = torch.stack(ctx_updated, dim=1)  # (batch, K, d)

        # Recompute χ from unchanged amplitudes (hidden states changed, not amplitudes)
        from qrr.branch_bank import BranchState as BS
        new_state = BS(
            hidden=h_new,
            amplitudes=state.amplitudes,
            probabilities=state.probabilities,
            chi=state.chi,
        )

        return new_state, ctx_new

    def entanglement_score(
        self, state: BranchState
    ) -> torch.Tensor:
        """
        Compute pairwise entanglement scores between branches.
        High score = branches are highly correlated (possibly redundant).
        Low score = branches are exploring different hypotheses.

        Returns: (batch, K, K) correlation matrix
        """
        h = state.hidden  # (batch, K, d)
        proj = self.entangle_proj(h)   # (batch, K, d//2)
        proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-9)
        return torch.bmm(proj_norm, proj_norm.transpose(1, 2))  # (batch, K, K)
