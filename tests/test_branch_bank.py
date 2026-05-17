"""
Unit tests for BranchBank and CollapseIndex.

Run:
    pytest tests/test_branch_bank.py -v
"""

import torch
import pytest
from qrr.branch_bank import BranchBank, BranchState
from qrr.collapse_index import CollapseIndex


BATCH = 4
D = 64
K = 4


@pytest.fixture
def bank():
    return BranchBank(hidden_dim=D, k_branches=K, init_uniform=True)


@pytest.fixture
def h_base():
    return torch.randn(BATCH, D)


class TestBranchBank:
    def test_output_shapes(self, bank, h_base):
        state = bank(h_base)
        assert state.hidden.shape == (BATCH, K, D)
        assert state.amplitudes.shape == (BATCH, K, 2)
        assert state.probabilities.shape == (BATCH, K)
        assert state.chi.shape == (BATCH,)

    def test_probabilities_sum_to_one(self, bank, h_base):
        state = bank(h_base)
        sums = state.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_chi_range(self, bank, h_base):
        state = bank(h_base)
        assert (state.chi >= 0).all()
        assert (state.chi <= 1).all()

    def test_uniform_init_gives_high_chi(self, bank, h_base):
        """Uniform init → all branches equal weight → χ should be (K-1)/K."""
        state = bank(h_base)
        expected = (K - 1) / K
        # With uniform init, χ ≈ (K-1)/K = 0.75 for K=4
        assert (state.chi > 0.5).all(), f"Expected χ > 0.5 for uniform init, got {state.chi}"

    def test_diversity_scalar(self, bank, h_base):
        state = bank(h_base)
        div = bank.branch_diversity(state)
        assert div.ndim == 0  # scalar
        assert 0.0 <= div.item() <= 1.0


class TestCollapseIndex:
    def test_forward_range(self):
        idx = CollapseIndex(threshold=0.3)
        probs = torch.softmax(torch.randn(BATCH, K), dim=-1)
        chi = idx(probs)
        assert chi.shape == (BATCH,)
        assert (chi >= 0).all() and (chi <= 1).all()

    def test_should_collapse(self):
        idx = CollapseIndex(threshold=0.3)
        # Low χ → should collapse
        probs_confident = torch.zeros(BATCH, K)
        probs_confident[:, 0] = 1.0
        chi = idx(probs_confident)
        assert idx.should_collapse(chi).all()

    def test_trace_accumulates(self):
        idx = CollapseIndex()
        probs = torch.softmax(torch.randn(BATCH, K), dim=-1)
        for _ in range(5):
            idx(probs)
        assert len(idx.get_trace()) == 5

    def test_reset_clears_trace(self):
        idx = CollapseIndex()
        probs = torch.softmax(torch.randn(BATCH, K), dim=-1)
        idx(probs)
        idx.reset()
        assert len(idx.get_trace()) == 0
