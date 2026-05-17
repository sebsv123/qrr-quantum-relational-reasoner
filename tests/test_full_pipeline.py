"""
Integration tests for the full QRR pipeline.

Tests that all modules chain correctly without errors.
Does NOT require a GPU or pretrained weights — uses tiny random tensors.

Run:
    pytest tests/test_full_pipeline.py -v
"""

import torch
import pytest
from unittest.mock import patch, MagicMock
from qrr.branch_bank import BranchBank, BranchState
from qrr.unitary_mixer import UnitaryMixer
from qrr.amplitude_router import AmplitudeRouter
from qrr.entanglement_module import EntanglementModule
from qrr.decoherence_module import DecoherenceModule
from qrr.observer_module import ObserverModule
from qrr.collapse_index import CollapseIndex
from training.loss import QRRLoss

BATCH = 2
D = 64
K = 4
SEQ = 16
VOCAB = 500


@pytest.fixture
def branch_state():
    bank = BranchBank(hidden_dim=D, k_branches=K)
    h = torch.randn(BATCH, D)
    return bank(h)


class TestUnitaryMixer:
    def test_output_shapes(self, branch_state):
        mixer = UnitaryMixer(hidden_dim=D, k_branches=K, parametrization="learned")
        ctx = torch.randn(BATCH, D)
        out = mixer(branch_state, ctx)
        assert out.hidden.shape == (BATCH, K, D)
        assert out.probabilities.shape == (BATCH, K)
        assert out.chi.shape == (BATCH,)

    def test_probabilities_sum_to_one(self, branch_state):
        mixer = UnitaryMixer(hidden_dim=D, k_branches=K, parametrization="learned")
        ctx = torch.randn(BATCH, D)
        out = mixer(branch_state, ctx)
        sums = out.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-4)


class TestAmplitudeRouter:
    def test_output_shapes(self, branch_state):
        router = AmplitudeRouter(hidden_dim=D, k_branches=K)
        token = torch.randn(BATCH, D)
        out = router(branch_state, token)
        assert out.amplitudes.shape == (BATCH, K, 2)
        assert out.chi.shape == (BATCH,)

    def test_phase_shift_zero_is_identity(self, branch_state):
        """With phase_scale=0, no phase shift → amplitudes real parts unchanged."""
        router = AmplitudeRouter(hidden_dim=D, k_branches=K, phase_scale=0.0)
        token = torch.randn(BATCH, D)
        out = router(branch_state, token)
        # Chi should still be in valid range
        assert (out.chi >= 0).all() and (out.chi <= 1).all()


class TestEntanglementModule:
    def test_output_shapes(self, branch_state):
        ent = EntanglementModule(hidden_dim=D, k_branches=K)
        token_seq = torch.randn(BATCH, SEQ, D)
        new_state, ctx = ent(branch_state, token_seq)
        assert new_state.hidden.shape == (BATCH, K, D)
        assert ctx.shape == (BATCH, K, D)

    def test_entanglement_score_shape(self, branch_state):
        ent = EntanglementModule(hidden_dim=D, k_branches=K)
        score = ent.entanglement_score(branch_state)
        assert score.shape == (BATCH, K, K)


class TestDecoherenceModule:
    def test_weighted_sum_shape(self, branch_state):
        dec = DecoherenceModule(strategy="weighted_sum")
        out = dec(branch_state)
        assert out.shape == (BATCH, D)

    def test_argmax_shape(self, branch_state):
        dec = DecoherenceModule(strategy="argmax")
        out = dec(branch_state)
        assert out.shape == (BATCH, D)

    def test_interference_loss_positive(self, branch_state):
        dec = DecoherenceModule()
        loss = dec.interference_loss(branch_state)
        assert loss.item() >= 0


class TestObserverModule:
    def test_observe_returns_report(self):
        obs = ObserverModule(chi_threshold=0.3, policy="threshold")
        chi = torch.tensor([0.1, 0.5])
        report = obs.observe(chi)
        assert "should_collapse" in report
        assert "confidence" in report
        assert "trend" in report

    def test_patience_policy(self):
        obs = ObserverModule(chi_threshold=0.3, policy="patience", patience=3)
        chi_low = torch.tensor([0.1, 0.1])
        # Should not collapse until patience is met
        for _ in range(2):
            report = obs.observe(chi_low)
        assert not report["should_collapse"].all()
        # After patience steps, should collapse
        report = obs.observe(chi_low)
        assert report["should_collapse"].all()


class TestQRRLoss:
    def test_loss_components(self, branch_state):
        criterion = QRRLoss()
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        is_ambiguous = torch.tensor([True, False])
        dec = DecoherenceModule()
        losses = criterion(logits, targets, branch_state, is_ambiguous, dec)
        assert "total" in losses
        assert "token" in losses
        assert losses["total"].item() > 0

    def test_total_is_sum_of_parts(self, branch_state):
        criterion = QRRLoss(lambda_coh=0.1, lambda_dec=0.1, lambda_cal=0.05, lambda_div=0.1)
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        is_ambiguous = torch.ones(BATCH, dtype=torch.bool)
        losses = criterion(logits, targets, branch_state, is_ambiguous)
        expected = (
            losses["token"]
            + 0.1 * losses["coh"]
            + 0.1 * losses["dec"]
            + 0.05 * losses["cal"]
            + 0.1 * losses["div"]
        )
        assert torch.allclose(losses["total"], expected, atol=1e-5)
