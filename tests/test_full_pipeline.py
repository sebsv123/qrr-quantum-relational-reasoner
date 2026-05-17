"""
Integration tests — full QRR forward pass pipeline.

Run:
    pytest tests/test_full_pipeline.py -v
"""

import torch
import pytest
from qrr.qrr_model import QRRModel
from qrr.unitary_mixer import UnitaryMixer
from qrr.amplitude_router import AmplitudeRouter
from qrr.entanglement_module import EntanglementModule
from training.loss import QRRLoss


BATCH = 2
K = 4
D = 32   # small for fast tests
SEQ = 8


@pytest.fixture(scope="module")
def model():
    """Small GPT-2 QRR for fast integration tests."""
    return QRRModel(base_model_name="gpt2", k_branches=K, freeze_base=True)


class TestUnitaryMixer:
    def test_output_shape(self):
        mixer = UnitaryMixer(hidden_dim=D, k_branches=K)
        h = torch.randn(BATCH, K, D)
        c = torch.randn(BATCH, D)
        out = mixer(h, c)
        assert out.shape == (BATCH, K, D)

    def test_learned_does_not_explode(self):
        mixer = UnitaryMixer(hidden_dim=D, k_branches=K, parametrize="learned")
        h = torch.randn(BATCH, K, D)
        c = torch.randn(BATCH, D)
        out = mixer(h, c)
        assert not torch.isnan(out).any()
        assert out.abs().max() < 1e4


class TestAmplitudeRouter:
    def test_output_shape(self):
        router = AmplitudeRouter(k_branches=K, context_dim=D)
        a = torch.randn(BATCH, K, 2)
        c = torch.randn(BATCH, D)
        out = router(a, c)
        assert out.shape == (BATCH, K, 2)

    def test_phase_map_range(self):
        router = AmplitudeRouter(k_branches=K, context_dim=D)
        a = torch.randn(BATCH, K, 2)
        pmap = router.phase_interference_map(a)
        assert pmap.shape == (BATCH, K, K)
        assert (pmap >= -1.0).all() and (pmap <= 1.0).all()

    def test_complex_activation(self):
        router = AmplitudeRouter(k_branches=K, context_dim=D, use_complex=False)
        assert not router.use_complex
        router.activate_complex_phases()
        assert router.use_complex


class TestEntanglementModule:
    def test_output_shapes(self):
        ent = EntanglementModule(hidden_dim=D, k_branches=K)
        h = torch.randn(BATCH, K, D)
        a = torch.randn(BATCH, K, 2)
        c = torch.randn(BATCH, D)
        h_out, a_out = ent(h, a, c)
        assert h_out.shape == (BATCH, K, D)
        assert a_out.shape == (BATCH, K, 2)

    def test_amplitude_normalization(self):
        ent = EntanglementModule(hidden_dim=D, k_branches=K)
        h = torch.randn(BATCH, K, D)
        # Start with normalized amplitudes
        a_raw = torch.randn(BATCH, K, 2)
        a_norm = a_raw / (a_raw.norm(dim=-1, keepdim=True) + 1e-9)
        c = torch.randn(BATCH, D)
        _, a_out = ent(h, a_norm, c)
        # After update, amplitudes should still be reasonable magnitude
        assert not torch.isnan(a_out).any()


class TestQRRModelForward:
    def test_forward_shapes(self, model):
        input_ids = torch.randint(0, 1000, (BATCH, SEQ))
        attn_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids, attn_mask)
        assert out["logits"].shape == (BATCH, model.base.config.vocab_size)
        assert out["chi"].shape == (BATCH,)

    def test_chi_in_range(self, model):
        input_ids = torch.randint(0, 1000, (BATCH, SEQ))
        out = model(input_ids)
        assert (out["chi"] >= 0).all() and (out["chi"] <= 1).all()

    def test_no_nan(self, model):
        input_ids = torch.randint(0, 1000, (BATCH, SEQ))
        out = model(input_ids)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isnan(out["chi"]).any()

    def test_forward_with_branches(self, model):
        result = model.forward_with_branches("The bank was steep.")
        assert "chi" in result
        assert "probabilities" in result
        assert 0.0 <= result["chi"] <= 1.0


class TestQRRLoss:
    def test_loss_terms_finite(self, model):
        loss_fn = QRRLoss()
        input_ids = torch.randint(0, 1000, (BATCH, SEQ))
        out = model(input_ids)
        targets = torch.randint(0, model.base.config.vocab_size, (BATCH,))
        is_ambiguous = torch.tensor([True, False])
        losses = loss_fn(out["logits"], targets, out["branch_state"], is_ambiguous, model.decoherence)
        for name, val in losses.items():
            assert torch.isfinite(val), f"Loss term '{name}' is not finite: {val}"

    def test_total_is_sum_of_terms(self, model):
        loss_fn = QRRLoss(lambda_coh=0.1, lambda_dec=0.1, lambda_cal=0.05, lambda_div=0.1)
        input_ids = torch.randint(0, 1000, (BATCH, SEQ))
        out = model(input_ids)
        targets = torch.randint(0, model.base.config.vocab_size, (BATCH,))
        is_ambiguous = torch.tensor([True, False])
        losses = loss_fn(out["logits"], targets, out["branch_state"], is_ambiguous, model.decoherence)
        expected = (losses["token"] + 0.1*losses["coh"] + 0.1*losses["dec"]
                    + 0.05*losses["cal"] + 0.1*losses["div"])
        assert torch.allclose(losses["total"], expected, atol=1e-5)
