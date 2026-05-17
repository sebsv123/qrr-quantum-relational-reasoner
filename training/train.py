"""
QRR Training Loop.

Trains the QRR modules on top of a frozen base transformer.
Only the QRR modules (BranchBank, UnitaryMixer, AmplitudeRouter,
EntanglementModule) are updated. The base transformer is frozen.

Training requires an ambiguity-labeled dataset:
  Each sample: (text, is_ambiguous: bool)
  'is_ambiguous' drives L_coh and L_dec supervision.

Curriculum:
  Stage 1 (epochs 1-N): real amplitudes only (phase_scale=0)
  Stage 2 (epochs N+):  full complex amplitudes with phase shifts

Usage:
  python training/train.py --model gpt2 --epochs 20 --batch_size 8
"""

from __future__ import annotations
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from qrr.qrr_model import QRRModel
from training.loss import QRRLoss


class AmbiguityDataset(Dataset):
    """
    Minimal dataset wrapper for (text, is_ambiguous) pairs.
    Replace with your actual dataset loader.
    """

    def __init__(self, samples: list[tuple[str, bool]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, bool]:
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length=128):
    texts, labels = zip(*batch)
    encoding = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    is_ambiguous = torch.tensor(labels, dtype=torch.bool)
    return encoding["input_ids"], encoding["attention_mask"], is_ambiguous


def train(
    model_name: str = "gpt2",
    k_branches: int = 4,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    complex_phase_epoch: int = 10,  # Stage 2 starts here
    save_dir: str = "checkpoints/qrr_gpt2",
    lambda_coh: float = 0.1,
    lambda_dec: float = 0.1,
    lambda_cal: float = 0.05,
    lambda_div: float = 0.1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = QRRModel(
        base_model_name=model_name,
        k_branches=k_branches,
        freeze_base=True,
    ).to(device)

    # Only train QRR modules
    qrr_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("base_model.") and p.requires_grad
    ]
    print(f"Trainable QRR params: {sum(p.numel() for p in qrr_params):,}")

    optimizer = optim.AdamW(qrr_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = QRRLoss(
        lambda_coh=lambda_coh,
        lambda_dec=lambda_dec,
        lambda_cal=lambda_cal,
        lambda_div=lambda_div,
    )

    # Placeholder dataset — replace with real data
    dummy_samples = [
        ("The bank was steep and muddy.", True),
        ("Paris is the capital of France.", False),
        ("I saw her duck under the table.", True),
        ("Water boils at 100°C at sea level.", False),
    ] * 50

    dataset = AmbiguityDataset(dummy_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model.tokenizer),
    )

    print(f"\nStarting training: {epochs} epochs, {len(dataset)} samples")
    print(f"Stage 1 (real amplitudes): epochs 1-{complex_phase_epoch}")
    print(f"Stage 2 (complex phases):  epochs {complex_phase_epoch+1}-{epochs}\n")

    for epoch in range(1, epochs + 1):
        model.train()

        # Curriculum: disable phase shifts in Stage 1
        if epoch <= complex_phase_epoch:
            model.amplitude_router.phase_scale = 0.0
        else:
            import math
            model.amplitude_router.phase_scale = math.pi

        total_loss = 0.0
        loss_terms = {"token": 0., "coh": 0., "dec": 0., "cal": 0., "div": 0.}

        for step, (input_ids, attn_mask, is_ambiguous) in enumerate(loader):
            input_ids = input_ids.to(device)
            is_ambiguous = is_ambiguous.to(device)

            optimizer.zero_grad()

            out = model.forward(input_ids, return_branch_states=True)
            state = out["branch_state"]

            # Shift for next-token prediction
            logits = out["logits"]   # (batch, vocab)
            targets = input_ids[:, -1]  # last token as target (simplified)

            losses = criterion(
                logits=logits,
                targets=targets,
                state=state,
                is_ambiguous=is_ambiguous,
                decoherence_module=model.decoherence,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(qrr_params, max_norm=1.0)
            optimizer.step()

            total_loss += losses["total"].item()
            for k in loss_terms:
                loss_terms[k] += losses[k].item()

        scheduler.step()
        n = len(loader)
        stage = "S1" if epoch <= complex_phase_epoch else "S2"
        print(
            f"Epoch {epoch:03d}/{epochs} [{stage}] "
            f"loss={total_loss/n:.4f} "
            f"tok={loss_terms['token']/n:.4f} "
            f"coh={loss_terms['coh']/n:.4f} "
            f"dec={loss_terms['dec']/n:.4f} "
            f"cal={loss_terms['cal']/n:.4f} "
            f"div={loss_terms['div']/n:.4f}"
        )

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            model.save_qrr_modules(f"{save_dir}_ep{epoch:03d}")

    model.save_qrr_modules(save_dir)
    print(f"\nTraining complete. Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QRR modules")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--k_branches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints/qrr_gpt2")
    args = parser.parse_args()
    train(**vars(args))
