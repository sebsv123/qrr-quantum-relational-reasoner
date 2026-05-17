"""
QRR Training Loop — Stage 1 (real amplitudes, frozen base).

Trains QRR modules (BranchBank, UnitaryMixer, AmplitudeRouter, DecoherenceModule)
on top of a frozen pre-trained language model.

Usage:
    python training/train.py --config training/config_gpt2_stage1.yaml
    python training/train.py --model gpt2 --branches 4 --epochs 5 --lr 1e-4
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from qrr.qrr_model import QRRModel
from training.loss import QRRLoss


class AmbiguityDataset(Dataset):
    """
    Simple dataset of (text, label) pairs where label=1 means ambiguous.
    Used for Stage 1 curriculum training.
    """
    def __init__(self, samples: list[tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    texts  = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.float)
    return texts, labels


def train_epoch(
    model: QRRModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: QRRLoss,
    device: str,
    epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    metrics_sum = {"L_div": 0.0, "L_dec": 0.0, "L_cal": 0.0}
    n_batches = 0

    for texts, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad()

        out = model.forward_with_branches(texts)
        state = out["branch_state"]

        loss, breakdown = loss_fn(
            logits=out["logits"],
            branch_state=state,
            labels=labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += breakdown.get(k, 0.0)
        n_batches += 1

    scheduler.step()

    return {
        "epoch": epoch,
        "loss":  total_loss / max(n_batches, 1),
        **{k: v / max(n_batches, 1) for k, v in metrics_sum.items()},
    }


@torch.no_grad()
def eval_epoch(
    model: QRRModel,
    loader: DataLoader,
    loss_fn: QRRLoss,
    device: str,
) -> dict:
    model.eval()
    chi_ambiguous, chi_clear = [], []

    for texts, labels in loader:
        labels = labels.to(device)
        out = model.forward_with_branches(texts)
        chi = out["chi_per_sample"].cpu()
        for i, lbl in enumerate(labels.cpu()):
            if lbl == 1:
                chi_ambiguous.append(chi[i].item())
            else:
                chi_clear.append(chi[i].item())

    if chi_ambiguous and chi_clear:
        delta = sum(chi_ambiguous) / len(chi_ambiguous) - sum(chi_clear) / len(chi_clear)
    else:
        delta = 0.0

    return {
        "chi_ambiguous": sum(chi_ambiguous) / max(len(chi_ambiguous), 1),
        "chi_clear":     sum(chi_clear)     / max(len(chi_clear), 1),
        "delta_chi":     delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="gpt2")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--epochs",   type=int, default=10)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--output",   default="checkpoints/stage1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Model: {args.model} | K={args.branches} | lr={args.lr} | epochs={args.epochs}\n")

    model = QRRModel(
        base_model_name=args.model,
        k_branches=args.branches,
        freeze_base=True,
    ).to(device)

    # Minimal demo data — replace with AmbigQA in real training
    from benchmarks.ambiguity_bench import AMBIGUOUS_INPUTS, CLEAR_INPUTS
    samples = [(t, 1) for t in AMBIGUOUS_INPUTS] + [(t, 0) for t in CLEAR_INPUTS]
    dataset = AmbiguityDataset(samples)
    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = QRRLoss(lambda_div=0.1, lambda_dec=0.05, lambda_cal=0.05)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        log = {**train_metrics, **val_metrics, "elapsed": elapsed}
        history.append(log)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={log['loss']:.4f} | "
            f"Δχ={log['delta_chi']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"epoch_{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": log,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {output_dir}/training_history.json")


if __name__ == "__main__":
    main()
