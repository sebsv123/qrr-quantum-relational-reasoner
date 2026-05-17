"""
QRR Training Loop.

Trains the QRR modules (branch bank, mixer, router, entanglement)
on top of a frozen base transformer.

The base LM is frozen by default — we only train the QRR modules.
This keeps compute requirements low for Phase 1 experiments.

Usage:
    python training/train.py \\
        --model gpt2 \\
        --k 4 \\
        --epochs 3 \\
        --lr 1e-4 \\
        --batch_size 16 \\
        --dataset wikitext
"""

from __future__ import annotations
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from qrr.qrr_model import QRRModel
from training.loss import QRRLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QRR modules")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--k", type=int, default=4, help="Number of branches")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dataset", default="wikitext", choices=["wikitext", "ptb"])
    p.add_argument("--chi_threshold", type=float, default=0.3)
    p.add_argument("--lambda_coh", type=float, default=0.1)
    p.add_argument("--lambda_dec", type=float, default=0.1)
    p.add_argument("--lambda_div", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_dataloader(tokenizer, dataset_name: str, split: str, batch_size: int, max_length: int) -> DataLoader:
    if dataset_name == "wikitext":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    else:
        raw = load_dataset("ptb_text_only", split=split)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt"
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True, drop_last=True)


def train_epoch(
    model: QRRModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    loss_fn: QRRLoss,
    device: str,
    log_every: int,
) -> dict:
    model.train()
    totals = {"total": 0.0, "token": 0.0, "coh": 0.0, "dec": 0.0, "cal": 0.0, "div": 0.0}
    steps = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Targets: next-token prediction (shift right)
        targets = input_ids[:, 1:].contiguous().view(-1)
        input_ids_in = input_ids[:, :-1].contiguous()
        attention_mask_in = attention_mask[:, :-1].contiguous()

        out = model(input_ids_in, attention_mask_in)
        logits = out["logits"].view(-1, model.output_head.out_features)
        state = out["branch_state"]

        # Heuristic ambiguity label: chi > 0.5 at initialization ≈ ambiguous
        # In Phase 3, this will be replaced with dataset-provided labels
        is_ambiguous = (out["chi"] > 0.5).detach()

        losses = loss_fn(
            logits, targets, state, is_ambiguous,
            decoherence_module=model.decoherence
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        for k, v in losses.items():
            totals[k] += v.item()
        steps += 1

        if steps % log_every == 0:
            print(f"  step {steps} | " + " | ".join(f"{k}={v/steps:.4f}" for k, v in totals.items()))

    return {k: v / steps for k, v in totals.items()}


def main():
    args = parse_args()
    print(f"Training QRR | model={args.model} | K={args.k} | device={args.device}")

    model = QRRModel(
        base_model_name=args.model,
        k_branches=args.k,
        chi_threshold=args.chi_threshold,
        freeze_base=True,
    ).to(args.device)

    # Only train QRR modules (base is frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    train_loader = build_dataloader(model.tokenizer, args.dataset, "train", args.batch_size, args.max_length)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    loss_fn = QRRLoss(
        lambda_coh=args.lambda_coh,
        lambda_dec=args.lambda_dec,
        lambda_div=args.lambda_div,
    )

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, args.device, args.log_every)
        print(f"Epoch {epoch} summary: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        ckpt = f"{args.save_dir}/qrr_{args.model.replace('/', '_')}_K{args.k}_ep{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics}, ckpt)
        print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
