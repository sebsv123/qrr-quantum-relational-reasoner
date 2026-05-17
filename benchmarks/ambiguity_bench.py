"""
Ambiguity Benchmark — EXP-001 / EXP-001b.

EXP-001b adds warm_start option: trains QRR router for N steps on a
subset of the data before measuring chi discrimination.

Usage:
    # EXP-001 (zero-shot, original):
    python benchmarks/ambiguity_bench.py --model gpt2 --branches 4

    # EXP-001b (warm-start, 50 steps):
    python benchmarks/ambiguity_bench.py --model gpt2 --branches 4 --warm-start 50

    # EXP-001b with K=8:
    python benchmarks/ambiguity_bench.py --model gpt2 --branches 8 --warm-start 50
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from qrr.qrr_model import QRRModel

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

AMBIGUOUS_INPUTS = [
    "The bank was steep and covered in moss.",
    "She couldn't bear the weight of the news.",
    "I saw the bat in the cave.",
    "The pitcher was left on the mound.",
    "He spotted the duck near the pond.",
    "The crane stood perfectly still by the water.",
    "The bark was rough and peeling.",
    "They found the spring behind the old house.",
    "The match was over in minutes.",
    "She left her address at the counter.",
    "I saw the man with the telescope.",
    "The chicken is ready to eat.",
    "Visiting relatives can be boring.",
    "The professor urged the students to study harder in her office.",
    "The horse raced past the barn fell.",
    "I know a taller man than John.",
    "She told him that she loved him every day.",
    "They are cooking apples.",
    "He fed the dog the bone near the fence.",
    "John told Mark that he had won.",
    "The trophy didn't fit in the bag because it was too big.",
    "The city council refused the demonstrators a permit because they feared violence.",
    "Mary said she would come, but she didn't.",
    "The old man and his son walked to the river. He was exhausted.",
    "Book the flight or the hotel first.",
    "Call me when you arrive or if there's a problem.",
    "Can you pass the salt?",
    "Would you mind closing the window?",
    "Do you know what time it is?",
    "The government plans to raise taxes next year or the opposition will.",
]

CLEAR_INPUTS = [
    "Water boils at 100 degrees Celsius at sea level.",
    "The capital of France is Paris.",
    "The Earth orbits the Sun once per year.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The speed of light in a vacuum is approximately 299792 kilometers per second.",
    "Humans have 46 chromosomes arranged in 23 pairs.",
    "The Amazon River is the largest river in the world by discharge volume.",
    "World War II ended in 1945.",
    "Sodium chloride is the chemical compound commonly known as table salt.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    "To make tea, boil water and steep the tea bag for three minutes.",
    "Press the power button to turn on the device.",
    "Add the flour, then mix until smooth.",
    "The function returns the sum of its two integer arguments.",
    "Click the submit button to send the form.",
    "Open the file, read its contents, then close it.",
    "Sort the array in ascending order before returning it.",
    "Connect the red wire to the positive terminal.",
    "Divide the total by the number of participants to get the average.",
    "Save the document before closing the application.",
    "The red car was parked outside the building.",
    "She opened the door and walked inside.",
    "The meeting starts at 3 PM on Tuesday.",
    "The package weighs exactly 2.5 kilograms.",
    "There are seven days in a week.",
    "The project deadline is the last Friday of the month.",
    "He ran 10 kilometers in 45 minutes.",
    "The temperature dropped below zero overnight.",
    "The report has twelve pages and three appendices.",
    "The train departs at 08:15 from platform 4.",
]


def run_exp001(
    model_name: str = "gpt2",
    k_branches: int = 4,
    n_samples: int = 30,
    warm_start_steps: int = 0,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run EXP-001 or EXP-001b (with warm_start_steps > 0).
    """
    exp_id = "EXP-001b" if warm_start_steps > 0 else "EXP-001"

    if verbose:
        print(f"\n{'='*60}")
        print(f"{exp_id}: QRR χ Discrimination Benchmark")
        print(f"{'='*60}")
        print(f"Model:      {model_name}")
        print(f"Branches:   K={k_branches}")
        print(f"Samples:    {n_samples} ambiguous + {n_samples} clear")
        print(f"Warm-start: {warm_start_steps} steps")
        print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
        print("Loading model (v2.1)...")

    t0 = time.time()
    model = QRRModel(
        base_model_name=model_name,
        k_branches=k_branches,
        freeze_base=True,
    ).to(device)
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"QRR trainable params: {n_params:,}")
        print(f"Model loaded in {time.time()-t0:.1f}s")

    ambiguous = AMBIGUOUS_INPUTS[:n_samples]
    clear     = CLEAR_INPUTS[:n_samples]

    # Optional warm-start: brief supervised training to give router signal
    if warm_start_steps > 0:
        if verbose:
            print(f"\nWarm-starting router for {warm_start_steps} steps...")
        # Use first 20 samples for warm-start, rest for evaluation
        ws_amb = ambiguous[:20]
        ws_clr = clear[:20]
        model.warm_start_router(ws_amb, ws_clr, steps=warm_start_steps, lr=1e-3)
        # Evaluate on held-out samples
        eval_amb = ambiguous[20:] if n_samples > 20 else ambiguous
        eval_clr = clear[20:]     if n_samples > 20 else clear
    else:
        eval_amb = ambiguous
        eval_clr = clear

    if verbose:
        print(f"\nComputing χ for {len(eval_amb)} ambiguous inputs...")
    chi_ambiguous, chi_clear = [], []
    diversity_ambiguous, diversity_clear = [], []

    model.eval()
    for i, prompt in enumerate(eval_amb):
        with torch.no_grad():
            out = model.forward_with_branches(prompt)
        chi_ambiguous.append(out["chi"])
        diversity_ambiguous.append(out["branch_diversity"])
        if verbose and (i % 5 == 0 or i == len(eval_amb)-1):
            print(f"  [{i+1:02d}/{len(eval_amb)}] χ={out['chi']:.3f}  '{prompt[:55]}'")

    if verbose:
        print(f"\nComputing χ for {len(eval_clr)} clear inputs...")
    for i, prompt in enumerate(eval_clr):
        with torch.no_grad():
            out = model.forward_with_branches(prompt)
        chi_clear.append(out["chi"])
        diversity_clear.append(out["branch_diversity"])
        if verbose and (i % 5 == 0 or i == len(eval_clr)-1):
            print(f"  [{i+1:02d}/{len(eval_clr)}] χ={out['chi']:.3f}  '{prompt[:55]}'")

    chi_a = np.array(chi_ambiguous)
    chi_c = np.array(chi_clear)
    delta_chi = float(chi_a.mean() - chi_c.mean())

    if len(chi_a) > 1 and len(chi_c) > 1:
        u_stat, p_value = stats.mannwhitneyu(chi_a, chi_c, alternative="greater")
    else:
        u_stat, p_value = 0.0, 1.0

    success = delta_chi > 0.15 and p_value < 0.05

    results = {
        "experiment":            exp_id,
        "model":                 model_name,
        "k_branches":            k_branches,
        "n_samples":             n_samples,
        "warm_start_steps":      warm_start_steps,
        "chi_ambiguous_mean":    float(chi_a.mean()),
        "chi_ambiguous_std":     float(chi_a.std()),
        "chi_clear_mean":        float(chi_c.mean()),
        "chi_clear_std":         float(chi_c.std()),
        "delta_chi":             delta_chi,
        "mann_whitney_u":        float(u_stat),
        "p_value":               float(p_value),
        "success_criterion":     "> 0.15 delta_chi AND p < 0.05",
        "success":               bool(success),
        "diversity_ambiguous":   float(np.mean(diversity_ambiguous)),
        "diversity_clear":       float(np.mean(diversity_clear)),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS — {exp_id}")
        print(f"{'='*60}")
        print(f"  χ ambiguous:  {results['chi_ambiguous_mean']:.4f} ± {results['chi_ambiguous_std']:.4f}")
        print(f"  χ clear:      {results['chi_clear_mean']:.4f} ± {results['chi_clear_std']:.4f}")
        print(f"  Δχ:           {delta_chi:+.4f}  (need > +0.15)")
        print(f"  p-value:      {p_value:.4f}  (need < 0.05)")
        print(f"  Diversity Δ:  {np.mean(diversity_ambiguous) - np.mean(diversity_clear):+.4f}")
        print()
        if success:
            print("  ✅ SUCCESS — χ discriminates ambiguous from clear.")
            print("     → Proceed to Phase 2: UnitaryMixer + complex phases.")
        elif delta_chi > 0.05:
            print("  ⚠️  PARTIAL — signal present but below threshold.")
            print("     → Try --warm-start 100 or --branches 8.")
        else:
            print("  ❌ FAILED — no discrimination signal.")
            print("     → See CRITIQUE.md for next steps.")
        print(f"{'='*60}\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="gpt2")
    parser.add_argument("--branches",   type=int, default=4)
    parser.add_argument("--samples",    type=int, default=30)
    parser.add_argument("--warm-start", type=int, default=0,
                        help="Steps of supervised warm-start (0=zero-shot)")
    parser.add_argument("--output",     default="experiments/EXP-001b_results.json")
    args = parser.parse_args()

    run_exp001(
        model_name=args.model,
        k_branches=args.branches,
        n_samples=args.samples,
        warm_start_steps=args.warm_start,
        output_path=args.output,
        verbose=True,
    )
