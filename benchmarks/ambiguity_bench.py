"""
Ambiguity Benchmark — EXP-001.

Tests whether QRR's collapse index χ discriminates semantically ambiguous
inputs from clear ones, without any fine-tuning.

Hypothesis: Mean χ(ambiguous) − Mean χ(clear) > 0.15
Success criterion: Δχ > 0.15, p < 0.05 (Mann-Whitney U)

Usage:
    python benchmarks/ambiguity_bench.py
    python benchmarks/ambiguity_bench.py --model gpt2 --branches 4 --output experiments/EXP-001_results.json
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
# Built-in dataset (no external downloads required for first run)
# ---------------------------------------------------------------------------

AMBIGUOUS_INPUTS = [
    # Lexical ambiguity
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
    # Structural ambiguity
    "I saw the man with the telescope.",
    "The chicken is ready to eat.",
    "Visiting relatives can be boring.",
    "The professor urged the students to study harder in her office.",
    "The horse raced past the barn fell.",
    "I know a taller man than John.",
    "She told him that she loved him every day.",
    "They are cooking apples.",
    "The government plans to raise taxes next year or the opposition will.",
    "He fed the dog the bone near the fence.",
    # Referential ambiguity
    "John told Mark that he had won.",
    "The trophy didn't fit in the bag because it was too big.",
    "The city council refused the demonstrators a permit because they feared violence.",
    "Mary said she would come, but she didn't.",
    "The old man and his son walked to the river. He was exhausted.",
    # Intent ambiguity
    "Book the flight or the hotel first.",
    "Call me when you arrive or if there's a problem.",
    "Can you pass the salt?",
    "Would you mind closing the window?",
    "Do you know what time it is?",
]

CLEAR_INPUTS = [
    # Unambiguous factual statements
    "Water boils at 100 degrees Celsius at sea level.",
    "The capital of France is Paris.",
    "The Earth orbits the Sun once per year.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Humans have 46 chromosomes arranged in 23 pairs.",
    "The Amazon River is the largest river in the world by discharge volume.",
    "World War II ended in 1945.",
    "Sodium chloride is the chemical compound commonly known as table salt.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    # Clear procedural statements
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
    # Clear descriptive statements
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


def load_ambiguity_dataset(n: int = 30) -> tuple[list[str], list[str]]:
    """Return up to n ambiguous and n clear samples."""
    return AMBIGUOUS_INPUTS[:n], CLEAR_INPUTS[:n]


def run_exp001(
    model_name: str = "gpt2",
    k_branches: int = 4,
    n_samples: int = 30,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run EXP-001: χ discrimination benchmark.

    Returns results dict with all metrics.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("EXP-001: QRR χ Discrimination Benchmark")
        print(f"{'='*60}")
        print(f"Model:    {model_name}")
        print(f"Branches: K={k_branches}")
        print(f"Samples:  {n_samples} ambiguous + {n_samples} clear")
        print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
        print("Loading model...")

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
        print()

    ambiguous, clear = load_ambiguity_dataset(n_samples)
    chi_ambiguous, chi_clear = [], []
    diversity_ambiguous, diversity_clear = [], []

    if verbose:
        print("Computing χ for ambiguous inputs...")
    for i, prompt in enumerate(ambiguous):
        with torch.no_grad():
            out = model.forward_with_branches(prompt)
        chi_ambiguous.append(out["chi"])
        diversity_ambiguous.append(out["branch_diversity"])
        if verbose and i % 10 == 0:
            print(f"  [{i+1:02d}/{n_samples}] '{prompt[:50]}...' χ={out['chi']:.3f}")

    if verbose:
        print("\nComputing χ for clear inputs...")
    for i, prompt in enumerate(clear):
        with torch.no_grad():
            out = model.forward_with_branches(prompt)
        chi_clear.append(out["chi"])
        diversity_clear.append(out["branch_diversity"])
        if verbose and i % 10 == 0:
            print(f"  [{i+1:02d}/{n_samples}] '{prompt[:50]}...' χ={out['chi']:.3f}")

    # Statistical analysis
    chi_a = np.array(chi_ambiguous)
    chi_c = np.array(chi_clear)
    delta_chi = chi_a.mean() - chi_c.mean()
    u_stat, p_value = stats.mannwhitneyu(chi_a, chi_c, alternative="greater")
    success = delta_chi > 0.15 and p_value < 0.05

    results = {
        "experiment": "EXP-001",
        "model": model_name,
        "k_branches": k_branches,
        "n_samples": n_samples,
        "chi_ambiguous_mean": float(chi_a.mean()),
        "chi_ambiguous_std":  float(chi_a.std()),
        "chi_clear_mean":     float(chi_c.mean()),
        "chi_clear_std":      float(chi_c.std()),
        "delta_chi":          float(delta_chi),
        "mann_whitney_u":     float(u_stat),
        "p_value":            float(p_value),
        "success_criterion":  "> 0.15 delta_chi AND p < 0.05",
        "success":            bool(success),
        "diversity_ambiguous_mean": float(np.mean(diversity_ambiguous)),
        "diversity_clear_mean":     float(np.mean(diversity_clear)),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  χ ambiguous:  {results['chi_ambiguous_mean']:.4f} ± {results['chi_ambiguous_std']:.4f}")
        print(f"  χ clear:      {results['chi_clear_mean']:.4f} ± {results['chi_clear_std']:.4f}")
        print(f"  Δχ:           {delta_chi:.4f}  (need > 0.15)")
        print(f"  p-value:      {p_value:.4f}  (need < 0.05)")
        print(f"  Diversity Δ:  {np.mean(diversity_ambiguous) - np.mean(diversity_clear):.4f}")
        print()
        if success:
            print("  ✅ SUCCESS — χ discriminates ambiguous from clear inputs.")
            print("     → Proceed to Phase 2: UnitaryMixer + complex phases.")
        elif delta_chi > 0.05:
            print("  ⚠️  PARTIAL — Δχ > 0.05 but criterion not fully met.")
            print("     → Try K=8 branches or adjust amplitude router.")
        else:
            print("  ❌ FAILED — χ shows no discrimination signal.")
            print("     → Revisit branch initialization and router design.")
        print(f"{'='*60}\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-001: QRR χ Discrimination Benchmark")
    parser.add_argument("--model",   default="gpt2",  help="HuggingFace model name")
    parser.add_argument("--branches", type=int, default=4, help="Number of QRR branches K")
    parser.add_argument("--samples",  type=int, default=30, help="Samples per category")
    parser.add_argument("--output",  default="experiments/EXP-001_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_exp001(
        model_name=args.model,
        k_branches=args.branches,
        n_samples=args.samples,
        output_path=args.output,
        verbose=True,
    )
