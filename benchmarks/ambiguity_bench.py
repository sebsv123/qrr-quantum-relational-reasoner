"""
Ambiguity Benchmark (EXP-001).

Measures whether QRR's collapse index χ discriminates between
ambiguous and unambiguous inputs without any fine-tuning.

Usage:
    python benchmarks/ambiguity_bench.py --model gpt2 --branches 4

Success criterion (from ROADMAP.md):
    Δχ = mean(χ_ambiguous) - mean(χ_clear) > 0.15, p < 0.05
"""

from __future__ import annotations
import argparse
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from qrr.qrr_model import QRRModel


# --- Curated dataset ---
# 30 ambiguous + 30 clear sentences for quick local testing.
# For the full EXP-001 run, replace with AmbigQA dataset loader.

AMBIGUOUS_SENTENCES = [
    # Lexical ambiguity
    "The bank was steep.",
    "I went to the bank yesterday.",
    "He saw her duck.",
    "She can't bear children.",
    "The chicken is ready to eat.",
    "Time flies like an arrow.",
    "I saw the man with the binoculars.",
    "Visiting relatives can be annoying.",
    "The professor said on Monday he would give the exam.",
    "I need more time to understand the problem.",
    # Structural ambiguity
    "I saw the man on the hill with a telescope.",
    "The horse raced past the barn fell.",
    "Book the flight or the hotel first.",
    "The old men and women left the room.",
    "Police police police police police.",
    # Referential ambiguity
    "John told Peter that he had won.",
    "The committee approved the proposal after its review.",
    "Mary said she was tired, and Susan agreed.",
    "Call me a cab and I'll be happy.",
    "I know more beautiful cities than Paris.",
    # Pragmatic ambiguity
    "Can you open the window?",
    "Do you have the time?",
    "It's cold in here.",
    "Would you like some coffee?",
    "Could you pass the salt?",
    # Multi-sense words
    "The crane was at the construction site.",
    "She hit the nail on the head.",
    "The light was too bright.",
    "He left in a rush.",
    "The pool was full.",
]

CLEAR_SENTENCES = [
    # Unambiguous factual statements
    "Water boils at 100 degrees Celsius at sea level.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a programming language created in 1991.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "The human body has 206 bones.",
    "Carbon dioxide is composed of one carbon and two oxygen atoms.",
    "The Earth revolves around the Sun once per year.",
    "Shakespeare wrote Hamlet in approximately 1600.",
    "The boiling point of water decreases at higher altitudes.",
    "DNA stands for deoxyribonucleic acid.",
    "The Amazon river is the largest river by discharge volume.",
    "Photosynthesis converts sunlight into glucose.",
    "The Pythagorean theorem states that a² + b² = c².",
    "Mammals are warm-blooded vertebrates.",
    "The capital of Japan is Tokyo.",
    "Hydrogen is the lightest element on the periodic table.",
    "The Great Wall of China is over 13,000 miles long.",
    "Gravity causes objects to fall toward the Earth.",
    "The mitochondria is the powerhouse of the cell.",
    "The speed of sound in air is approximately 343 meters per second.",
    "Rome was not built in a day.",
    "The moon orbits the Earth every 27.3 days.",
    "Penguins are flightless birds native to the Southern Hemisphere.",
    "The currency of Japan is the yen.",
    "Albert Einstein published the special theory of relativity in 1905.",
    "The Sahara is the largest hot desert on Earth.",
    "Oxygen has atomic number 8 on the periodic table.",
    "The brain is the most complex organ in the human body.",
    "TCP/IP is the foundational protocol of the internet.",
    "The binary system uses only the digits 0 and 1.",
]


def load_ambiguity_dataset(n: int = 30) -> tuple[list[str], list[str]]:
    """Return up to n ambiguous and n clear sentences."""
    return AMBIGUOUS_SENTENCES[:n], CLEAR_SENTENCES[:n]


def run_benchmark(model: QRRModel, n: int = 30) -> dict:
    ambiguous, clear = load_ambiguity_dataset(n)
    chi_amb, chi_clr = [], []

    model.eval()
    with torch.no_grad():
        for text in tqdm(ambiguous, desc="Ambiguous"):
            r = model.forward_with_branches(text)
            chi_amb.append(r["chi"])
        for text in tqdm(clear, desc="Clear"):
            r = model.forward_with_branches(text)
            chi_clr.append(r["chi"])

    chi_amb = np.array(chi_amb)
    chi_clr = np.array(chi_clr)
    delta_chi = chi_amb.mean() - chi_clr.mean()
    _, p_value = stats.mannwhitneyu(chi_amb, chi_clr, alternative="greater")

    results = {
        "chi_ambiguous_mean": float(chi_amb.mean()),
        "chi_ambiguous_std":  float(chi_amb.std()),
        "chi_clear_mean":     float(chi_clr.mean()),
        "chi_clear_std":      float(chi_clr.std()),
        "delta_chi":          float(delta_chi),
        "p_value":            float(p_value),
        "success":            bool(delta_chi > 0.15 and p_value < 0.05),
        "n_ambiguous":        len(chi_amb),
        "n_clear":            len(chi_clr),
    }
    return results


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EXP-001: χ Discrimination Benchmark")
    print("=" * 60)
    print(f"  χ ambiguous : mean={results['chi_ambiguous_mean']:.3f}  std={results['chi_ambiguous_std']:.3f}")
    print(f"  χ clear     : mean={results['chi_clear_mean']:.3f}  std={results['chi_clear_std']:.3f}")
    print(f"  Δχ          : {results['delta_chi']:+.3f}")
    print(f"  p-value     : {results['p_value']:.4f}")
    print(f"  n           : {results['n_ambiguous']} ambiguous / {results['n_clear']} clear")
    print("-" * 60)
    if results["success"]:
        print("  ✅ SUCCESS: Δχ > 0.15 and p < 0.05")
        print("     χ discriminates ambiguous vs clear inputs.")
        print("     Proceed to Phase 2 (UnitaryMixer + complex phases).")
    elif results["delta_chi"] > 0.05:
        print("  ⚠️  PARTIAL: Δχ > 0.05 but below 0.15 threshold.")
        print("     Signal is present but weak. Tune K, χ_threshold, or init.")
    else:
        print("  ❌ FAIL: Δχ ≤ 0.05. χ carries no discriminative signal.")
        print("     Revisit amplitude router design (see CRITIQUE.md).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EXP-001: χ Discrimination Benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--n", type=int, default=30, help="Samples per class")
    parser.add_argument("--chi_threshold", type=float, default=0.3)
    parser.add_argument("--save_json", type=str, default="", help="Path to save results JSON")
    args = parser.parse_args()

    print(f"Loading QRR [{args.model}, K={args.branches}]...")
    model = QRRModel(
        base_model_name=args.model,
        k_branches=args.branches,
        chi_threshold=args.chi_threshold,
    )

    results = run_benchmark(model, n=args.n)
    print_results(results)

    if args.save_json:
        import json
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
