"""
QRR Quickstart — 5 minutes to first χ score.

This example shows the three most common QRR use cases:
  1. Compute χ for a single sentence
  2. Compare χ between ambiguous vs. clear inputs
  3. Inspect branch states and diversity

Run:
    pip install -e .
    python examples/quickstart.py
"""

import torch
from qrr.qrr_model import QRRModel

def main():
    print("QRR Quickstart\n" + "="*50)

    # Load model (downloads GPT-2 on first run ~500MB)
    print("Loading QRR model (GPT-2 base, K=4 branches)...")
    model = QRRModel(base_model_name="gpt2", k_branches=4, freeze_base=True)
    model.eval()
    print("Model ready.\n")

    # --- 1. Single sentence χ score ---
    print("1. Single sentence χ score")
    print("-"*40)
    sentence = "The bank was steep and covered in moss."
    chi = model.compute_chi(sentence)
    print(f"   Input: '{sentence}'")
    print(f"   χ = {chi:.4f}  (χ → 1 = ambiguous, χ → 0 = clear)\n")

    # --- 2. Ambiguous vs. clear comparison ---
    print("2. Ambiguous vs. clear comparison")
    print("-"*40)
    pairs = [
        ("I saw the bat in the cave.",             "Lexical: bat (animal/sport)"),
        ("The chicken is ready to eat.",            "Structural: who eats whom?"),
        ("John told Mark that he had won.",         "Referential: who won?"),
        ("Water boils at 100 degrees Celsius.",     "Clear factual statement"),
        ("The Earth orbits the Sun once per year.", "Clear factual statement"),
        ("Press the power button to turn on.",      "Clear procedural statement"),
    ]
    with torch.no_grad():
        for text, label in pairs:
            chi = model.compute_chi(text)
            indicator = "↑ ambiguous" if chi > 0.4 else "↓ clear"
            print(f"   [{label}]")
            print(f"   χ={chi:.4f} {indicator}")
            print(f"   '{text[:60]}'\n")

    # --- 3. Branch state inspection ---
    print("3. Branch state inspection")
    print("-"*40)
    ambiguous_text = "Can you pass the salt?"
    with torch.no_grad():
        out = model.forward_with_branches(ambiguous_text)

    state = out["branch_state"]
    probs = state.probabilities[0]  # (K,)
    print(f"   Input: '{ambiguous_text}'")
    print(f"   χ = {out['chi']:.4f}")
    print(f"   Branch diversity = {out['branch_diversity']:.4f}")
    print(f"   Branch probabilities (Born rule):")
    for k, p in enumerate(probs.tolist()):
        bar = "█" * int(p * 40)
        print(f"     Branch {k}: {p:.4f}  {bar}")
    print()
    print("Done! See benchmarks/ambiguity_bench.py to run EXP-001.")


if __name__ == "__main__":
    main()
