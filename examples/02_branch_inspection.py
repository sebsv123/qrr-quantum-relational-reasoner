"""Example 02: Visualize how branches diverge and merge on ambiguous input.

Run:
    python examples/02_branch_inspection.py
"""

import torch
from qrr.qrr_model import QRRModel
from qrr.collapse_index import CollapseIndex

model = QRRModel(base_model_name="gpt2", k_branches=4)
model.eval()

AMBIGUOUS_PROMPTS = [
    "The bank was steep.",          # bank = river bank OR financial bank
    "I saw the man with the telescope.",  # attachment ambiguity
    "Book the flight or the hotel first.",  # sequence ambiguity
]

CLEAR_PROMPTS = [
    "Water boils at 100 degrees Celsius.",
    "The capital of France is Paris.",
]

print("\n" + "=" * 60)
print("QRR Branch Divergence Analysis")
print("=" * 60)

for label, prompts in [("AMBIGUOUS", AMBIGUOUS_PROMPTS), ("CLEAR", CLEAR_PROMPTS)]:
    print(f"\n[{label} inputs]")
    for prompt in prompts:
        with torch.no_grad():
            output = model.forward_with_branches(prompt)
        chi = output['chi']
        diversity = output['branch_diversity']  # std of branch hidden states
        print(f"  '{prompt[:45]}...'" if len(prompt) > 45 else f"  '{prompt}'")
        print(f"    χ={chi:.3f}  diversity={diversity:.4f}  {'⚡ collapsed' if chi < 0.3 else '🌊 superposed'}")

print("\n✓ Hypothesis: ambiguous inputs should yield higher χ and diversity.")
