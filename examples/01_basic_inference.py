"""Example 01: Basic QRR inference with branch inspection.

Run:
    python examples/01_basic_inference.py
"""

import torch
from qrr.qrr_model import QRRModel

# --- Config ---
BASE_MODEL = "gpt2"   # swap for any HuggingFace causal LM
K_BRANCHES = 4
CHI_THRESHOLD = 0.3

# --- Load model ---
model = QRRModel(
    base_model_name=BASE_MODEL,
    k_branches=K_BRANCHES,
    chi_threshold=CHI_THRESHOLD,
)
model.eval()

# --- Ambiguous prompt ---
prompt = "Book a meeting with John on Tuesday."
print(f"\nPrompt: {prompt}")
print("-" * 50)

# --- Inference ---
with torch.no_grad():
    output = model.generate(
        prompt,
        max_new_tokens=50,
        return_branch_states=True,
    )

# --- Results ---
print(f"Generated: {output['text']}")
print(f"\nCollapse Index χ trace:")
for t, chi in enumerate(output['chi_trace']):
    bar = "█" * int(chi * 20) + "░" * (20 - int(chi * 20))
    status = "⚡ COLLAPSED" if chi < CHI_THRESHOLD else "   branching"
    print(f"  t={t:02d}  [{bar}]  χ={chi:.3f}  {status}")

print(f"\nFinal branch weights:")
for k, w in enumerate(output['final_branch_weights']):
    print(f"  Branch {k}: {w:.4f}")
