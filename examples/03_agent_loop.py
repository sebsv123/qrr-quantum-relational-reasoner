"""Example 03: QRR-powered agent that defers tool calls until χ is low.

Demonstrates the core agent use case: maintain branch superposition over
possible actions, and only call a tool when the model is confident enough.

Run:
    python examples/03_agent_loop.py
"""

import torch
from qrr.qrr_model import QRRModel

# Toy tool registry
TOOLS = {
    "search_calendar": lambda q: f"[Calendar] No events found for '{q}'",
    "lookup_contact": lambda n: f"[Contact] {n}: john.doe@company.com",
    "send_email": lambda to, subj: f"[Email] Sent to {to} — subject: {subj}",
}

CHI_ACT_THRESHOLD = 0.2   # only act when χ < this
CHI_ASK_THRESHOLD = 0.5   # ask for clarification when χ > this for N steps

model = QRRModel(base_model_name="gpt2", k_branches=4, chi_threshold=CHI_ACT_THRESHOLD)
model.eval()

print("\n" + "=" * 60)
print("QRR Agent — Deferred Action Loop")
print("=" * 60)

user_request = "Schedule a meeting with John from sales next Tuesday at 3pm."
print(f"\nUser: {user_request}\n")

steps = []
for step in range(5):
    with torch.no_grad():
        state = model.agent_step(user_request, history=steps)

    chi = state['chi']
    branches = state['top_branches']  # list of (hypothesis_str, weight)

    print(f"Step {step+1} | χ={chi:.3f}")
    for i, (hyp, w) in enumerate(branches[:3]):
        print(f"  Branch {i}: [{w:.2f}] {hyp}")

    if chi < CHI_ACT_THRESHOLD:
        action = state['selected_action']
        print(f"  ⚡ χ below threshold — executing: {action['tool']}({action['args']})")
        result = TOOLS[action['tool']](**action['args'])
        print(f"  ✓ Tool result: {result}")
        steps.append({"action": action, "result": result})
        break
    elif chi > CHI_ASK_THRESHOLD and step >= 2:
        print(f"  ❓ High ambiguity sustained — requesting clarification")
        print(f"  Agent: 'Which John? John from sales or John from engineering?'")
        break
    else:
        print(f"  🌊 Maintaining superposition — gathering more context...")
        steps.append({"observation": state['best_observation']})

print("\nDone.")
