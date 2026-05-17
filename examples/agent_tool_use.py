"""
QRR Agent Tool Use Example.

Demonstrates how to use QRR's collapse index to decide WHEN to commit
to a tool call vs. seek disambiguation first.

The pattern:
  1. Receive ambiguous user request
  2. Compute χ — if high, ask clarifying question first
  3. Once χ drops (evidence accumulated), execute tool call

This is the core value proposition of QRR for agentic systems:
  "Don't call the tool until you're sure which tool to call."

Run:
    python examples/agent_tool_use.py
"""

from __future__ import annotations
import torch
from qrr.qrr_model import QRRModel

# Simulated tool registry
TOOLS = {
    "book_meeting":   lambda ctx: f"Meeting booked: {ctx}",
    "send_email":     lambda ctx: f"Email sent to: {ctx}",
    "search_calendar":lambda ctx: f"Calendar searched for: {ctx}",
    "clarify":        lambda ctx: f"Clarification requested: {ctx}",
}


def route_with_chi(
    model: QRRModel,
    user_input: str,
    context: str = "",
    chi_threshold: float = 0.35,
) -> dict:
    """
    Route a user request to a tool, gated by collapse index χ.

    High χ → ambiguous → request clarification first.
    Low χ  → clear     → commit to action.
    """
    full_input = f"{context} {user_input}".strip() if context else user_input

    with torch.no_grad():
        out = model.forward_with_branches(full_input)

    chi = out["chi"]
    diversity = out["branch_diversity"]

    if chi > chi_threshold:
        action = "clarify"
        result = TOOLS["clarify"](f"Request is ambiguous (chi={chi:.3f}): '{user_input}'")
    else:
        # Simplified routing based on keywords
        text = user_input.lower()
        if "meet" in text or "schedule" in text or "calendar" in text:
            action = "book_meeting"
        elif "email" in text or "send" in text or "message" in text:
            action = "send_email"
        else:
            action = "search_calendar"
        result = TOOLS[action](user_input)

    return {
        "input":     user_input,
        "chi":       chi,
        "diversity": diversity,
        "action":    action,
        "result":    result,
        "committed": action != "clarify",
    }


def main():
    print("QRR Agent Tool-Use Demo\n" + "="*50)
    print("Demonstrating χ-gated tool commitment\n")

    model = QRRModel(base_model_name="gpt2", k_branches=4, freeze_base=True)
    model.eval()

    test_cases = [
        # Ambiguous — should request clarification
        ("Book it for Tuesday",         "Should delay: 'it' is unresolved"),
        ("Send it to John",             "Should delay: 'it' and 'which John' unresolved"),
        ("Schedule the meeting",        "Ambiguous: which meeting, when, who?"),
        # Clear — should commit
        ("Book a meeting with Alice on Friday at 3pm",  "Should commit to book_meeting"),
        ("Send an email to bob@example.com about Q3",   "Should commit to send_email"),
        ("Check my calendar for next Monday",           "Should commit to search_calendar"),
    ]

    for request, expected in test_cases:
        result = route_with_chi(model, request, chi_threshold=0.35)
        status = "✅ COMMITTED" if result["committed"] else "⚠️  CLARIFY"
        print(f"{status} | χ={result['chi']:.3f} | {result['action']}")
        print(f"         Input:    '{request}'")
        print(f"         Expected: {expected}")
        print(f"         Result:   {result['result']}")
        print()

    print("\nNote: routing accuracy improves after fine-tuning (Phase 3).")
    print("This demo uses zero-shot QRR on frozen GPT-2.")


if __name__ == "__main__":
    main()
