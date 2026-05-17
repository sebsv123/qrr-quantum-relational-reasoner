"""
Ambiguity Benchmark for QRR.

Tests whether QRR maintains appropriate uncertainty
on structurally ambiguous inputs rather than committing prematurely.

Metrics:
  - Accuracy on disambiguated queries
  - ECE (Expected Calibration Error)
  - χ distribution on ambiguous vs. unambiguous inputs
  - Premature collapse rate (collapses before sufficient evidence)
"""

import torch
from typing import List, Dict


AMBIGUOUS_EXAMPLES = [
    {
        "query": "Book a meeting with Juan on Tuesday.",
        "interpretations": [
            "Juan from sales team",
            "Juan from tech team",
        ],
        "ambiguous": True,
        "resolving_context": "Last meeting was with Juan from sales about Q2 targets.",
    },
    {
        "query": "The bank was steep.",
        "interpretations": [
            "river bank",
            "financial bank",
        ],
        "ambiguous": True,
        "resolving_context": "We were hiking near the river.",
    },
    {
        "query": "What is the capital of France?",
        "interpretations": ["Paris"],
        "ambiguous": False,
        "resolving_context": None,
    },
]


def evaluate_ambiguity(
    model,
    examples: List[Dict],
    device: str = "cpu",
) -> Dict:
    """
    Evaluate QRR on ambiguity benchmark.

    Returns:
        results dict with:
          - mean_chi_ambiguous: should be HIGH (maintaining uncertainty)
          - mean_chi_unambiguous: should be LOW (collapsed)
          - chi_discrimination: difference between the two (higher is better)
          - premature_collapse_rate: fraction of ambiguous inputs where χ < 0.3
    """
    results = {
        "chi_ambiguous": [],
        "chi_unambiguous": [],
        "premature_collapses": 0,
        "total_ambiguous": 0,
    }

    model.eval()
    with torch.no_grad():
        for ex in examples:
            # Tokenize and run (placeholder — adapt to your tokenizer)
            # output = model(input_ids=..., return_branch_map=True)
            # chi = output["chi"].item()

            # Placeholder
            chi = torch.rand(1).item()

            if ex["ambiguous"]:
                results["chi_ambiguous"].append(chi)
                results["total_ambiguous"] += 1
                if chi < 0.3:
                    results["premature_collapses"] += 1
            else:
                results["chi_unambiguous"].append(chi)

    mean_amb = sum(results["chi_ambiguous"]) / len(results["chi_ambiguous"]) if results["chi_ambiguous"] else 0
    mean_unamb = sum(results["chi_unambiguous"]) / len(results["chi_unambiguous"]) if results["chi_unambiguous"] else 0

    return {
        "mean_chi_ambiguous": mean_amb,
        "mean_chi_unambiguous": mean_unamb,
        "chi_discrimination": mean_amb - mean_unamb,
        "premature_collapse_rate": results["premature_collapses"] / max(results["total_ambiguous"], 1),
    }
