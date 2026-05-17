# QRR — Frequently Asked Questions

---

## Is this actually quantum computing?

No. QRR is a classical neural network architecture. The word "quantum" refers to the
conceptual framework borrowed from quantum mechanics — superposition, interference,
and decoherence — not to quantum hardware or qubits.

The complex-valued amplitudes run on standard CPUs and GPUs using PyTorch.

---

## Why use complex amplitudes instead of just probabilities?

Complex amplitudes allow interference: two branches with opposite phases can cancel
(destructive interference), while branches pointing in the same direction amplify
(constructive interference). This is richer than simply mixing probabilities, which
can only blend, not cancel.

Stage 1 uses real amplitudes (simpler, more stable). Stage 2 introduces full complex
phases once the model demonstrates branch diversity in EXP-001.

---

## What's the difference between QRR and Mixture of Experts (MoE)?

| Feature | MoE | QRR |
|---|---|---|
| Branch selection | Hard (top-k) | Soft (weighted by Born-rule probabilities) |
| Amplitude type | Real weights | Complex amplitudes |
| Interference | None | Constructive + destructive |
| Collapse timing | Per token | Deferred until χ < threshold |
| Goal | Efficiency | Ambiguity handling |

---

## What's the difference between QRR and beam search?

Beam search maintains multiple hypotheses at the *output token* level.
QRR maintains multiple hypotheses at the *hidden state / semantic* level.
QRR collapses based on semantic confidence (χ), not just likelihood.

---

## Does QRR require special hardware?

No. It runs on standard GPU or CPU with PyTorch >= 2.0. A GPU is recommended
for training (even a single RTX 3080 is sufficient for GPT-2 Stage 1).

---

## How do I pick K (number of branches)?

Start with K=4. The Phase 4 ablation will test K ∈ {1, 2, 4, 8} against accuracy
and latency. K=1 is equivalent to standard LM inference (no branching). K=8
increases expressiveness but adds compute proportionally.

---

## Why GPT-2 for EXP-001?

GPT-2 is small (124M), fast to run locally, and widely understood. It lets us
test whether χ carries information before committing to larger models.
Phases 3-4 will test on larger models (GPT-2 XL, Llama 3).

---

## The experiment failed (Δχ ≤ 0.05). What now?

See `ROADMAP.md` → Falsification Policy. The steps are:
1. Check `branch_diversity`: if it's near zero, branches aren't diverging — adjust the initialization perturbation scale in `BranchBank`.
2. Try K=8 branches.
3. Try a stronger base model (GPT-2 medium or large).
4. If all fail, revisit the `AmplitudeRouter` design — the compatibility scorer may not be sensitive enough to semantic content with a frozen base.

---

## Can I fine-tune the base model too (unfreeze it)?

Yes: set `freeze_base=False` in `QRRModel`. This increases compute and risk of
catastrophic forgetting but may improve EXP-001 results if the frozen base
produces insufficient signal. Recommended only after Stage 1 training stabilizes.
