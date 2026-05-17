# Experiment Log

All QRR experiments are tracked here with hypothesis, setup, results, and conclusions.  
Every experiment has a falsifiable hypothesis and a numerical success criterion defined **before** running.

---

## EXP-001 — χ Discrimination Baseline

**Status**: 🔄 In progress  
**Period**: June 2026  
**Notebook**: `notebooks/EXP-001_chi_discrimination.ipynb`  
**Hypothesis**: QRR's collapse index χ is significantly higher on semantically ambiguous inputs than on clear ones, even without fine-tuning.  
**Success criterion**: Δχ = Mean χ(ambiguous) − Mean χ(clear) > 0.15, p < 0.05  
**Model**: GPT-2 base (117M params, no fine-tuning)  
**Dataset**: AmbigQA-100 subset + 100 hand-curated clear factual statements  
**Metrics**: Mean χ, std χ, Δχ, Mann-Whitney U p-value  
**Results**: *(pending first run)*  
**Conclusion**: *(pending)*

---

## EXP-002 — Unitary Mixer Stability

**Status**: ⏳ Planned (Phase 1, July–August 2026)  
**Hypothesis**: A Cayley-parametrized unitary constraint on UnitaryMixer keeps amplitude norm drift < 5% over 100 inference steps, vs > 15% for a free linear mixer.  
**Success criterion**: ‖α_100‖ / ‖α_0‖ ∈ [0.95, 1.05] with constraint; < 0.85 or > 1.15 without  
**Model**: QRR-GPT2 with UnitaryMixer vs. LinearMixer ablation

---

## EXP-003 — Decoherence Timing vs. Accuracy

**Status**: ⏳ Planned (Phase 2, September–November 2026)  
**Hypothesis**: Higher χ threshold (later collapse) improves multi-hop QA accuracy at the cost of increased latency.  
**Success criterion**: Pareto improvement over beam search at iso-latency on HotpotQA dev set  
**Model**: Fine-tuned QRR-GPT2 (from Phase 3 prep)

---

## EXP-004 — Branch Diversity vs. Semantic Entropy

**Status**: ⏳ Planned (Phase 2, November 2026)  
**Hypothesis**: Branch hidden-state diversity (std across K branches) correlates positively with semantic entropy of the input (Spearman ρ > 0.4).  
**Dataset**: SemEval ambiguity tasks + WiC (Word-in-Context)

---

## Log Format

When adding a new experiment:
1. Assign ID sequentially: EXP-NNN
2. State hypothesis as a **falsifiable claim with a number**
3. Define numerical success criterion **before** running
4. Record all results — including negative ones
5. Link notebook and output files in `experiments/`
