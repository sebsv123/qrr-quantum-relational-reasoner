# Experiment Log

All experiments are tracked here with their hypothesis, setup, results, and conclusions.

---

## EXP-001 — χ Discrimination Baseline

**Status**: 🔄 In progress  
**Notebook**: `notebooks/EXP-001_chi_discrimination.ipynb`  
**Hypothesis**: QRR's collapse index χ is significantly higher on ambiguous inputs than on clear ones, even without fine-tuning.  
**Success criterion**: Δχ = Mean χ(ambiguous) − Mean χ(clear) > 0.15  
**Model**: GPT-2 base  
**Dataset**: AmbigQA-100 subset + hand-curated clear statements  
**Results**: *(pending)*

---

## EXP-002 — Unitary Mixer Stability

**Status**: ⏳ Planned  
**Hypothesis**: A unitary-constrained branch mixer produces more stable amplitude norms across 100 inference steps than a fully free linear mixer.  
**Success criterion**: Norm drift < 5% over 100 steps with unitary constraint vs. > 15% without.  
**Model**: QRR with UnitaryMixer vs. LinearMixer ablation  

---

## EXP-003 — Decoherence Timing vs. Task Accuracy

**Status**: ⏳ Planned  
**Hypothesis**: Later decoherence (higher χ threshold) improves accuracy on multi-hop QA but degrades latency.  
**Success criterion**: Pareto improvement vs. beam search at iso-latency.  

---

## Log Format

When adding a new experiment:
1. Assign ID: EXP-NNN
2. State hypothesis as a falsifiable claim
3. Define numerical success criterion before running
4. Record all results, including negative ones
5. Link notebook and output files
