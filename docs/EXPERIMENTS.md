# QRR Experiment Log

Living document. Updated after each experiment run.
All raw results stored in `experiments/` as JSON files.

---

## EXP-001 — χ Discrimination on GPT-2

**Status**: ⏳ Pending (June 2026)

**Question**: Does χ carry information about semantic ambiguity *without any fine-tuning*?

**Setup**:
- Model: GPT-2 base (124M params, frozen)
- QRR modules: BranchBank + UnitaryMixer + AmplitudeRouter (Stage 1, real amplitudes)
- Dataset: 30 ambiguous + 30 clear sentences (see `benchmarks/ambiguity_bench.py`)
- K = 4 branches

**Hypothesis**: Mean χ(ambiguous) − Mean χ(clear) > 0.15

**Success criterion**: Δχ > 0.15, p < 0.05 (Mann-Whitney U)

**How to run**:
```bash
python benchmarks/ambiguity_bench.py --model gpt2 --branches 4 \
  --output experiments/EXP-001_results.json
```

**Results**: *To be filled after run.*

| Metric | Value |
|---|---|
| χ ambiguous mean | — |
| χ clear mean | — |
| Δχ | — |
| p-value | — |
| Success | — |

---

## EXP-002 — Unitary Mixer Norm Stability

**Status**: ⏳ Pending (July 2026)

**Question**: Does Cayley parametrization keep amplitude norm drift < 5% vs. > 15% for linear mixer?

**Setup**:
- Fixed input stream, 100 forward steps
- Compare Cayley (`parametrization='cayley'`) vs. Linear (`parametrization='linear'`)
- Metric: `UnitaryMixer.norm_drift()` averaged over 100 steps

**Success criterion**: Cayley drift ≤ 5%, Linear drift > 15%

**How to run**:
```bash
python benchmarks/norm_drift_bench.py --steps 100
```

**Results**: *To be filled after run.*

---

## EXP-003 — Decoherence Timing vs. Accuracy

**Status**: ⏳ Pending (September 2026)

**Question**: Does collapsing later (higher χ threshold) improve accuracy on multi-hop QA?

**Setup**:
- Dataset: HotpotQA dev subset (500 questions)
- Thresholds: χ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
- Metric: Exact Match and F1 per threshold level

**Success criterion**: Best accuracy at χ threshold > 0.2 (later collapse is better)

---

## EXP-004 — Branch Diversity vs. Semantic Entropy

**Status**: ⏳ Pending (October 2026)

**Question**: Does branch diversity correlate with semantic entropy of the input?

**Setup**:
- 200 sentences spanning 10 semantic entropy levels
- Pearson correlation between branch diversity and semantic entropy
- Success criterion: r > 0.4
