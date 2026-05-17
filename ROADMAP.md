# QRR Research Roadmap

---

## Phase 1 — Prototype (Months 0–3)

**Goal**: Demonstrate that QRR-8B outperforms LLM-8B on ambiguity benchmarks at ≤4x compute.

**Actions**:
- [ ] Implement Branch Bank with K=8, real amplitudes (φ=0)
- [ ] Implement Unitary Mixer with dense U (no topology yet)
- [ ] Implement Amplitude Router (real-valued MLP)
- [ ] Implement Observer + Decoherence modules
- [ ] Fine-tune on ambiguous QA dataset (multi-interpretation questions)
- [ ] Measure: accuracy, ECE, premature collapse rate, χ distribution

**Go/No-Go Criteria**:
- ✅ Go: QRR-8B improves ECE by >10% vs. baseline on ambiguity benchmark
- ❌ Stop: No measurable improvement AND branch diversity collapses to K=1 consistently

---

## Phase 2 — Complex Amplitudes and Interference (Months 3–6)

**Goal**: Validate that complex phases `φ` carry semantic information and interference provides measurable gains.

**Actions**:
- [ ] Curriculum: unlock phase from real to complex
- [ ] Phase regularization and syndrome correction
- [ ] Probing experiments: can `Δφ` predict semantic similarity?
- [ ] Interference term analysis: is it nonzero and correlated with performance?
- [ ] Benchmark: multi-hop reasoning, counterfactual QA

**Go/No-Go Criteria**:
- ✅ Go: Phase probe accuracy >60% on semantic similarity task
- ❌ Stop: Phase collapses to near-zero or is random across contexts

---

## Phase 3 — Topological Entanglement and Geometry (Months 6–12)

**Goal**: Learn sparse branch interaction graph and Riemannian metric.

**Actions**:
- [ ] Implement topological entanglement graph `G = (V, E)`
- [ ] Learn Riemannian metric `G` over branch space
- [ ] Test: does semantic distance in branch space correlate with task relevance?
- [ ] Reduce compute via sparse E (target: mean degree ≤ 4)
- [ ] Agent benchmark: multi-step tool use with deferred commitment

**Go/No-Go Criteria**:
- ✅ Go: 30%+ reduction in planning task failure rate vs. single-branch agent
- ❌ Stop: Sparse graph provides no benefit and dense graph is compute-prohibitive

---

## Phase 4 — Hierarchical Scale and Online Learning (Months 12–18)

**Goal**: Scale to K₁+K₂+K₃ hierarchy. Enable branch birth/death during fine-tuning.

**Actions**:
- [ ] Implement 3-level hierarchical branch structure
- [ ] Branch birth/death during fine-tuning (frozen backbone)
- [ ] Evaluate fidelity decay across scale levels
- [ ] Dynamic SNR-based K* optimization

---

## Phase 5 — Moral Decoherence and Safety (Months 18–24)

**Goal**: Incorporate moral utility into collapse operator. Validate that model refuses forced harmful commitment.

**Actions**:
- [ ] Design U_mor from RLHF + explicit rules
- [ ] Adversarial probing: can U_mor be gamed?
- [ ] Human oversight protocol for high-χ, low-U_mor states
- [ ] Red-team: injection attacks on Observer module

---

## Falsification Criteria

The QRR research program should be abandoned or fundamentally reformulated if:

1. Branch diversity collapses to K=1 consistently across training runs, regardless of diversity loss.
2. Complex phases show no correlation with semantic compatibility across 3+ independent experiments.
3. χ index does not predict output error better than token-level entropy.
4. Decoherence timing RL fails to learn non-trivial policies after 10k episodes.
5. QRR provides no measurable benefit over a well-tuned MoE baseline at equivalent compute.

If any of these hold, the model degenerates to a known architecture (ensemble, MoE, beam search) and its differential claim is invalid.
