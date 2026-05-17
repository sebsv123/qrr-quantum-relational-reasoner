# QRR Research Roadmap

> All milestones are defined with falsification criteria. A milestone is complete only when its numerical criterion is met.

---

## Current Status: Phase 1 — Empirical Baseline

As of May 2026, the theory (v2.0) and codebase scaffolding (v0.2.0) are complete.  
The next step is running EXP-001 to establish whether χ discriminates ambiguous inputs.

---

## Phase 0 — Theory Stabilization ✅
**Period**: May 2026  
**Deliverables**:
- [x] MANIFESTO.md: 9 foundational axioms
- [x] FORMALISM.md: full mathematical specification
- [x] ARCHITECTURE.md: module-level technical spec
- [x] CRITIQUE.md: 15 known failure modes with mitigations
- [x] Codebase scaffold: all modules stubbed

---

## Phase 1 — Empirical Baseline 🔄
**Period**: June – August 2026  
**Goal**: Establish that χ carries information about semantic ambiguity without any fine-tuning.

### EXP-001: χ Discrimination on GPT-2
- **Hypothesis**: Mean χ(ambiguous) − Mean χ(clear) > 0.15 on a 100-sample dataset
- **Model**: GPT-2 base (no fine-tuning)
- **Dataset**: AmbigQA subset + hand-curated clear statements
- **Success criterion**: Δχ > 0.15, p < 0.05 (Mann-Whitney U)
- **Falsification**: If Δχ ≤ 0.05, revisit branch initialization and amplitude router design

### EXP-002: Unitary Mixer Stability
- **Hypothesis**: Orthogonality constraint on UnitaryMixer keeps amplitude norm drift < 5% over 100 steps
- **Baseline**: Free linear mixer (expected drift > 15%)
- **Success criterion**: Constrained norm drift ≤ 5% vs unconstrained > 15%

---

## Phase 2 — Architecture Refinement ⏳
**Period**: September – November 2026  
**Goal**: Implement full complex-amplitude pipeline and validate interference.

- Implement `UnitaryMixer` with hard orthogonality (Cayley parametrization)
- Stage 2 curriculum: real amplitudes → complex phases
- EXP-003: Decoherence timing vs. accuracy tradeoff on multi-hop QA
- EXP-004: Branch diversity metric vs. semantic entropy correlation

---

## Phase 3 — Fine-Tuning ⏳
**Period**: December 2026 – February 2027  
**Goal**: Supervised fine-tuning on ambiguity-rich datasets.

- Fine-tune on AmbigQA, multi-hop QA (HotpotQA, MuSiQue)
- Fine-tune on planning tasks (ALFWorld, WebArena subsets)
- EXP-005: Fine-tuned QRR vs. standard LLM on ambiguity resolution
- EXP-006: Calibration improvement (ECE reduction vs. baseline)

---

## Phase 4 — Comparative Benchmarks ⏳
**Period**: March – May 2027  
**Goal**: Rigorous comparison vs. Beam Search, MoE, Ensemble.

- Iso-latency benchmark: QRR vs Beam Search (same FLOPs budget)
- Multi-task benchmark: ambiguity QA + planning + calibration
- Ablation study: K branches (1, 2, 4, 8) vs. accuracy/latency tradeoff
- Ablation: with/without interference (complex vs. real amplitudes)

---

## Phase 5 — Paper Draft v0 ⏳
**Period**: June 2027  
**Goal**: Preprint on arXiv.

- Draft covering: motivation, formalism, EXP-001–006 results
- Reproducibility: full training code, datasets, checkpoints released
- Target venues: NeurIPS 2027, ICML 2027 (conditional on Phase 4 results)

---

## Falsification Policy

QRR is a falsifiable research project. The following results would require fundamental revision:

| Finding | Implication |
|---|---|
| Δχ ≤ 0.05 in EXP-001 | Rethink amplitude router — χ carries no signal |
| No accuracy improvement over beam search (EXP-004) | Branches add compute without benefit — revisit K and decoherence |
| Complex amplitudes no better than real (Phase 2 ablation) | Drop complex arithmetic — use real-valued branch weights |
| Fine-tuned QRR ≤ standard LLM (EXP-005) | Architecture overhead not justified — consider lighter variants |

Negative results will be published. Failure to falsify is not the goal — understanding is.
