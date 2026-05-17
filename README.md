<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:01696f,100:0d1117&height=200&section=header&text=Quantum%20Relational%20Reasoner&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=Don't%20collapse%20reality%20until%20evidence%20demands%20it.&descSize=16&descAlignY=60&descColor=4f98a3" alt="QRR Banner"/>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4f98a3?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Status-Active%20Research-f59e0b?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/HuggingFace-Compatible-fbbf24?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"/>
</p>

<p align="center">
  <a href="#-what-is-qrr">What is QRR?</a> ·
  <a href="#-core-idea">Core Idea</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-roadmap">Roadmap</a> ·
  <a href="#-author">Author</a>
</p>

---

## 🧠 What is QRR?

**Quantum Relational Reasoner (QRR)** is a novel AI architecture that delays commitment to a single interpretation until evidence demands it.

> *Classical LLMs collapse too early — they pick one path through ambiguous space before they have enough information. QRR doesn't.*

Instead of propagating a single hidden state, QRR maintains a **superposition of K competing hypotheses** at every inference step, each with a complex amplitude. It is a fully classical model — no quantum hardware required — that imports three structural ideas from quantum mechanics into classical inference.

```
Traditional LLM:       token → [single hidden state] → output
                                      ↑
                              forced early collapse

QRR:                   token → [branch₁ α₁] ─────┐
                               [branch₂ α₂] ──── coherent evolution → collapse when ready
                               [branch_K αK] ─────┘
```

---

## 💡 Core Idea

| Quantum Principle | QRR Mechanism | Why it matters |
|---|---|---|
| **Superposition** | K latent branches with complex amplitudes | Multiple interpretations coexist explicitly |
| **Interference** | Branch compatibility modulates mutual amplification/cancellation | Consistent hypotheses reinforce; contradictory ones suppress |
| **Decoherence** | Collapse only when evidence/action/cost requires commitment | No premature decisions; better calibration |
| **Entanglement** | Context-branch correlation across time steps | Rich relational memory between observations |

The key metric is **χ (Chi)** — the *collapse index* — a scalar in `[0, 1]` that represents the residual ambiguity in the model's belief state at any moment:

- `χ → 0`: high confidence, safe to act
- `χ → 1`: high ambiguity, defer commitment
- `χ = threshold`: trigger decoherence, commit to a branch distribution

---

## 🏗 Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Transformer Base (frozen)          │
└─────────────────────┬───────────────────────────┘
                      │ h_t ∈ ℝ^d
                      ▼
┌─────────────────────────────────────────────────┐
│                  Branch Bank                    │
│  K branches: {(h_t^(k), α_t^(k))} k=1..K       │
│  α_t^(k) ∈ ℂ  →  p_t^(k) = |α_t^(k)|²         │
└──────┬──────────────┬──────────────┬────────────┘
       │              │              │
       ▼              ▼              ▼
 UnitaryMixer   EntanglementMod  AmplitudeRouter
 (coherent      (context-branch  (complex routing
  evolution)     correlation)     between branches)
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Decoherence Module                 │
│  Triggered when χ < threshold                  │
│  Selects or merges branches → final distribution│
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
             Output + Collapse Index χ
```

### Mathematical Core

Each step evolves the branch bank via:

$$\tilde{h}_{t+1}^{(k)} = \sum_{j=1}^{K} U_{kj}(x_t, c_t) \cdot h_t^{(j)}$$

$$p_t^{(k)} = \frac{|\tilde{\alpha}_{t+1}^{(k)}|^2}{\sum_j |\tilde{\alpha}_{t+1}^{(j)}|^2}$$

$$\chi_t = 1 - \max_k p_t^{(k)}$$

The multi-objective training loss:

$$\mathcal{L} = \mathcal{L}_{token} + \lambda_1 \mathcal{L}_{coh} + \lambda_2 \mathcal{L}_{dec} + \lambda_3 \mathcal{L}_{cal} + \lambda_4 \mathcal{L}_{div}$$

---

## 📁 Repository Structure

```
qrr-quantum-relational-reasoner/
│
├── 📄 README.md                   ← You are here
├── 📜 MANIFESTO.md                ← Foundational axioms and theory
├── 🏗 ARCHITECTURE.md             ← Full technical spec
├── 📐 FORMALISM.md                ← Mathematical formalism
├── 🗺 ROADMAP.md                  ← Experimental agenda (0–24 months)
├── 🔍 CRITIQUE.md                 ← Hostile review: known failures + mitigations
│
├── qrr/
│   ├── branch_bank.py             ← Core: K latent branches with complex amplitudes
│   ├── unitary_mixer.py           ← Approximate unitary evolution operator
│   ├── amplitude_router.py        ← Complex amplitude routing
│   ├── entanglement_module.py     ← Context-branch correlation
│   ├── decoherence_module.py      ← Controlled collapse mechanism
│   ├── observer_module.py         ← Internalized selective observer
│   ├── collapse_index.py          ← χ: residual ambiguity scalar
│   └── qrr_model.py               ← Full model (wraps HuggingFace base)
│
├── training/
│   ├── loss.py                    ← Multi-objective loss
│   ├── curriculum.py              ← Real amplitudes → complex phases curriculum
│   └── train.py                   ← Training loop
│
├── benchmarks/
│   ├── ambiguity_bench.py         ← Disambiguation QA (primary benchmark)
│   ├── planning_bench.py          ← Multi-step planning
│   ├── calibration_bench.py       ← ECE + branch calibration
│   └── decoherence_timing.py      ← Collapse timing evaluation
│
├── experiments/
│   └── README.md                  ← Experiment log (EXP-001 pending)
│
└── tests/
    └── test_branch_bank.py        ← Unit tests
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/sebsv123/qrr-quantum-relational-reasoner
cd qrr-quantum-relational-reasoner

# Install
pip install -r requirements.txt

# Run demo
python -m qrr.qrr_model --demo

# Run first benchmark
python benchmarks/ambiguity_bench.py --model gpt2 --branches 4
```

---

## 🗺 Roadmap

| Phase | Milestone | Target | Status |
|---|---|---|---|
| **Phase 0** | Theory stabilization | Q1 2025 | ✅ Done |
| **Phase 1** | EXP-001: χ discrimination on GPT-2 | Q2 2025 | 🔄 In progress |
| **Phase 2** | Unitary mixer + complex phases | Q3 2025 | ⏳ Planned |
| **Phase 3** | Fine-tuning on ambiguity datasets | Q4 2025 | ⏳ Planned |
| **Phase 4** | Benchmarks vs MoE / Beam Search | Q1 2026 | ⏳ Planned |
| **Phase 5** | Paper draft v0 | Q2 2026 | ⏳ Planned |

See [ROADMAP.md](ROADMAP.md) for detailed falsification criteria.

---

## 🆚 How QRR Differs

| Approach | Multi-hypothesis? | Interference? | Late collapse? |
|---|---|---|---|
| Standard LLM | ❌ single path | ❌ | ❌ |
| Beam Search | ✅ parallel beams | ❌ | ✅ |
| Mixture of Experts | ✅ expert mixing | ❌ | ❌ |
| Ensemble | ✅ separate models | ❌ | ✅ |
| **QRR** | ✅ K latent branches | ✅ complex amplitudes | ✅ χ-gated |

---

## 📊 Target Use Cases

```
🔍 Ambiguous Reasoning    → NLU with competing interpretations
🛠 Agent Planning         → Multi-route tool-use before committing to actions
🐛 Code Debugging         → Multiple bug hypotheses before patching
🏥 Medical Diagnosis      → Differential diagnosis as explicit branch state
📋 Root Cause Analysis    → Multi-hypothesis incident investigation
```

---

## 🤝 Contributing

This is an open research project. Contributions welcome:

1. **Theory**: Formalism improvements, alternative decoherence triggers
2. **Code**: Module implementations, unit tests, benchmarks
3. **Experiments**: Running EXP-001 on your hardware
4. **Critique**: Adversarial challenges to the architecture

Open an issue or submit a PR. See [CRITIQUE.md](CRITIQUE.md) for known weaknesses.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 👤 Author

<p align="left">
  <strong>Sebastián Sifontes Valentín</strong><br/>
  Economist × Data Scientist × AI Engineer — Madrid, ES<br/>
  <a href="https://github.com/sebsv123">GitHub</a> ·
  <a href="https://valentinproteccionintegral.com">Web</a>
</p>

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:01696f,100:0d1117&height=100&section=footer" alt="footer"/>
</p>
