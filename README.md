# Quantum Relational Reasoner (QRR)

> *A new AI paradigm: do not collapse reality until evidence demands it.*

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()

---

## What is QRR?

The **Quantum Relational Reasoner** is a novel AI architecture inspired by the relational interpretation of quantum mechanics. It does **not** require quantum hardware. It is a classical model with a fundamentally different internal representation:

- Instead of a single hidden state → a **superposition of K competing hypotheses** (branches).
- Instead of early commitment → **coherent evolution** of all branches until evidence forces a decision.
- Instead of entropy-as-uncertainty → **complex amplitudes with phase**, enabling constructive and destructive interference between hypotheses.
- Instead of a fixed observer → an **internalized, selective observer** that only triggers collapse when action demands it.

This is not "quantum ML" as a marketing label. It is a formal attempt to import three structural ideas from quantum mechanics into classical inference:

| Quantum Principle | QRR Mechanism |
|---|---|
| Superposition | K latent branches with complex amplitudes |
| Interference | Branch compatibility modulates mutual amplification or cancellation |
| Decoherence | Collapse only when evidence/action/cost requires commitment |

---

## Repository Structure

```
qrr-quantum-relational-reasoner/
├── README.md                   ← This file
├── MANIFESTO.md                ← Foundational axioms and theory
├── ARCHITECTURE.md             ← Technical spec of all modules
├── FORMALISM.md                ← Full mathematical formalism
├── ROADMAP.md                  ← Experimental agenda (0–24 months)
├── CRITIQUE.md                 ← Hostile review: known failures and mitigations
├── qrr/
│   ├── __init__.py
│   ├── branch_bank.py          ← Core: K latent branches with amplitudes
│   ├── unitary_mixer.py        ← Approximate unitary evolution operator
│   ├── amplitude_router.py     ← Complex amplitude routing between branches
│   ├── entanglement_module.py  ← Context-branch correlation
│   ├── decoherence_module.py   ← Controlled collapse mechanism
│   ├── observer_module.py      ← Internalized selective observer
│   ├── collapse_index.py       ← χ index: residual ambiguity measure
│   └── qrr_model.py            ← Full QRR model (wraps transformer base)
├── training/
│   ├── loss.py                 ← Multi-objective loss function
│   ├── curriculum.py           ← Curriculum: real amplitudes → complex phases
│   └── train.py                ← Training loop
├── benchmarks/
│   ├── ambiguity_bench.py      ← Disambiguation QA benchmark
│   ├── planning_bench.py       ← Multi-step planning benchmark
│   ├── calibration_bench.py    ← ECE + branch calibration
│   └── decoherence_timing.py   ← Collapse timing evaluation
├── experiments/
│   └── README.md               ← Experiment log
├── docs/
│   └── diagrams/               ← Architecture diagrams
└── tests/
    └── test_branch_bank.py     ← Unit tests
```

---

## Quick Start

```bash
git clone https://github.com/sebsv123/qrr-quantum-relational-reasoner
cd qrr-quantum-relational-reasoner
pip install -r requirements.txt
python -m qrr.qrr_model --demo
```

---

## Core Principle in One Sentence

> Classical AI collapses too early. QRR delays commitment until the cost of ambiguity exceeds the cost of decision.

---

## Status

This is an **active research project**. The theory is in version 2.0. The codebase is in early MVP phase.

See [ROADMAP.md](ROADMAP.md) for the experimental agenda and falsification criteria.

---

## Author

**Sebastián Sifontes Valentín** — Madrid, ES  
Economist × Data Scientist × AI Engineer  
[GitHub](https://github.com/sebsv123) · [Web](https://valentinproteccionintegral.com)

---

## License

MIT — see [LICENSE](LICENSE)
