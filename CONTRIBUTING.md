# Contributing to QRR

Thank you for your interest in QRR. This is an open research project and all contributions are welcome — from theory to code to experiments to critique.

---

## Ways to Contribute

| Type | Where to start |
|---|---|
| 🐛 Bug fix | Open a bug report issue, then submit a PR |
| 💡 New module | Open a feature request first; discuss the design |
| 🧪 Experiment | Use the Experiment Proposal issue template |
| 📐 Theory | Open an issue with the proposed formalism change |
| 📄 Documentation | Direct PR welcome |
| 🔍 Adversarial critique | Add to CRITIQUE.md or open an issue |

---

## Development Setup

```bash
git clone https://github.com/sebsv123/qrr-quantum-relational-reasoner
cd qrr-quantum-relational-reasoner
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff mypy pytest pytest-cov
```

## Code Style

```bash
ruff check qrr/ training/ benchmarks/ tests/
mypy qrr/ --ignore-missing-imports
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Commit Convention

```
feat:     New module or capability
fix:      Bug fix
docs:     Documentation only
test:     Tests only
refactor: Code change without new feature or bug fix
exp:      Experiment script or result
chore:    Dependency update, CI, tooling
```

---

## Research Integrity

QRR is a falsifiable research project. If you find evidence that contradicts the theory:
- Please open an issue with label `research: challenge`
- Add your finding to CRITIQUE.md
- Propose an experiment that would resolve the disagreement

We welcome adversarial contributions as much as supportive ones.

---

## Code of Conduct

Be respectful, be precise, be honest about uncertainty. This is science.
