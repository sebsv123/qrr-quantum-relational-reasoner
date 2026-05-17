# Changelog

All notable changes to QRR are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In progress
- EXP-001: χ discrimination baseline on GPT-2
- UnitaryMixer with approximate orthogonality constraint
- Complex phase curriculum (Stage 2)

---

## [0.2.0] — 2026-05-18

### Added
- Full repository structure and all module stubs
- Multi-objective loss: `L_token + L_coh + L_dec + L_cal + L_div`
- Collapse index χ (Chi) definition and module
- Ambiguity benchmark (`ambiguity_bench.py`)
- Planning benchmark (`planning_bench.py`)
- Calibration benchmark (`calibration_bench.py`)
- MANIFESTO.md, FORMALISM.md, ARCHITECTURE.md, CRITIQUE.md, ROADMAP.md
- GitHub Actions CI: lint + test + typecheck (Python 3.10 / 3.11 / 3.12)
- Issue templates: bug, feature, experiment proposal
- PR template
- CITATION.cff
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Examples: basic inference, branch inspection, agent loop
- Notebook EXP-001: χ discrimination baseline
- pyproject.toml, setup.cfg, .gitignore
- Official QRR logo

### Changed
- README overhauled: logo, animated banner, badges, ASCII diagrams, LaTeX math, roadmap table

---

## [0.1.0] — 2026-05-17

### Added
- Initial theory draft and repo scaffold
- Core `branch_bank.py` and `decoherence_module.py`
- Base QRR model wrapping HuggingFace transformer
- MANIFESTO.md v1 and FORMALISM.md v1

---

## Versioning Policy

- `0.x.y` — Pre-publication research phase. Breaking changes possible between minor versions.
- `1.0.0` — First stable release, published alongside paper draft v0.
