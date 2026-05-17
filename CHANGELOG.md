# Changelog

All notable changes to QRR are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### In progress
- EXP-001: χ discrimination baseline on GPT-2
- UnitaryMixer with hard orthogonality constraint
- Complex phase curriculum (Stage 2)

---

## [0.2.0] — 2026-05-17

### Added
- Full repository structure and all module stubs
- Multi-objective loss: `L_token + L_coh + L_dec + L_cal + L_div`
- Collapse index χ (Chi) definition and module
- Ambiguity benchmark (`ambiguity_bench.py`)
- Planning benchmark (`planning_bench.py`)
- Calibration benchmark (`calibration_bench.py`)
- MANIFESTO.md, FORMALISM.md, ARCHITECTURE.md, CRITIQUE.md, ROADMAP.md
- GitHub Actions CI: lint + test + typecheck
- Issue templates: bug, feature, experiment proposal
- CITATION.cff
- Examples: basic inference, branch inspection, agent loop

### Changed
- README overhauled with logo, badges, ASCII diagrams, LaTeX math

---

## [0.1.0] — 2026-05-16

### Added
- Initial theory draft and repo scaffold
- Core branch_bank.py and decoherence_module.py
- Base QRR model wrapping HuggingFace transformer

---

## Versioning Policy

- `0.x.y` — Pre-publication research phase. Breaking changes allowed.
- `1.0.0` — First stable release, published alongside paper draft v0.
