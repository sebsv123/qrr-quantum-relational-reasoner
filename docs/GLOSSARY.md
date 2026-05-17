# QRR Glossary

Terms used throughout the QRR codebase, papers, and experiments.
Ordered alphabetically. See FORMALISM.md for mathematical definitions.

---

## A

**Amplitude** (`α^(k) ∈ ℂ`)
A complex number associated with each branch. Its squared modulus |`α^(k)`|^2 gives the probability of that branch (Born rule). Real part: support. Imaginary part: phase.

**AmplitudeRouter**
Module that updates complex amplitudes based on evidence from the current token. Computes a per-branch gain `r^(k) · e^{iθ^(k)}` from branch-context compatibility scores.

## B

**Born rule**
Adapted from quantum mechanics. The probability of branch k is `p^(k) = |α^(k)|^2 / Σ_j |α^(j)|^2`.

**Branch**
One competing hypothesis maintained in the branch bank. Each branch has a hidden state `h^(k) ∈ R^d` and a complex amplitude `α^(k) ∈ ℂ`.

**Branch Bank**
The set of K branches `{(h^(k), α^(k))}_{k=1}^K`. Initialized from the base transformer's hidden state with small perturbations. See `qrr/branch_bank.py`.

**Branch Diversity**
Mean pairwise cosine distance between branch hidden states. Low diversity = branches collapsed to similar representations. High diversity = genuine multi-hypothesis exploration.

## C

**Cayley Parametrization**
A way to parametrize orthogonal matrices: `U = (I - A)(I + A)^{-1}` where A is skew-symmetric. Guarantees `U^T U = I`, preserving amplitude norms exactly. Used in UnitaryMixer.

**Chi (χ)** — *Collapse Index*
A scalar in [0, 1] measuring branch diversity. Formally: `χ = 1 - Σ_k p^(k)^2` (Simpson's diversity index). `χ = 0`: all probability on one branch (definite). `χ = 1 - 1/K`: uniform branches (maximal ambiguity).

**Coherent Evolution**
Branch update that preserves total amplitude norm. Analogous to unitary evolution in quantum mechanics. Implemented via UnitaryMixer with Cayley or linear parametrization.

**Collapse**
The act of reducing K branches to a single hidden state for output generation. Triggered when `χ < threshold` (confident enough to commit). See DecoherenceModule and CollapseIndex.

**Collapse Index** → see Chi.

## D

**Decoherence**
Inspired by quantum decoherence: the process by which interference between branches is suppressed and the system settles into a definite state. In QRR, triggered externally when action is required.

**DecoherenceModule**
Module that collapses K branches into one hidden state. Supports two strategies:
- `weighted_sum`: soft collapse weighted by `p^(k)`
- `argmax`: hard collapse to the most probable branch

**Delta Chi (Δχ)**
Key metric for EXP-001: mean `χ(ambiguous) − mean χ(clear)`. Success criterion: Δχ > 0.15.

## E

**EMA (Exponential Moving Average)**
Used in CollapseIndex to smooth `χ` estimates across steps, preventing premature collapse from noise. Controlled by `smoothing` parameter (default 0.9).

**Entanglement (Contextual)**
QRR's analog: each new token correlates different branches with different sub-spaces of the context. Branches that are compatible with the new evidence are amplified; incompatible branches are suppressed.

## H

**Hidden State** (`h^(k) ∈ R^d`)
The latent representation of branch k. Updated by UnitaryMixer at each step. Projected to logits via LM head after collapse.

## I

**Interference**
When two branches share similar representations (high cosine similarity), their amplitudes can add constructively (both rise) or destructively (they cancel). Controlled by UnitaryMixer's mixing matrix.

**Interference Loss** (`ℒ_div`)
Training objective that penalizes low branch diversity, encouraging the model to maintain genuinely distinct hypotheses rather than collapsing all branches to the same representation.

## K

**K** — *Number of Branches*
Hyperparameter controlling how many competing hypotheses the model maintains. Default: 4. Ablation study (Phase 4) will test K ∈ {1, 2, 4, 8}.

## L

**Late Collapse**
The core principle of QRR: delay committing to a single interpretation until evidence forces it. Contrast with standard autoregressive decoding, which commits at each token.

## P

**Phase** (`θ^(k) ∈ [-π, π]`)
Imaginary component of the complex amplitude. Controls interference angle between branches. Introduced in Stage 2 training. Zero in Stage 1 (real amplitudes only).

**Probabilities** (branch)
Born-rule probabilities: `p^(k) = |α^(k)|^2 / Σ_j |α^(j)|^2`. Used for weighted collapse and χ computation.

## S

**Stage 1 / Stage 2**
Curriculum training schedule:
- Stage 1: frozen base, real amplitudes, basic branch diversity
- Stage 2: frozen base, full complex amplitudes with phases, interference training

**Superposition (Semantic)**
Not quantum superposition literally. The metaphor: the model maintains multiple semantic interpretations simultaneously before collapsing to one. Implemented as a classical branch bank with complex-valued weights.

## U

**UnitaryMixer**
Module that performs coherent mixing of branch hidden states and amplitudes using an approximately unitary transformation. See Cayley Parametrization.

## χ
See Chi.
