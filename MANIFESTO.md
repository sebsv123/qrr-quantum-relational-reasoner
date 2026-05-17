# QRR Manifesto — Foundational Theory

> Version 2.0 — May 2026

---

## Thesis

Current large language models collapse the uncertainty of their internal representation too early, too often, and without principled criteria. This produces fragile planning, miscalibrated confidence, and brittle reasoning under structural ambiguity.

The **Quantum Relational Reasoner** proposes a different internal ontology: a model whose state is not a single vector, but a structured superposition of competing hypotheses, evolving coherently until observation, action, or resource constraints force a commitment.

This is not a metaphor. It is a formal architectural constraint.

---

## Axioms

**A1 — Superposition of Hypotheses**  
A model state at time `t` is not a single latent vector but a set of K branches `{(h_k, α_k)}` where `h_k ∈ ℝ^d` is a latent hypothesis and `α_k ∈ ℂ` is a complex amplitude. Probability of branch k is `|α_k|²`.

**A2 — Coherent Evolution**  
In the absence of forced observation, branches evolve via approximate unitary mixing. This preserves total amplitude mass and enables interference between compatible hypotheses.

**A3 — Interference as Semantic Compatibility**  
Branches with aligned phases reinforce each other (constructive interference). Branches with opposing phases cancel (destructive interference). Phase alignment is a learned proxy for semantic compatibility.

**A4 — Controlled Decoherence**  
Collapse to a single output occurs only when: (a) an action requires commitment, (b) evidence mutual information exceeds a threshold, or (c) computational budget demands it. Collapse is never the default.

**A5 — Relational Reality**  
There is no absolute state. Branch amplitudes are meaningful only relative to: the context, the memory, the available tools, and the internalized observer. A branch that is invisible to all observers does not participate in inference.

**A6 — Topological Entanglement**  
Not all branches can interact with all others. A learned graph `G = (V, E)` defines which branch pairs may exchange amplitude. Edges are sparse, learned, and updated during training.

**A7 — Causal Irreversibility**  
When a branch causes an irreversible external event (tool call, write, send), it becomes causally isolated. The arrow of time is a hard constraint: collapse toward past states is forbidden.

**A8 — Hierarchical Scale**  
Branches are organized in multiple scales: macro (strategy), meso (tactic), micro (detail). Decoherence propagates top-down. This limits complexity to `O(K₁ + K₂ + K₃)` rather than `O(K³)`.

**A9 — Moral Decoherence**  
The decoherence operator includes a learned moral utility `U_mor(R_k)`. Branches with utility below threshold do not collapse without human oversight. The system cannot force a morally-weighted decision alone.

---

## What QRR Is NOT

- It is **not** quantum computing on classical hardware.
- It is **not** a mixture-of-experts with a new name.
- It is **not** a claim that consciousness is quantum.
- It is **not** a metaphor dressed as architecture.
- It is **not** provably correct — it is a falsifiable research program.

---

## What QRR Claims to Solve

| Problem in Current LLMs | QRR Mechanism |
|---|---|
| Early hypothesis collapse | Deferred decoherence (A4) |
| Miscalibrated confidence | Complex amplitudes replace logit entropy (A1) |
| Fragile multi-step planning | Parallel branches with implicit backtracking (A2) |
| Creativity limited to recombination | Constructive interference generates new branches (A3) |
| Single forced narrative | Relational reality, no absolute state (A5) |
| Runaway complexity at scale | Hierarchical multiescale branches (A8) |
| Amoral forced outputs | Moral decoherence operator (A9) |

---

## Core Open Questions

- Can complex phases be trained stably in deep networks?
- Does branch interference provide measurable gains over entropy-based uncertainty?
- Is there an optimal K* for a given task complexity and compute budget?
- Can a learned decoherence operator generalize across domains?
- Is the moral utility function learnable without collapsing into reward hacking?

These are not weaknesses. They are the research agenda.
