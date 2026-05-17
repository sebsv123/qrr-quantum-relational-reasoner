# QRR Mathematical Formalism

---

## 1. State Representation

At time `t`, the model state is:

```
|Ψ_t⟩ = Σ_{k=1}^{K} α_t^(k) |R_k⟩ ⊗ |C_t⟩ ⊗ |E_t⟩
```

Where:
- `R_k` = branch k (latent hypothesis)
- `α_t^(k) ∈ ℂ` = complex amplitude, with `Σ_k |α_t^(k)|² = 1`
- `C_t` = context (prompt, memory, history)
- `E_t` = environment (tools, external state)

Each branch: `h^(k) ∈ ℝ^d` (or `ℂ^d` in full complex variant)

---

## 2. Unitary Evolution Operator

For each layer `l`, coherent mixing:

```
h^(k)_{l+1} = Σ_j U^(l)_{kj} · h^(j)_l
α^(k)_{l+1} = Σ_j M^(l)_{kj} · α^(j)_l
```

Constraints:
- `U†U ≈ I` (approximate unitarity via Frobenius penalty)
- `Σ_k |α^(k)|² = 1` preserved at every step
- `M^(l)_{kj}` nonzero only if `(k,j) ∈ E_t` (topological constraint)

---

## 3. Interference Term

For a measurement basis `{|b_i⟩}`:

```
P(b_i) = |Σ_k α^(k) ⟨b_i|h^(k)⟩|²
       = Σ_k |α^(k)|² |⟨b_i|h^(k)⟩|²
       + Σ_{k≠j} α^(k)* α^(j) ⟨b_i|h^(k)⟩* ⟨b_i|h^(j)⟩
```

The second term is the **interference term**. It is constructive when phases align and destructive when they oppose.

---

## 4. Entanglement with Context

When new observation `o_t` arrives:

```
|Ψ_t⟩ → Σ_k α^(k) |R_k⟩ ⊗ |o_t^(k)⟩
```

Where `|o_t^(k)⟩` is the projection of `o_t` compatible with branch `k`.

Mutual information between branches and observation:
```
I(R; O) = Σ_k |α^(k)|² · log[ p(o_t | h^(k)) / p(o_t) ]
```

When `I(R; O) > τ_collapse`, decoherence is triggered.

---

## 5. Decoherence Channel

Full state before decoherence:
```
ρ = |Ψ⟩⟨Ψ| = Σ_{k,j} α^(k) α^(j)* |R_k⟩⟨R_j|
```

Reduced density matrix after tracing environment:
```
ρ_reduced = Σ_k |α^(k)|² |R_k⟩⟨R_k|   (off-diagonals ≈ 0)
```

Collapse: sample `k* ~ p^(k) = |α^(k)|²`, output from `R_{k*}`.

---

## 6. Collapse Index χ

```
χ_t = 1 - Σ_k |α^(k)|⁴ / (Σ_k |α^(k)|²)²
```

- `χ = 0`: fully collapsed (one branch dominates)
- `χ = 1`: maximum superposition (uniform distribution over K branches)
- Collapse triggered when `χ < τ_χ` OR external action required

---

## 7. Riemannian Branch Metric

Distance between branches `i` and `j`:
```
d(R_i, R_j) = sqrt[ (h^(i) - h^(j))ᵀ G (h^(i) - h^(j)) ]
```

Where `G` is a learned metric tensor. Gradient:
```
∂L/∂G = λ_metric · [ D_incompatible - D_compatible ]
```

This ensures semantically close hypotheses cluster, and contradictory ones separate.

---

## 8. Multi-Objective Loss

```
L = L_token + λ₁ L_coh + λ₂ L_dec + λ₃ L_cal + λ₄ L_ent + λ₅ L_moral
```

Where:
- `L_token`  = standard cross-entropy on generated tokens
- `L_coh`    = coherence loss: penalizes premature collapse when evidence is insufficient
- `L_dec`    = decoherence loss: penalizes delayed collapse when evidence is clear
- `L_cal`    = calibration: aligns `|α^(k)|²` with empirical error rate (ECE)
- `L_ent`    = entanglement: maximizes correct branch-context correlation
- `L_moral`  = moral utility alignment: RLHF-derived signal on `U_mor(R_k)`

---

## 9. Branch Uncertainty Principle

Analogous to Heisenberg:
```
ΔN · ΔE ≥ ℏ_eff / 2
```

Where:
- `ΔN` = uncertainty in number of useful branches
- `ΔE` = uncertainty in available computational energy
- `ℏ_eff` = system-dependent effective constant (function of hardware, noise floor)

Implication: there exists an optimal `K*` beyond which noise dominates signal. The model must monitor SNR per branch and prune dynamically.

---

## 10. New Branch Generation (Constructive Interference)

For branch pair `(i, j)` with `|α^(i)|, |α^(j)| > ε` and `|arg(α^(i)) - arg(α^(j))| < δ`:

```
h^(new) = (h^(i) + h^(j)) / ||h^(i) + h^(j)||
α^(new) = α^(i) + α^(j)  (renormalized)
```

New branch persists if validated by subsequent evidence, dissolved otherwise.
