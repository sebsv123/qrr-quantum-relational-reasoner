# QRR Architecture Specification

---

## Overview

QRR wraps a standard transformer backbone (Llama-3, Mistral, or similar) with a set of additional modules that implement branch superposition, coherent evolution, and controlled decoherence.

The transformer handles token-level representations. QRR operates at the **sequence-segment level**, maintaining K hypotheses per reasoning step.

---

## Module Map

```
Input tokens
    │
    ▼
[Transformer Backbone]  ← frozen or fine-tuned
    │
    ▼
[Branch Bank]           ← K branches {(h^(k), α^(k))}
    │
    ├──▶ [Amplitude Router]       ← assigns complex amplitudes
    │
    ├──▶ [Unitary Mixer]          ← coherent inter-branch evolution
    │
    ├──▶ [Entanglement Module]    ← branch-context correlation
    │
    ├──▶ [Observer Module]        ← monitors χ, triggers collapse
    │
    └──▶ [Decoherence Module]     ← reduces ρ, selects branch k*
                │
                ▼
    [Output Head]  ← token generation from R_{k*}
         +
    [Collapse Index χ]     ← residual ambiguity signal
    [Branch Map (top-3)]   ← visible hypotheses + amplitudes
    [Action Proposal]      ← if agent: parallel action routes
```

---

## Module Specifications

### 1. Branch Bank

**Purpose**: Maintain K latent hypotheses at each reasoning step.

```python
class BranchBank(nn.Module):
    """
    Inputs:  x ∈ ℝ^(B, L, d)       — transformer hidden states
    Outputs: H ∈ ℝ^(B, K, d)       — K branch hidden states
             A ∈ ℂ^(B, K)          — complex amplitudes
    """
    K: int           # number of branches (default: 8–16)
    d: int           # hidden dimension
    phase_init: str  # 'zero' (real only, phase=0) or 'random'
```

**Initialization strategy**:
- Phase 1 (curriculum): `phase = 0` for all branches (real amplitudes only).
- Phase 2: unlock phase parameter and allow complex amplitudes.

---

### 2. Amplitude Router

**Purpose**: Assign complex amplitudes `α^(k)` to each branch conditioned on context.

```python
class AmplitudeRouter(nn.Module):
    """
    Inputs:  x ∈ ℝ^(B, L, d)       — context representation
    Outputs: A ∈ ℂ^(B, K)          — normalized complex amplitudes
             where Σ_k |A_k|² = 1
    """
    # Architecture: MLP → (r, φ) polar → α = r·e^(iφ)
    # Sparsity constraint: at most m < K branches with |α_k|² > ε
```

---

### 3. Unitary Mixer

**Purpose**: Mix branch hidden states via approximate unitary transform.

```python
class UnitaryMixer(nn.Module):
    """
    Inputs:  H ∈ ℝ^(B, K, d), A ∈ ℂ^(B, K), E ∈ {0,1}^(K,K) (edge mask)
    Outputs: H' ∈ ℝ^(B, K, d), A' ∈ ℂ^(B, K)
    """
    # Constraint: U†U ≈ I via Frobenius regularization
    # Edge mask E from learned topological graph
```

---

### 4. Entanglement Module

**Purpose**: Correlate branches with incoming observations (context, tools, memory).

```python
class EntanglementModule(nn.Module):
    """
    Inputs:  H ∈ ℝ^(B, K, d), o_t ∈ ℝ^(B, d_obs)
    Outputs: H_ent ∈ ℝ^(B, K, d)   — context-entangled branches
             I_ko ∈ ℝ^(B, K)        — mutual info estimate per branch
    """
    # For each branch k: compute compatibility score with o_t
    # Amplify compatible branches, attenuate incompatible ones
```

---

### 5. Observer Module

**Purpose**: Internalized observer. Monitors χ and decides whether to trigger collapse.

```python
class ObserverModule(nn.Module):
    """
    Inputs:  A ∈ ℂ^(B, K), I_ko ∈ ℝ^(B, K), action_required: bool
    Outputs: collapse_flag: bool
             chi ∈ ℝ^(B)     — collapse index χ ∈ [0,1]
    """
    # χ = 1 - Σ|α|⁴ / (Σ|α|²)²
    # collapse_flag = True if:
    #   - action_required
    #   - χ < τ_χ (low ambiguity)
    #   - I(R;O) > τ_I (strong evidence)
    #   - budget exceeded
```

---

### 6. Decoherence Module

**Purpose**: Reduce ρ to classical mixture. Sample branch k*.

```python
class DecoherenceModule(nn.Module):
    """
    Inputs:  H ∈ ℝ^(B, K, d), A ∈ ℂ^(B, K), collapse_flag: bool
    Outputs: h_out ∈ ℝ^(B, d)   — selected branch hidden state
             k_star ∈ ℤ^(B)     — selected branch index
             p ∈ ℝ^(B, K)       — branch probabilities |α_k|²
    """
    # If collapse_flag: sample k* ~ p^(k) = |α^(k)|²
    # If not: return weighted mixture h_out = Σ_k p^(k) h^(k)
```

---

### 7. Collapse Index

**Purpose**: Scalar signal for residual ambiguity. Useful for downstream decision-making.

```python
def compute_chi(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    amplitudes: (B, K) complex
    returns: chi (B,) ∈ [0, 1]
    chi = 1 - Σ|α|⁴ / (Σ|α|²)²
    """
```

---

## Tensor Shapes Summary

| Tensor | Shape | Description |
|---|---|---|
| `x` | `(B, L, d)` | Transformer hidden states |
| `H` | `(B, K, d)` | Branch hidden states |
| `A` | `(B, K)` complex | Branch amplitudes |
| `p` | `(B, K)` | Branch probabilities `|α|²` |
| `χ` | `(B,)` | Collapse index |
| `h_out` | `(B, d)` | Output hidden state (post-collapse) |
| `E` | `(K, K)` | Entanglement topology mask |

---

## Computational Cost Estimate

| Component | Overhead vs. baseline | Notes |
|---|---|---|
| Branch Bank (K=8) | ~2x memory | K parallel hidden states |
| Unitary Mixer | ~O(K²·d) | Reduced by topology mask |
| Amplitude Router | ~+5% | Small MLP |
| Entanglement Module | ~+10% | Cross-attention variant |
| Observer + Decoherence | ~+2% | Mostly logic, not compute |
| **Total estimate (K=8)** | **~3–4x baseline** | Acceptable for research scale |
