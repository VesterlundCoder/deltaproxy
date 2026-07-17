# V4 — Why the proxy works: the spectral identity `δ = −1 + E/Q = −λ₂/λ₁`

**Proof-ladder level V4 (symbolic generalization).** This establishes the *rule*, not a
single example: the cheap float observable `δ̂ = −λ₂/λ₁` is the asymptotic value of the
arbitrary-precision double-depth delta `δ` for *every* `6F5` CMF trajectory. The numerical
certificate (`v4_identity_certificate.py`) confirms the two limits below to high accuracy
and shows `δ_arith(N) → −λ₂/λ₁` as `N → ∞`.

---

## 1. Objects

A CMF trajectory is the matrix product
```
P_N = M(1) · M(2) ··· M(N),      M(n) ∈ ℤ^{6×6}  (integer-cleared gauge).
```
Let the singular values of `P_N` be `σ₁(N) ≥ σ₂(N) ≥ …`, with **Lyapunov exponents**
```
λ_k = lim_{N→∞} (1/N) log σ_k(N).
```
The recurrence's two relevant solutions are read off the last column: `p_N = P_N[i,5]`,
`q_N = P_N[j,5]` for the optimal row pair `(i,j)`; the constant is `L = lim p_N/q_N`, and
`r_N = p_N/q_N`. The preprint observable is
```
δ_N = −1 − log|r_N − r_{2N}| / log|q_N|.      (★)
```

## 2. Two growth laws

**(a) Denominator growth `Q`.** The denominator is a generic combination of the columns of
`P_N`, hence grows at the dominant rate:
```
|q_N| ≍ σ₁(N) ≍ exp(N λ₁)   ⟹   Q := lim (1/N) log|q_N| = λ₁.
```

**(b) Error decay `E`.** Write `P_N`'s action in its singular basis. The ratio `r_N = p_N/q_N`
converges to `L`, and the deviation is controlled by the **gap between the two leading
directions**: the subdominant component decays relative to the dominant one like `σ₂/σ₁`, so
```
|r_N − L| ≍ σ₂(N)/σ₁(N) ≍ exp(−N(λ₁ − λ₂)).
```
Because the double-depth difference is dominated by its slower (depth-`N`) term,
```
|r_N − r_{2N}| ≍ |r_N − L| ≍ exp(−N·E),      E := λ₁ − λ₂.
```

## 3. The identity

Substitute the two growth laws into (★):
```
δ_N = −1 − log|r_N − r_{2N}| / log|q_N|
    → −1 − (−N·E)/(N·λ₁)
    = −1 + E/Q
    = −1 + (λ₁ − λ₂)/λ₁
    = −λ₂/λ₁.                              ∎
```
The float detector is *defined* as `δ̂ = −λ₂/λ₁` (ratio of the top two Lyapunov exponents
from a QR-stabilized propagation). Therefore **`δ̂` is exactly the asymptotic value of the
arithmetic delta**, which is why a `O(N·d³)` float computation predicts an
arbitrary-precision integer invariant. The finite-`N` discrepancy is the difference between
the finite-time and limiting exponents, `O(1/N)` generically — consistent with the validated
MAE ≈ 0.009 (controlled family) and the `δ_arith(N)→δ̂` convergence in the certificate.

## 4. The `δ̂ = 1` family ⟺ effective dimension 2

The Delta Ladder gives the boundary value `δ_∞ = 1/(d−1)` for an effectively `d`-dimensional
trajectory. Combining with the identity:
```
δ = −λ₂/λ₁ = 1   ⟺   λ₂ = −λ₁   ⟺   d = 2.
```
So the harvested `δ̂ = 1.000` hits (e.g. `CMF_6F5_001`, `z = −1`) are trajectories whose
asymptotics collapse to a **2-term (reciprocal-pair) recurrence**: the subdominant exponent
is the exact negative mirror of the dominant one. The certificates' depth-stability
(`δ_arith`: 1.0106 → 1.0078 → 1.0051 at `N = 120,160,240`) shows convergence **down to 1**
from above, the expected finite-`N` approach to the `d = 2` ladder rung.

## 5. Scope / hypotheses (honesty)

The derivation assumes (i) a **simple leading exponent** `λ₁` (no degeneracy `λ₁ = λ₂`), and
(ii) the optimal row pair aligns with the top-two singular directions. Both hold generically
and are confirmed numerically across the confirmed set. They **fail in the degenerate tail**
(`δ̂ ≳ 1.1`, near-zero or sign-flipped `λ₁`), exactly the 48 float artifacts rejected by exact
arithmetic — so the identity's domain of validity coincides with the detector's conservative
band. A fully formal (V5) statement would quantify (i)–(ii) as a spectral-gap condition.
