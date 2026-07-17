# pFq / Companion CMF Sweeps — Spectral Dynamics Report

**Scope.** This documents, in one place, (1) exactly how the trajectory-step
matrices are built for both matrix constructions, (2) exactly how the delta
proxy is defined and used, (3) the consolidated results of all recent LUMI
sweeps, (4) why the two 200B 6F5 runs returned nearly identical survivor counts,
and (5) a multi-ratio spectral diagnostic (λ₂/λ₁, λ₃/λ₁, …) across 4F3/5F4/6F5/8F7.

The goal here is to understand the **dynamics**, not to assemble a theory. Every
claim below is either a definition (code) or a measured number (a run). Caveats
are flagged explicitly.

Generated: 2026-07-05. Author context: D. Vesterlund, project_465002669 (LUMI).

---

## 1. The two matrix constructions

Both are `dim × dim` real cocycles `M(n)` whose infinite left-product defines the
continued-fraction / CMF limit. The finite-time Lyapunov spectrum of the product
is the object we screen. **They are different families and must not be conflated.**

### 1.1 Companion (elementary-symmetric) matrix — `proxy_batch.build_M_batch`

This is the construction used by the production **6F5 and 8F7** sweeps. It is the
companion matrix of a scalar recurrence whose coefficients are elementary
symmetric polynomials of two shifted arithmetic ladders.

For a trajectory with integer `shift ∈ [-box,box]^{2·dim-1}`, direction `dir`
(from `cg.dir_pool`), and `z = z_num/z_den`, at depth `n`:

```
f_i = shift[i]      + n·dir[i]      + 1     (i = 0 .. dim-1)      # dim terms
g_j = shift[dim+j]  + n·dir[dim+j]  + 2     (j = 0 .. dim-2)      # dim-1 terms
ef  = esym(f)   (elementary symmetric polys, length dim+1)
eg  = esym(g)   (padded to length dim+1)
h_f = -z_num ,  h_g = z_den
c[0]   = h_f · ef[dim]
c[k]   = h_g · eg[dim-k] + h_f · ef[dim-k]        (k = 1 .. dim-1)
```

`M` is the companion form: sub-diagonal ones (`M[r, r-1] = 1`) and last column `c`.
This is the GENERIC family — most sampled `shift` vectors yield a recurrence that
is **not** hypergeometric; the search space is broad.

Parameter draw for a global index `gid`: splitmix64 counter RNG
(`params.gen_params`) produces `shift`, a `dir` index into `cg.dir_pool(dim)`, and
a `z` index into `cg.z_pool` — identical on host (re-derivation) and device
(kernel), so nothing but the `gid` is stored.

### 1.2 pFq (exact hypergeometric) matrix — `pfq_gpu.PFqBuilder`

This is the construction used by the production **4F3 and 5F4** sweeps. It is the
EXACT RamanujanTools `pFq(p, q)` CMF, with `q = p-1` so `N = dim = p`
(`4F3 → pFq(4,3)`, `5F4 → pFq(5,4)`, `6F5 → pFq(6,5)`, `8F7 → pFq(8,7)`).

We do **not** re-derive the recurrence. We take RamanujanTools' own per-axis
matrices `M(axis, sign) = cmf.M(axis, sign)` (sympy, in `x_i, y_j, z`), and
`lambdify` each entry using only `+ - * / **`. The diagonal-trajectory step matrix
at base position `P` is the ordered product over sorted axes, exactly as
`_calculate_diagonal_matrix_backtrack` computes it:

```
M_step(P) = Π_{axis ∈ sorted(axes)}  M(axis, sign_axis)   (each factor shifts its
                                                            own coord afterwards)
```

The per-axis symbolic grids are pre-generated once (`generate_defs`) into
`pfq_matrix_defs.json` (currently holds (2,1),(3,2),(4,3),(5,4),(6,5),(8,7)) so the
LUMI container needs only numpy/torch — no ramanujantools at runtime. The builder
is validated entrywise against `cmf.walk` in `pfq_gpu.selftest` (rel-err < 1e-7).

Parameter mapping (`map_shift_to_xy`) places starts in **disjoint** positive
ranges so `x_i ≠ y_j` (no spurious poles):

```
x_i ∈ [2, 2+2·box] ,   y_j ∈ [3·box+4, 4·box+4]
```

Trajectory patterns come from `pfq_dir_pool(dim)`: the all-ones diagonal plus each
single axis dropped (forward `{0,1}` steps).

### 1.3 Why they differ

The companion family is a *generic* companion parametrised by arbitrary shift
ladders; the pFq family is the *specific* structured hypergeometric CMF. They
coincide only when the sampled recurrence is itself hypergeometric — a measure-zero
event under generic shift sampling. **Consequence:** "6F5 companion" and
"pFq(6,5)" have genuinely different Lyapunov geometry and different survivor
statistics (Section 5 quantifies this).

---

## 2. The delta proxy

### 2.1 Definition (certified identity)

The irrationality-measure quantity is
```
δ = -1 + E/Q = -λ₂/λ₁                       (Lyapunov form)
```
where `λ₁ ≥ λ₂ ≥ … ≥ λ_dim` is the finite-time Lyapunov spectrum of the cocycle.
The float detector is
```
δ̂ = r₂ = -λ₂/λ₁ ,   λ = lyapunov_spectrum(M, dim, N)   (float64, QR-stabilised)
```
`δ̂` is a conservative (negative-biased) estimator of the exact asymptotic δ,
**except** in the degenerate tail `δ̂ ≳ 1.05` (λ₂ ≈ -λ₁, block collapse / pole),
which must always be exact-verified.

### 2.2 QR-stabilised spectrum (`spectral_delta.lyapunov_spectrum`, `pfq_gpu._spectrum_pfq`, `proxy_batch.lyapunov_batch`)

`λ_i = (1/N) Σ_n log|R_ii|` for `P = M(0)…M(N-1)`, with per-step QR
renormalisation (the product is never materialised), stable to arbitrary depth.
On ROCm the batched QR uses a **manual modified Gram–Schmidt over columns** because
`torch.linalg.qr` (rocSOLVER) is pathologically slow (18-min hangs). The manual
MGS matches numpy to ~8e-16. Production depth: `N = 120` (pFq) / sweep default
(companion).

### 2.3 Full subdominant ladder (`spectral_ratios`, `pfq_gpu.lyapunov_ladder_pfq`, `ratios_batch`)

```
r_k = -λ_k / λ₁ ,   k = 2, 3, …, dim
```
`r₂` is the standard detector. The deeper ratios `r₃, r₄, …` were added to
investigate whether barren `r₂` hides a positive deeper mode. **Caveat (critical):
only `r₂` is tied to the certified identity δ = -1 + E/Q.** Deeper-mode positivity
describes internal cocycle structure (sub-dominant convergents) and is **not** an
irrationality signal unless separately verified. `spectral_dhat_robust` will only
fall through to `r₃, r₄, …` when `r₂` is *degenerate* (|r₂| ≥ 1.05); a merely
negative `r₂` is reported as-is.

### 2.4 Two-stage funnel

- **Stage 1 (GPU proxy):** screen billions of `gid`s; keep `δ̂ > thresh` survivors
  (parameters re-derivable from `gid`, so only survivors are written).
- **Stage 2 (exact):** `cmf_generic.independent_delta` — non-circular pure-integer
  double-depth `2N` delta — confirms each survivor. This is the same V1 engine that
  certified the original 6F5 hits.

---

## 3. Consolidated run results

| Run | Matrix | dim | Trajectories | Survivors (δ̂>thresh) | max δ̂ | Funnel | Notes |
|---|---|---|---|---|---|---|---|
| 6F5 batch 1 (job 19558891) | companion | 6 | 200B | 559,605 | — | 2.80e-6 | thresh 0.02, box 6, fp32 |
| 6F5 **v2** (job 19728166) | companion | 6 | 200B | 559,530 | +22.60 | 2.80e-6 | fresh seed; disjoint from batch 1 |
| 5F4 (job 19723897) | pFq | 5 | ~117B | **0** | **−0.0165** | 0 | fully barren; max δ̂ negative |
| 4F3 (job 19723871) | pFq | 4 | ~317B | **431,873,878** | +360,928 | 1.36e-3 | not a bug — see §5.3 |
| 8F7 (earlier, 500B) | companion | 8 | 500B | 0 (thresh) | ~+0.003 | ~0 | barren |

### 3.1 6F5 v2 dedup + exact verify (this session)

- **Dedup vs batch 1:** 559,530 v2 survivors → **559,417 genuinely new** (only 113
  overlap). The seed change (`20260626 → 20270704`) produced near-disjoint coverage.
- **Stage-2 exact verify (559,417 new, depth 2N=200):**
  - **551,149 confirmed** with exact δ > 0 (**98.5%**).
  - **Proxy vs exact: sign-agree = 0.996, MAE = 0.0497** — the `−λ₂/λ₁` proxy is
    highly faithful.
  - Top exact deltas cluster at **+1.01 with r₂ = +1.0000** — the degenerate tail
    (λ₂ ≈ −λ₁, block collapse), the same artifact family as before, not new hits.

---

## 4. Why the two 200B 6F5 runs gave (almost) the same count

This is **expected sampling behaviour**, not a hidden geometric connection.

Each run draws `N = 2×10¹¹` i.i.d. trajectories from the *same* finite discrete
parameter space (same box, z-pool, dir-pool) and applies the *same* threshold. The
survivor count is therefore `Binomial(N, p)` where `p` is the intrinsic hit
probability of (companion-6F5, box 6, z-pool, thresh 0.02).

```
pooled p            = 2.798e-6
Binomial mean N·p   = 559,568
Binomial sigma      = sqrt(N·p·(1-p)) = 748.0
observed |c1 - c2|  = |559,605 - 559,530| = 75  = 0.10 sigma
c1, c2 vs mean      = +0.05 sigma, -0.05 sigma
relative spread     = 1.3e-4   (~ 1/sqrt(mean) scale)
```

Two independent 200B samples of the same distribution agree to **0.10σ** — exactly
what the Law of Large Numbers predicts. `p` is a well-defined invariant of
(family + sampling box + threshold); `N` is astronomical, so `N·p` is pinned down
tightly. The near-equality confirms the sampler is unbiased and the geometry is
stationary — it does **not** imply any deeper coincidence. (If the counts had
differed by ≫ a few σ, *that* would have been the anomaly worth investigating.)

---

## 5. Multi-ratio spectral diagnostic (4F3 / 5F4 / 6F5 / 8F7)

Tool: `spectrum_diag.py` (`--matrix pfq|companion`). It samples 40,000 trajectories
from the identical distribution as each sweep, at depth 200, and records the full
ladder `r_2…r_N`. Raw arrays in `spectrum_diag_out/{fam}_{matrix}_ladder.npz`;
summaries in `spectrum_diag_summary{,_companion}.json`.

Counts below are **out of 40,000**. `r₂>0.02` is the certified survivor signal;
`r₃>0.02` and "rescue" (a dead `r₂` with some positive deeper mode) are diagnostic
only.

| family | matrix | N | r₂>0 | r₂>0.02 | r₂ degen | r₂ max | r₃>0.02 | deeper rescue |
|---|---|---|---|---|---|---|---|---|
| 4F3 | pFq | 4 | 51 | **46** | 2 | +2.698 | 7465 | 16311 |
| 5F4 | pFq | 5 | 0 | **0** | 0 | −0.068 | 13 | 21640 |
| 6F5 | pFq | 6 | 0 | **0** | 0 | −0.136 | 0 | 17356 |
| 8F7 | pFq | 8 | 0 | **0** | 0 | −0.591 | 0 | **0** |
| 4F3 | companion | 4 | 260 | **119** | 0 | +0.633 | 4411 | 22054 |
| 5F4 | companion | 5 | 17 | **6** | 0 | +0.097 | 387 | 22388 |
| 6F5 | companion | 6 | 1 | **0** | 0 | +0.008 | 27 | 21971 |
| 8F7 | companion | 8 | 0 | **0** | 0 | −0.042 | 1 | 21041 |

### 5.1 The primary signal (r₂) collapses monotonically with dimension

The rate of the *certified* signal `r₂ > 0.02` falls off fast with `dim`, for BOTH
constructions:

```
companion:  4F3 ~ 3.0e-3   5F4 ~ 1.5e-4   6F5 ~ (0 in 40k; 2.8e-6 from 200B)   8F7 ~ 0
pFq:        4F3 ~ 1.2e-3   5F4 ~ 0         6F5 ~ 0                              8F7 ~ 0
```

`r₂ > 0` requires `λ₂ < 0`: a single dominant expanding direction with the rest
contracting. As dimension grows, the cocycle typically develops **multiple positive
Lyapunov exponents** (`λ₂ ≥ 0`), so `r₂ = -λ₂/λ₁ ≤ 0` almost everywhere. This is the
concrete dynamical reason 5F4/8F7 come back barren: it is not a defect in the proxy
or the pipeline — it is that the positive-δ configuration becomes an exponentially
rare corner of the geometry as `N` increases.

**On "non-universality":** the J* geometry is universal in the *mechanism* (δ =
−λ₂/λ₁ everywhere), but the *measure* of the positive-δ set is strongly
dimension-dependent and construction-dependent. Companion is uniformly richer than
pFq at equal dimension (e.g. 5F4: 6 vs 0 hits per 40k) because it samples a broader,
less rigid family.

### 5.2 The 6F5 companion sweep is consistent with this diagnostic

6F5 companion shows ~0 hits in 40k (max r₂ = +0.008 < 0.02). The 200B sweep found
559k because `p ≈ 2.8e-6` and `2×10¹¹ × 2.8e-6 ≈ 5.6×10⁵`. The richness of the
survivor file is a **rare event × astronomical sample size**, not a dense signal.
A 40k probe cannot see it; only the full sweep can. This fully reconciles the
diagnostic with the sweep.

### 5.3 The 4F3 "flood" (431M) is mostly genuine, not degeneracy

Earlier reading (that 431M were degenerate artifacts) is **corrected here.** The pFq
4F3 hit rate `r₂ > 0.02` is ~1.2e-3 → `317×10⁹ × 1.2e-3 ≈ 3.6×10⁸`, matching the
observed 431,873,878 (funnel 1.36e-3). So the flood is dominated by *genuine
small-positive* `r₂` (dim-4 simply has a high hit rate), with only a thin degenerate
tail (the `max δ̂ = +360,928` is a rare pole, `λ₁ → 0`). The problem with 4F3 is
therefore **volume, not validity**: 4×10⁸ exact verifications are infeasible, so 4F3
needs a tighter band (e.g. `0.02 < δ̂ < 1.05`, plus a higher floor) to become a
verifiable candidate set — not because the hits are fake.

### 5.4 Deeper modes carry positive structure — but are NOT δ

`r₃`/`r₄` are positive in ~50% of trajectories for dims 4–6 (both matrices), and a
dead `r₂` is "rescued" by a positive deeper mode in ~40–56% of cases. **This is a
geometric observation only.** Deeper ratios measure sub-dominant convergence gaps,
not the primary CMF limit's irrationality measure; a positive `r₃` is not an Apéry-
type δ. Notably, **pFq(8,7) is spectrally dead across the entire ladder** (rescue =
0, even `r₃>0.02 = 0`): its whole subdominant spectrum is non-positive/degenerate in
this sample — markedly more rigid than the generic companion at the same dimension.

---

## 6. Open questions (to look into — deliberately not answered here)

1. **Is the positive-δ measure decay in `dim` a power law or exponential?** A clean
   `p(dim)` curve for the companion family (say `dim = 3..8`, fixed box/z, large
   samples) would quantify it. Current points: 4F3 ~3e-3, 5F4 ~1.5e-4, 6F5 ~2.8e-6.
2. **Does box / z-region shift the positive-δ measure?** All numbers above are box 6,
   |z|<1. Whether a different z-band re-opens 5F4/8F7 is untested.
3. **pFq vs companion gap at fixed dim.** Companion is ~20–∞× richer per sample. Is
   that purely the "generic vs hypergeometric" volume effect, or is the pFq gauge
   actively suppressing λ₂ < 0?
4. **Deeper-mode positivity — is any of it a genuine convergent?** Would require a
   dedicated exact check on `r₃`-driven candidates; unknown whether any survive.

## 7. Reproduce

```bash
cd 6F5Sweeps/continuous_cmf
# multi-ratio diagnostic (both constructions)
python3 spectrum_diag.py --family all --nsamp 40000 --depth 200 --matrix pfq
python3 spectrum_diag.py --family all --nsamp 40000 --depth 200 --matrix companion
# exact verify of any survivor file
python3 gpu_proxy/verify_survivors.py --in <survivors>.jsonl --dim <D> --workers 8
```
