# GPU spectral-proxy sweep (Stage 1) + exact verify (Stage 2)

A two-stage discovery funnel for positive-delta CMF trajectories.

The spectral proxy `δ̂ = -λ₂/λ₁` is QR-stabilized **float** Lyapunov on a tiny
`d×d` companion cocycle. Unlike the RNS brute force it needs **no big
integers** and is embarrassingly parallel, so it maps directly onto a GPU:
**one thread per trajectory**. This screens billions; only the handful of
survivors are confirmed with the exact pure-integer engine.

```
  billions of (shift, dir, z)            ~thousands           final hits
  ───────────────────────────►  Stage 1  ──────────►  Stage 2  ─────────►
        GPU float proxy r₂        (δ̂>thr)   exact integer δ (V1)
```

## Files
- `params.py` — deterministic splitmix64 param generator. The kernel contains a
  **bit-identical** copy, so trajectories are addressed by a 64-bit `gid` only
  (no param storage/transfer); survivors are re-derived on the host.
- `proxy_kernel.py` — `ProxyKernel` (cupy `RawModule`, CUDA/HIP) with on-device
  param generation + in-register MGS-QR. `run_numpy()` is the CPU reference.
- `gpu_sweep.py` — Stage-1 driver (`--backend cupy|numpy`), writes survivors.
- `verify_survivors.py` — Stage-2 exact confirmation (`independent_delta`).
- `lumi_gpu_sweep.sbatch` — 8-GCD MI250X job (one task per GCD, disjoint gids).

## Quick local test (no GPU)
```bash
cd gpu_proxy
python3 params.py
python3 gpu_sweep.py --dim 6 --total 40000 --batch 20000 --backend numpy \
    --thresh -0.2 --out survivors.jsonl
python3 verify_survivors.py --in survivors.jsonl --dim 6 --nverify 80 \
    --workers 5 --out confirmed.jsonl
```
Expected: Stage-2 prints `sign-agree=1.000` and a small `MAE` between the proxy
`r₂` and the exact delta — the proxy is a faithful, conservative ranker.

## Run on LUMI (AMD MI250X, ROCm)
1. Build a venv with **cupy-rocm**:
   ```bash
   module load LUMI/23.09 partition/G rocm
   python3 -m venv $HOME/cmf-rocm-venv && source $HOME/cmf-rocm-venv/bin/activate
   pip install --upgrade pip
   pip install cupy-rocm-5-0 numpy mpmath   # match the loaded ROCm major version
   ```
2. **Kernel parity check** (do once on a GPU node) — confirm the on-device RNG +
   kernel match the numpy reference:
   ```bash
   python3 - <<'PY'
   from proxy_kernel import ProxyKernel, run_numpy
   import numpy as np
   k = ProxyKernel(dim=6, dtype="float64")
   r2_g,_ = k.run(seed=20260626, gid0=0, n=4096); r2_g = r2_g.get()
   r2_c,_ = run_numpy(20260626, 0, 4096, dim=6)
   m = np.isfinite(r2_g) & np.isfinite(r2_c)
   print("max|gpu-cpu| r2:", np.max(np.abs(r2_g[m]-r2_c[m])))   # expect ~1e-10 (fp64)
   PY
   ```
3. Set `--account` in `lumi_gpu_sweep.sbatch`, then `sbatch lumi_gpu_sweep.sbatch`.

## Tiers & tuning
- **fp32 screen, fp64 confirm**: `--dtype float32` for max throughput; survivors
  are re-checked exactly in Stage 2, so fp32 rounding only affects the funnel,
  never the final hits. Use `--thresh` slightly below 0 (e.g. `-0.02`) to keep
  fp32-borderline positives.
- **Throughput**: CPU numpy reference ≈ 14M traj/h/core (dim 6). On one MI250X
  GCD expect ~10⁹–10¹⁰ traj/h (custom in-register QR), i.e. **10⁴–10⁵×** the
  ~10⁵ traj/GPU-h RNS brute force. 8 GCDs/node ⇒ ~10¹⁰–10¹¹/node-h.
- **Coverage without overlap**: each GCD takes a disjoint `gid` range via
  `--seed (SEED+GCD)` and the global counter; no duplicates, full reproducibility.

## Why this is sound
- The proxy equals the asymptotic exact delta (`δ = -λ₂/λ₁`, proven + certified).
- It is **conservative** (negative bias) except the degenerate tail `δ̂≳1.1`.
- Every survivor is confirmed by the **non-circular** integer engine, so the
  GPU float math is only ever a *ranker*, never the arbiter of a hit.
