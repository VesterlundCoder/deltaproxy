# Reproducibility Package — Visible-Mode Lyapunov Geometry Paper

## Overview

This package contains all scripts and data needed to reproduce the computational
results in "Visible-Mode Lyapunov Geometry and High-Throughput Spectral Screening
of 6F5 Conservative Matrix Fields".

## Requirements

- Python 3.10+
- numpy, mpmath
- (GPU only) cupy-rocm or torch+ROCm
- tectonic (for LaTeX compilation)

## Scripts

### Benchmark
```bash
# CPU benchmark (reproducible on any machine)
python3 benchmark_proxy_vs_exact.py --dim 6 --n-candidates 5000 \
    --depth 120 --box 6 --workers 8 --out benchmark_results.json
```
Outputs: raw throughput speedup, sign-agreement, MAE, precision/recall,
end-to-end discovery speedup, cost per confirmed hit.

### Validation holdouts
```bash
python3 validation_holdout.py --dim 6 --n 2000 --depth 120 --box 6 \
    --workers 8 --out validation_results.json
```
Five independent holdout sets: shard, direction, z-value, full-dimensional,
adversarial degeneracy.

### J* estimator
```bash
# Blind full-ladder screening
python3 jstar_estimator.py --mode blind --dim 6 --n 1000 --depth 120 --box 6

# Post-hoc J* identification for a known trajectory
python3 jstar_estimator.py --mode posthoc --dim 6 --depth 120 \
    --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
    --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20

# Singular-vector visibility analysis
python3 jstar_estimator.py --mode svvis --dim 6 --depth 120 \
    --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
    --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20
```

### PSLQ sampling
```bash
python3 sample_pslq_lumi.py --input survivors_all.jsonl \
    --sample-size 500 --depth 300 --dps 200 --workers 8
```

### GPU sweep (LUMI)
```bash
# See gpu_proxy/README.md for LUMI setup
python3 gpu_proxy/gpu_sweep.py --dim 6 --total 2000000000 \
    --batch 4000000 --backend cupy --dtype float32 --thresh 0.02 \
    --out survivors_6f5.jsonl
```

## Random seeds

| Seed | Purpose |
|------|---------|
| 20260626 | LUMI 6F5 sweep 1 (200B trajectories) |
| 20270704 | LUMI 6F5 sweep 2 (200B trajectories, replication) |
| 99999 | Shard holdout validation |
| 88888 | Direction holdout validation |
| 77777 | Z-holdout validation |
| 66666 | Full-dimensional holdout validation |
| 55555 | Adversarial degeneracy validation |

## LUMI environment

- Container: PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
- Project: project_465002669
- GPU: AMD MI250X (128 GB HBM2e per GCD)
- 8 GCDs per node

## Verification ladder

| Level | Method | Status |
|-------|--------|--------|
| V0 | Float proxy flag | Automated |
| V1 | Independent integer engine | Automated |
| V2 | Exact confirmation | Automated |
| V3 | CAS certificates (mpmath, Python, SageMath) | Automated |
| V4 | Monotonic convergence check | Automated |
| V5 | Formal theorem | Open |

## Claim matrix

| Class | Claim |
|-------|-------|
| Theorem | Visible-mode delta theorem (conditional) |
| Exact identity | Wedge/cross-product formula |
| Verified computation | Exact arithmetic for specified trajectories |
| Empirical law | Proxy tracks exact delta in test population |
| Benchmark result | 10^9 trajectories/GPU-hour, ~10^4 speedup |
| Conjecture | Generic J*-CMF law |
| Conjecture | Delta Ladder: delta -> 1/(d_eff - 1) |
| Research program | Global CMF/Stokes geometry |

## File listing

```
benchmark_proxy_vs_exact.py   — Reproducible benchmark
validation_holdout.py          — Five independent holdout validations
jstar_estimator.py             — J* estimation (blind, post-hoc, SV)
sample_pslq_lumi.py            — PSLQ sampling for LUMI survivors
spectral_delta.py              — Core Lyapunov spectrum computation
cmf_generic.py                 — Dimension-generic CMF machinery
pslq_companion.py              — Exact delta + PSLQ identification
gpu_proxy/
  proxy_kernel.py              — GPU QR kernel (cupy/torch)
  gpu_sweep.py                 — Stage-1 GPU sweep driver
  calibrate.py                 — GPU calibration probe
  README.md                    — LUMI setup instructions
spectrum_diag.py               — Multi-ratio spectral diagnostics
lyapunov_delta_proxy.tex       — LaTeX paper source
lyapunov_delta_proxy.pdf       — Compiled PDF
```
