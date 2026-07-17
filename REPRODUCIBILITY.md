# Reproducibility Package

## Repository

**GitHub:** https://github.com/VesterlundCoder/deltaproxy  
**License:** MIT  
**Version:** v1.0 (July 2026)

## Software Requirements

- Python 3.10+
- numpy, mpmath
- (GPU) cupy-rocm or torch+ROCm (LUMI PyTorch container)
- tectonic (LaTeX compilation)

## Code Inventory

| File | Purpose |
|------|---------|
| `cmf_generic.py` | Generic CMF construction, exact integer delta |
| `spectral_delta.py` | Lyapunov spectrum computation (QR-based) |
| `proxy_batch.py` | Batched proxy computation |
| `gpu_proxy/proxy_kernel.py` | GPU proxy kernel (cupy/torch/numpy) |
| `gpu_proxy/gpu_sweep.py` | Stage-1 GPU screening driver |
| `gpu_proxy/params.py` | Deterministic parameter generation (bit-identical to GPU) |
| `gpu_proxy/calibrate.py` | GPU calibration and parity checks |
| `benchmark_proxy_vs_exact.py` | Reproducible benchmark: proxy vs exact |
| `validation_holdout.py` | Five independent holdout validations |
| `jstar_estimator.py` | J* estimator (blind, post-hoc, SV-based) |
| `pslq_companion.py` | PSLQ integer relation identification |
| `verify_survivors.py` | Stage-2 exact verification |
| `sweep_zpm1.py` | Local z=+1/-1 sweep driver |
| `deploy_zpm1_full.sh` | LUMI deployment (rsync + SLURM) |
| `gpu_proxy/lumi_zpm1_positive_669.sbatch` | SLURM script for z=+1/-1 sweep |
| `gpu_proxy/lumi_sweep_669.sbatch` | SLURM script for general sweeps |

## Reproducible Commands

### CPU Benchmark (any machine)
```bash
python3 benchmark_proxy_vs_exact.py --dim 6 --n-candidates 2000 \
    --depth 120 --box 6 --workers 8 --out benchmark_results.json
```

### Validation Holdouts
```bash
python3 validation_holdout.py --dim 6 --n 500 --depth 120 --box 6 \
    --workers 8 --out validation_results.json
```

### J* Estimator
```bash
python3 jstar_estimator.py --mode posthoc --dim 6 --depth 120 \
    --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
    --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20
```

### GPU Sweep (LUMI)
```bash
bash deploy_zpm1_full.sh sync     # rsync code
bash deploy_zpm1_full.sh launch   # submit 16 waves
bash deploy_zpm1_full.sh status   # check queue
bash deploy_zpm1_full.sh fetch    # download survivors
```

## LUMI Environment

- **Project:** project_465002669
- **Container:** lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1
- **Hardware:** AMD MI250X, 64 GB HBM2e per GCD, 2 GCDs per module
- **Billing:** 1 GPU-hour per MI250X module (2 GCDs)
- **Throughput:** ~301k trajectories/s/GCD (float32, dim=6, N=120)

## Seeds

| Seed | Purpose |
|------|---------|
| 20260626 | LUMI 6F5 sweep 1 (200B trajectories) |
| 20270704 | LUMI 6F5 sweep 2 (200B trajectories) |
| 99999 | Shard holdout validation |
| 20260717 | z=+1/-1 boundary sweep |

## Data Availability

All trajectories are re-derivable from (seed, gid) using the deterministic parameter generator in `gpu_proxy/params.py`, which is bit-identical to the on-device GPU generator in `proxy_kernel.py`. Survivor manifests and verification certificates are available in the repository.

## Citation

```bibtex
@software{deltaproxy2026,
  author = {Vesterlund, David},
  title = {Delta Proxy: Visible-Mode Lyapunov Geometry and Spectral Screening for CMFs},
  url = {https://github.com/VesterlundCoder/deltaproxy},
  year = {2026}
}
```
