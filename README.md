# Delta Proxy: Visible-Mode Lyapunov Geometry and Spectral Screening for Conservative Matrix Fields

## Overview

This repository contains the complete code, data, and reproducibility package for the paper:

> **Visible-Mode Lyapunov Geometry and High-Throughput Spectral Screening of 6F5 Conservative Matrix Fields**

The paper presents a spectral proxy method for screening irrationality candidates in high-dimensional Conservative Matrix Fields (CMFs). The key insight is that the arithmetic convergence quality of a matrix-cocycle-generated approximation system is determined not by the second Lyapunov exponent alone, but by the first Lyapunov mode visible to the projective observable.

## Repository Structure

```
deltaproxy/
├── gpu_proxy/                    # GPU proxy kernel and sweep drivers
│   ├── proxy_kernel.py           # cupy/torch/numpy proxy kernel (QR-based Lyapunov)
│   ├── gpu_sweep.py              # Stage-1 GPU screening driver
│   ├── params.py                 # Deterministic parameter generation (matches on-device)
│   ├── calibrate.py              # GPU calibration and parity checks
│   ├── lumi_zpm1_positive_669.sbatch  # LUMI SLURM script for z=+1/-1 sweep
│   ├── lumi_sweep_669.sbatch     # LUMI SLURM script for general sweeps
│   └── README.md                 # GPU proxy documentation
├── cmf_generic.py                # Generic CMF construction (companion matrix cocycle)
├── spectral_delta.py             # Exact arithmetic delta computation
├── proxy_batch.py                # Batched proxy computation (numpy/torch)
├── benchmark_proxy_vs_exact.py   # Reproducible benchmark: proxy vs exact RNS
├── validation_holdout.py         # Independent validation holdout sets
├── jstar_estimator.py            # Practical J* estimator (blind, post-hoc, SV-based)
├── sweep_zpm1.py                 # Local z=+1/-1 sweep driver
├── deploy_zpm1_full.sh           # LUMI deployment script (rsync + launch)
├── pslq_companion.py             # PSLQ integer relation identification
├── verify_survivors.py           # Stage-2 exact verification of survivors
├── enrich.py                     # Enrichment factor measurement
├── harvest.py                    # Discovery run with checkpointing
├── filter_survivors.py           # Post-sweep filtering utilities
├── lyapunov_delta_proxy.tex      # Main paper (LaTeX source)
├── REPRODUCIBILITY.md            # Full reproducibility documentation
└── benchmark_results.json        # Benchmark output (CPU reference)
```

## Requirements

- Python 3.10+
- numpy, mpmath
- (GPU only) cupy-rocm or torch+ROCm
- tectonic (for LaTeX compilation)

## Quick Start

### CPU Benchmark (reproducible on any machine)

```bash
python3 benchmark_proxy_vs_exact.py --dim 6 --n-candidates 5000 \
    --depth 120 --box 6 --workers 8 --out benchmark_results.json
```

### Validation Holdouts

```bash
python3 validation_holdout.py --dim 6 --n 2000 --depth 120 --box 6 \
    --workers 8 --out validation_results.json
```

### J* Estimator

```bash
# Post-hoc analysis (requires exact delta)
python3 jstar_estimator.py --mode posthoc --dim 6 --depth 120 \
    --shift '[-2,-2,0,0,-2,0,-2,-2,-2,0,-2]' \
    --dir '[0,0,0,0,0,0,0,0,0,1,0]' --z-num 7 --z-den 20

# Blind full-ladder screening
python3 jstar_estimator.py --mode blind --dim 6 --depth 120 --box 6
```

### GPU Sweep (LUMI)

```bash
# Sync code to LUMI
bash deploy_zpm1_full.sh sync

# Launch production sweep
bash deploy_zpm1_full.sh launch

# Check status
bash deploy_zpm1_full.sh status

# Fetch results
bash deploy_zpm1_full.sh fetch
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{deltaproxy2026,
  author = {Vesterlund, David},
  title = {Delta Proxy: Visible-Mode Lyapunov Geometry and Spectral Screening for CMFs},
  url = {https://github.com/VesterlundCoder/deltaproxy},
  year = {2026}
}
```

## Contact

David Vesterlund -- https://github.com/VesterlundCoder
