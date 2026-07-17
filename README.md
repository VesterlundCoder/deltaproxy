# Delta Proxy: visible-mode Lyapunov geometry for CMFs

This repository contains the manuscript, companion-CMF implementation, exact
integer verifier, corrected QR spectral proxy, GPU screening pipeline and local
regression data for:

> **Visible-Mode Lyapunov Geometry and High-Throughput Spectral Screening of
> 6F5 Conservative Matrix Fields**

## Current scientific status

The repository is in **pre-submission correction** status.

The conditional visible-mode theorem and the exterior-product identity are
mathematical statements. The GPU proxy is an experimental screening statistic,
not an arithmetic certificate. Historical large-sweep throughput and survivor
statistics were produced before a product-order audit and must be rerun with the
corrected implementation before they are used as submission-level evidence.

See [`SUBMISSION_READINESS_UPDATE.md`](SUBMISSION_READINESS_UPDATE.md) for the
complete correction log and mandatory rerun plan.

## Conventions fixed by the audit

### Matrix product

All code now uses

\[
P_N=M_1M_2\cdots M_N.
\]

The exact engine updates `P = P @ M_n`. The QR proxy propagates `M_n.T @ Q`,
which analyzes `P_N.T` and hence the same singular spectrum.

### Observable

Exact delta can be computed either for a fixed ordered row pair `(i,j)` or as

\[
\delta_N^{\max}=\max_{i\ne j}\delta_N(e_i^T,e_j^T)
\]

over a finite family declared in advance. The maximizing pair is returned and
must not be conflated with a fixed-observable theorem.

### Degeneracy

A finite-depth rank-loss event occurs when a numerator root

\[
f_i(n)=s_i+n d_i+1
\]

vanishes at an inspected integer step. The old `shift[i] == -1` filter was too
broad for moving roots and too weak for other integer zero crossings.

## Quick local checks

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v

python3 benchmark_proxy_vs_exact.py \
  --n-candidates 300 --n-exact 300 --depth 80 --workers 4 \
  --out benchmark_results_corrected.json

python3 validation_precision.py \
  --depth 80 --random 150 --attempts 3000 --boundary 0.05 --boundary-n 20 \
  --out precision_validation_results.json

python3 validation_holdout.py \
  --n 500 --depth 120 --workers 8 --out validation_results_corrected.json
```

## GPU preflight

```bash
cd gpu_proxy
python3 calibrate.py --dim 6 --backend torch --dtype float32 \
  --nrank 120 --batches 200000,1000000,4000000 --repeats 5 \
  --out gpu_calibration_r2.json

python3 calibrate.py --dim 6 --backend torch --dtype float32 \
  --nrank 120 --batches 200000,1000000,4000000 --repeats 5 \
  --full-ladder --out gpu_calibration_full_ladder.json
```

The full-ladder index is named `k_candidate`; it is **not** an independent
estimate of `J*`.

## Repository map

- `lyapunov_delta_proxy.tex` — revised manuscript source.
- `cmf_generic.py` — companion matrices, exact right products, fixed-pair and
  finite-family deltas, finite-depth degeneracy checks.
- `spectral_delta.py` — corrected right-product QR spectrum.
- `proxy_batch.py` — batched NumPy/Torch-compatible spectrum.
- `benchmark_proxy_vs_exact.py` — matched local CPU benchmark.
- `validation_precision.py` — float32/float64 versus exact validation.
- `validation_holdout.py` — structurally disjoint holdouts.
- `jstar_estimator.py` — candidate-mode and post-hoc diagnostics with explicit
  epistemic warnings.
- `gpu_proxy/` — GPU kernel, sweep driver, exact verification and LUMI scripts.
- `tests/` — product-order, observable, degeneracy and GPU/CPU parity tests.
- `results/regression/` — small corrected local results; not production evidence.

## Reproducibility and archival policy

The final submission release should contain or reference, by persistent DOI:

- matched corrected GPU/RNS benchmark logs;
- float32 and float64 precision-validation outputs;
- survivor and all-verification manifests;
- deduplicated unique-trajectory counts;
- job metadata and exact git commit;
- SHA-256 checksum manifest;
- completed rather than ongoing sweep results.

## License

MIT. See [`LICENSE`](LICENSE).
