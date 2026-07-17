# Reproducibility package - corrected pre-submission branch

## Status

This package records the product-order, observable and degeneracy corrections
found during the submission audit. Small local regressions are included. The
historical large GPU sweeps and the historical RNS comparison are **legacy
results** until rerun with the corrected implementation.

## Mathematical/computational conventions

### Product order

\[
P_N=M_1M_2\cdots M_N.
\]

- Exact integer and multiprecision engines update `P = P @ M_n`.
- QR propagates `M_n.T @ Q`, analyzing
  `P_N.T = M_N.T ... M_1.T` and therefore the same singular spectrum.

### Observables

The exact engine supports:

1. one fixed ordered row pair `(i,j)`;
2. the explicitly declared finite-family statistic
   \(\delta_N^{\max}=\max_{i\ne j}\delta_N(e_i^T,e_j^T)\).

The maximizing pair is saved. The second quantity is not silently treated as a
fixed-observable theorem.

### Finite-depth rank loss

A trajectory is marked singular in the inspected range if

\[
f_i(n)=s_i+n d_i+1=0
\]

for some numerator root and integer step. This replaces the incorrect blanket
filter `shift[i] == -1`.

## Requirements

```text
Python >= 3.10
numpy >= 1.24
mpmath >= 1.3
```

GPU runs use the ROCm/PyTorch or CuPy stack supplied by the target LUMI
container.

## CPU regression suite

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

The suite checks product-order consistency, fixed versus maximum observables,
dynamic numerator-root zero detection, and float32/float64 parity.

## Corrected local benchmark

```bash
python3 benchmark_proxy_vs_exact.py \
  --n-candidates 300 --n-exact 300 --depth 80 --workers 4 \
  --out benchmark_results_corrected_filtered.json
```

The included small CPU regression used 163 production-eligible exact cases,
with sign agreement 1.000 and MAE 0.00792. No positives occurred, so precision,
recall and cost per hit are not estimable. The comparator is pure Python integer
arithmetic, not GPU RNS.

## Float32/float64 versus exact validation

```bash
python3 validation_precision.py \
  --depth 50 --random 100 --attempts 3000 \
  --boundary 0.15 --boundary-n 20 \
  --out precision_validation_results_corrected.json
```

Local regression results:

| Stratum | n | float32 sign | float32 MAE | false negatives |
|---|---:|---:|---:|---:|
| Generic eligible | 100 | 1.000 | 0.01059 | 0 |
| Boundary | 17 | 1.000 | 0.01010 | 0 |
| Known nonsingular positive family | 5 | 1.000 | 0.03530 | 0 |

These are code-regression results at depth 50. Submission requires a larger,
production-depth GPU validation enriched near delta=0.

## Corrected holdouts

```bash
python3 validation_holdout.py \
  --n 500 --depth 120 --workers 8 \
  --out validation_results_corrected.json
```

A smaller completed regression at depth 60 is included in the correction bundle.
All eligible cases were negative; sign agreement was 1.000. The small-gap set
had the largest error, as expected.

## Apéry and reference-trajectory validation

```bash
python3 validate_apery.py --depths 100,500,1000
python3 validate_reference_trajectories.py --depths 40,80,120
```

The corrected Apéry calibration reproduces the expected convergence. For the
nonsingular Hit-B trajectory at depth 120, the corrected finite-family statistic
is approximately 0.143192 with maximizing pair `(1,3)`. The closest spectral
ratio varies with depth and is not interpreted as an independent J* estimate.

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

The output must distinguish GCD-hours from MI250X-module-hours.

## Corrected sweep and verification

```bash
python3 gpu_sweep.py --dim 6 --backend torch --dtype float32 \
  --total 2000000000 --gid0 0 --batch 4000000 --nrank 120 \
  --require-pos-lam1 --no-degenerate --out survivors.jsonl

python3 verify_survivors.py --in survivors.jsonl --dim 6 --nverify 120 \
  --workers 8 --all-out verified_all.jsonl --out confirmed.jsonl
```

Use `--full-ladder` only as a high-recall candidate generator. Its
`k_candidate` field is not J*.

## Sampling and resource arithmetic

The corrected campaign design uses one fixed seed and disjoint absolute gid
ranges. SplitMix still samples parameter space with replacement; it is not a
bijective enumeration.

For the proposed z=+1/-1 campaign:

- finite parameter count: `13^11 * 19 * 2 = 68,102,094,973,406`;
- proposed draws: `13,312,000,000,000`;
- draw/cardinality ratio: about `19.55%`;
- ideal iid expected unique fraction: about `17.76%`;
- resources: `24,576 GCD-hours = 12,288 MI250X-module-hours`.

These are projections, not completed results and not certified coverage.

## Required submission reruns

The following corrected artifacts are still required:

1. Repeated synchronized LUMI throughput logs for r2-only and full-ladder modes.
2. A matched GPU proxy versus GPU RNS run on identical candidates and depth.
3. Production-depth float32 versus exact validation including substantial positive strata.
4. Corrected large screening runs and complete `verified_all.jsonl` outputs.
5. Deduplicated unique-trajectory counts and overlap analysis.
6. Recomputed enrichment, end-to-end speedup and cost per confirmed hit.
7. Public survivor manifests, exact certificates, raw logs and SHA-256 manifests.
8. Independent observable-level visibility analysis for any claimed J*>2.

Historical large-sweep statistics must not be used as final evidence without a
corrected rerun or a demonstrated equivalence to the corrected product.

## Archival release

Before submission:

1. freeze corrected production data;
2. tag a release;
3. create SHA-256 manifests;
4. archive code, source, logs and result manifests on Zenodo or OSF;
5. insert the DOI and release commit in the manuscript;
6. obtain PI or external review of the theorem and observable setup.
