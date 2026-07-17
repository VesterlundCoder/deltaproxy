"""
calibrate.py
============
GPU calibration probe -- run on ONE GCD before committing big allocation.

Reports, for the chosen backend (torch/cupy) and dtype:
  1. PARITY: max|GPU - numpy_reference| on r_2 over a sample (must be tiny / sign-stable),
  2. THROUGHPUT: trajectories/second at several batch sizes,
  3. PROJECTION: trajectories per GCD-hour and per 8-GCD node-hour, and the wall
     time implied for a target sweep size.

Usage (inside the ROCm container on a GPU node):
  python3 calibrate.py --dim 6 --backend torch --dtype float32 --target 200e9
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from proxy_kernel import run_numpy


def measure(backend, seed, n, dim, nrank, box, dtype, device, matrix="companion"):
    if backend == "torch":
        from proxy_kernel import run_torch, run_torch_pfq
        fn = run_torch_pfq if matrix == "pfq" else run_torch
        # warm-up (kernel/cuDNN init)
        fn(seed, 0, min(4096, n), dim, nsteps=nrank, box=box,
           device=device, dtype=dtype)
        import torch
        torch.cuda.synchronize()
        t0 = time.time()
        r2, l1 = fn(seed, 0, n, dim, nsteps=nrank, box=box,
                    device=device, dtype=dtype)
        torch.cuda.synchronize()
        dt = time.time() - t0
    elif backend == "cupy":
        from proxy_kernel import ProxyKernel
        k = ProxyKernel(dim, nsteps=nrank, box=box, dtype=dtype)
        k.run(seed, 0, min(4096, n))
        import cupy as cp
        cp.cuda.Device().synchronize()
        t0 = time.time()
        r2d, l1d = k.run(seed, 0, n)
        cp.cuda.Device().synchronize()
        dt = time.time() - t0
        r2, l1 = r2d.get(), l1d.get()
    else:
        from proxy_kernel import run_numpy_pfq
        fn = run_numpy_pfq if matrix == "pfq" else run_numpy
        t0 = time.time()
        r2, l1 = fn(seed, 0, n, dim, nsteps=nrank, box=box)
        dt = time.time() - t0
    return r2, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--backend", choices=["torch", "cupy", "numpy"], default="torch")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--matrix", choices=["companion", "pfq"], default="companion")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--target", type=float, default=200e9,
                    help="target sweep size for the wall-time projection")
    ap.add_argument("--batches", default="200000,1000000,4000000")
    args = ap.parse_args()

    print(f"=== CALIBRATION dim={args.dim} backend={args.backend} matrix={args.matrix} "
          f"dtype={args.dtype} N={args.nrank} box=+-{args.box} ===")

    # 1) parity vs numpy reference (fp64 reference)
    npar = 4000
    r2_gpu, _ = measure(args.backend, args.seed, npar, args.dim, args.nrank,
                        args.box, args.dtype, args.device, args.matrix)
    if args.matrix == "pfq":
        from proxy_kernel import run_numpy_pfq
        r2_ref, _ = run_numpy_pfq(args.seed, 0, npar, args.dim, nsteps=args.nrank, box=args.box)
    else:
        r2_ref, _ = run_numpy(args.seed, 0, npar, args.dim, nsteps=args.nrank, box=args.box)
    m = np.isfinite(r2_gpu) & np.isfinite(r2_ref)
    d = np.abs(r2_gpu[m] - r2_ref[m])
    sign = np.mean((r2_gpu[m] > 0) == (r2_ref[m] > 0))
    print(f"\n[parity] vs numpy fp64 over {m.sum()} finite traj:")
    print(f"  median={np.median(d):.2e}  p99={np.percentile(d,99):.2e}  "
          f"max={d.max():.2e}  sign-agree={sign:.4f}")
    if sign < 0.999:
        print("  !! WARNING: sign agreement < 0.999 -- investigate before big run")

    # 2) throughput at several batch sizes
    print(f"\n[throughput]")
    best = 0.0
    for b in [int(float(x)) for x in args.batches.split(",")]:
        try:
            _, dt = measure(args.backend, args.seed, b, args.dim, args.nrank,
                            args.box, args.dtype, args.device, args.matrix)
        except Exception as e:
            print(f"  batch={b:,}: FAILED ({e})")
            continue
        thr = b / dt
        best = max(best, thr)
        print(f"  batch={b:>10,}: {dt:6.2f}s -> {thr:>12,.0f} traj/s  "
              f"({thr*3600:,.0f}/GCD-h)")

    # 3) projection
    print(f"\n[projection] best {best*3600:,.0f}/GCD-h  "
          f"-> {best*3600*8:,.0f}/node-h (8 GCDs)")
    if best > 0:
        gcd_h = args.target / (best * 3600)
        node_h = gcd_h / 8
        print(f"  target {args.target:.3g} traj: {gcd_h:,.1f} GCD-h "
              f"= {node_h:,.1f} node-h on 1 node "
              f"(~{node_h:.1f}h wall, or {node_h/2:.1f}h on 2 nodes)")


if __name__ == "__main__":
    main()
