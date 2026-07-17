#!/usr/bin/env python3
"""
sweep_zpm1.py
==============
Local driver for z=±1 6F5 delta proxy sweeps.

Screens trajectories with z restricted to +1 and -1 only.
Can run on CPU (numpy) for testing or GPU (cupy/torch) for production.

The z=±1 regime is the boundary case where the gauge factor equals ±1.
Most trajectories have delta near 0 (polynomial convergence), but the
positive tail may contain interesting structure.

Usage:
  # CPU test (100k trajectories, ~2 min)
  python3 sweep_zpm1.py --total 100000 --backend numpy --workers 8

  # Local GPU (if available)
  python3 sweep_zpm1.py --total 10000000 --backend cupy --dtype float32

  # Multi-billion: use the SLURM script instead
  # sbatch gpu_proxy/lumi_zpm1_positive_669.sbatch
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_proxy"))
import cmf_generic as cg
from params import gen_params, build_pools


def main():
    ap = argparse.ArgumentParser(description="z=±1 6F5 delta proxy sweep")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="keep r2 > thresh (0.0 = all non-negative)")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--backend", choices=["cupy", "torch", "numpy"], default="numpy")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="survivors_zpm1.jsonl")
    ap.add_argument("--workers", type=int, default=8,
                    help="CPU workers for numpy backend")
    ap.add_argument("--require-pos-lam1", action="store_true", default=True,
                    help="Filter out trajectories with lambda_1 < 0 (default on)")
    args = ap.parse_args()

    # z=+1 and z=-1 only
    zs_override = np.array([[1, 1], [-1, 1]], dtype=np.int64)

    nsh = cg.nshift_for(args.dim)
    dirs, _ = build_pools(args.dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs_override)

    print(f"=== z=±1 SWEEP dim={args.dim} N={args.nrank} box=+-{args.box} ===")
    print(f"  backend={args.backend}  dtype={args.dtype}")
    print(f"  total={args.total:,}  batch={args.batch:,}")
    print(f"  thresh={args.thresh}  seed={args.seed}")
    print(f"  z pool: +1, -1  (nz={nz})")
    print(f"  ndir={ndir}  nsh={nsh}")
    print(f"  total space: {ndir * nz * (2*args.box+1)**nsh:.2e}")
    print()

    # Import backend
    if args.backend == "cupy":
        from proxy_kernel import ProxyKernel
        kernel = ProxyKernel(args.dim, nsteps=args.nrank, box=args.box,
                             dtype=args.dtype, zs_override=zs_override)
        print(f"  cupy kernel compiled (nz={kernel.nz})")
    elif args.backend == "torch":
        from proxy_kernel import run_torch
        print(f"  torch backend ready")
    else:
        from proxy_kernel import run_numpy
        print(f"  numpy backend (workers={args.workers})")

    outf = open(args.out, "w")
    t0 = time.time()
    done = 0
    survivors = 0
    max_r2 = float("-inf")
    max_gid = -1
    r2_values = []

    while done < args.total:
        n = min(args.batch, args.total - done)
        if args.backend == "cupy":
            r2_d, l1_d = kernel.run(args.seed, done, n)
            r2 = r2_d.get(); l1 = l1_d.get()
        elif args.backend == "torch":
            r2, l1 = run_torch(args.seed, done, n, args.dim,
                               nsteps=args.nrank, box=args.box,
                               device=args.device, dtype=args.dtype,
                               zs_override=zs_override)
        else:
            r2, l1 = run_numpy(args.seed, done, n, args.dim,
                               nsteps=args.nrank, box=args.box,
                               zs_override=zs_override)

        finite = np.isfinite(r2)
        if finite.any():
            bi = int(np.argmax(np.where(finite, r2, -np.inf)))
            if r2[bi] > max_r2:
                max_r2 = float(r2[bi]); max_gid = int(done + bi)
            r2_values.extend(r2[finite].tolist())

        # Filter: require lambda_1 > 0 (expanding denominator) for valid proxy
        valid = finite & (l1 > 0)
        hit = valid & (r2 > args.thresh)
        idx = np.nonzero(hit)[0]
        if idx.size:
            gids = (done + idx).astype(np.uint64)
            shift, di, zi = gen_params(args.seed, gids, nsh, args.box, ndir, nz)
            # Non-degenerate filter: reject shift[i] == -1 for i in range(dim)
            filtered_idx = []
            for k, gi in enumerate(idx):
                is_degen = any(shift[k, i] == -1 for i in range(args.dim))
                if is_degen:
                    continue
                filtered_idx.append((k, gi))
            for k, gi in filtered_idx:
                rec = {
                    "gid": int(done + gi),
                    "r2": float(r2[gi]),
                    "lam1": float(l1[gi]),
                    "shift": shift[k].tolist(),
                    "dir": dirs[di[k]].tolist(),
                    "z_num": int(zs_override[zi[k], 0]),
                    "z_den": int(zs_override[zi[k], 1]),
                }
                outf.write(json.dumps(rec) + "\n")
            survivors += len(filtered_idx)
            outf.flush()

        done += n
        el = time.time() - t0
        thr = done / el if el else 0
        print(f"  screened {done:,}/{args.total:,}  survivors={survivors:,}  "
              f"max_r2={max_r2:+.6f}  {thr:,.0f} traj/s  "
              f"({thr*3600:,.0f}/h)", flush=True)

    outf.close()
    el = time.time() - t0

    # Statistics
    r2_arr = np.array(r2_values)
    print(f"\n=== DONE ===")
    print(f"  screened {done:,} in {el:.1f}s ({done/el:,.0f}/s)")
    print(f"  survivors={survivors:,} (thresh={args.thresh}) -> {args.out}")
    print(f"  funnel ratio: {survivors}/{done} = {survivors/max(done,1):.2e}")
    print(f"  max r2 = {max_r2:+.6f} at gid={max_gid}")
    if r2_arr.size:
        print(f"  r2 stats: mean={r2_arr.mean():.4f}  std={r2_arr.std():.4f}  "
              f"median={np.median(r2_arr):.4f}")
        print(f"  r2 > 0:   {(r2_arr > 0).sum():,} ({(r2_arr > 0).mean()*100:.1f}%)")
        print(f"  r2 > 0.02: {(r2_arr > 0.02).sum():,} ({(r2_arr > 0.02).mean()*100:.1f}%)")
        print(f"  r2 > 0.05: {(r2_arr > 0.05).sum():,} ({(r2_arr > 0.05).mean()*100:.1f}%)\n"
        f"  lambda_1 > 0: {(np.array(r2_values) > -999).sum():,}  "
        f"(filter removes contracting-denominator false positives)")
    print(f"\n  Stage 2: python3 pslq_companion.py --in {args.out} --dim {args.dim}")


if __name__ == "__main__":
    main()
