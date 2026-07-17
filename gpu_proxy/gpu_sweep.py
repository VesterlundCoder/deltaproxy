"""
gpu_sweep.py
============
Stage-1 driver: screen up to billions of trajectories with the GPU proxy and
funnel the SURVIVORS (r_2 above threshold) to disk for exact verification.

Only survivors are materialized: their parameters are re-derived from (seed,
gid) on the host, so a multi-billion sweep writes only a few thousand rows.

Backends:
  --backend cupy   real GPU (LUMI MI250X via cupy-rocm, or NVIDIA)
  --backend numpy  CPU fallback (validation / laptops; same r_2)

Examples:
  # local correctness + throughput on CPU
  python3 gpu_sweep.py --dim 6 --total 200000 --batch 50000 --backend numpy
  # LUMI: 2 billion 6F5 trajectories, fp32 screen
  python3 gpu_sweep.py --dim 6 --total 2000000000 --batch 4000000 \
      --backend cupy --dtype float32 --thresh 0.02 --out survivors_6f5.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg                       # noqa: E402
from params import gen_params, build_pools     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--total", type=int, default=200000, help="trajectories to screen")
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--z-max", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=0.02,
                    help="keep trajectories with r_2 > thresh (Stage-1 funnel)")
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--backend", choices=["cupy", "torch", "numpy"], default="numpy")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", default="cuda", help="torch device (cuda=ROCm too)")
    ap.add_argument("--matrix", choices=["companion", "pfq"], default="companion",
                    help="companion=esym cocycle; pfq=EXACT RamanujanTools pFq CMF")
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="instead of r2>thresh, keep LO < r2 < HI (e.g. -0.5 0)")
    ap.add_argument("--z-fixed", default=None,
                    help="override z pool with explicit pairs, e.g. '1,1;-1,1'")
    ap.add_argument("--out", default="survivors.jsonl")
    ap.add_argument("--require-pos-lam1", action="store_true", default=False,
                    help="Filter out trajectories with lambda_1 <= 0 (needed for z=±1)")
    ap.add_argument("--no-degenerate", action="store_true", default=False,
                    help="Reject degenerate trajectories (shift[i]==-1 for i in range(dim))")
    args = ap.parse_args()

    zs_override = None
    if args.z_fixed:
        pairs = [p.strip().split(",") for p in args.z_fixed.split(";")]
        zs_override = np.array([(int(a), int(b)) for a, b in pairs], dtype=np.int64)

    nsh = cg.nshift_for(args.dim)
    if args.matrix == "pfq":
        from params import z_pool_arr
        from pfq_gpu import pfq_dir_pool
        dirs = np.array(pfq_dir_pool(args.dim), dtype=np.int64)
        zs = zs_override if zs_override is not None else z_pool_arr(z_max=args.z_max)
    else:
        dirs, zs = build_pools(args.dim, z_max=args.z_max)
        if zs_override is not None:
            zs = zs_override
    ndir, nz = len(dirs), len(zs)

    pfq = args.matrix == "pfq"
    if pfq and args.backend == "cupy":
        print("[gpu_sweep] --matrix pfq has no cupy raw-kernel; use --backend torch")
        sys.exit(2)

    kernel = None
    if args.backend == "cupy":
        from proxy_kernel import ProxyKernel
        kernel = ProxyKernel(args.dim, nsteps=args.nrank, box=args.box,
                             z_max=args.z_max, dtype=args.dtype,
                             zs_override=zs_override)
        print(f"[gpu_sweep] cupy kernel dim={args.dim} N={args.nrank} "
              f"dtype={args.dtype} box=+-{args.box} ndir={ndir} nz={nz}")
    elif args.backend == "torch":
        from proxy_kernel import run_torch, run_torch_pfq
        print(f"[gpu_sweep] torch GPU dim={args.dim} N={args.nrank} matrix={args.matrix} "
              f"dtype={args.dtype} device={args.device} box=+-{args.box} "
              f"ndir={ndir} nz={nz}")
    else:
        from proxy_kernel import run_numpy, run_numpy_pfq
        print(f"[gpu_sweep] numpy fallback dim={args.dim} N={args.nrank} matrix={args.matrix}")

    outf = open(args.out, "w")
    t0 = time.time()
    done = 0
    survivors = 0
    max_r2 = float("-inf")
    max_gid = -1
    while done < args.total:
        n = min(args.batch, args.total - done)
        if args.backend == "cupy":
            r2_d, l1_d = kernel.run(args.seed, done, n)
            r2 = r2_d.get(); l1 = l1_d.get()
        elif args.backend == "torch":
            fn = run_torch_pfq if pfq else run_torch
            r2, l1 = fn(args.seed, done, n, args.dim, nsteps=args.nrank,
                        box=args.box, z_max=args.z_max,
                        device=args.device, dtype=args.dtype,
                        zs_override=zs_override)
        else:
            fn = run_numpy_pfq if pfq else run_numpy
            r2, l1 = fn(args.seed, done, n, args.dim,
                        nsteps=args.nrank, box=args.box, z_max=args.z_max,
                        zs_override=zs_override)
        finite = np.isfinite(r2)
        if finite.any():
            bi = int(np.argmax(np.where(finite, r2, -np.inf)))
            if r2[bi] > max_r2:
                max_r2 = float(r2[bi]); max_gid = int(done + bi)
        if args.band is not None:
            lo, hi = args.band
            hit = finite & (r2 > lo) & (r2 < hi)
        else:
            hit = finite & (r2 > args.thresh)
        if args.require_pos_lam1:
            hit = hit & (l1 > 0)
        idx = np.nonzero(hit)[0]
        if idx.size:
            gids = (done + idx).astype(np.uint64)
            shift, di, zi = gen_params(args.seed, gids, nsh, args.box, ndir, nz)
            n_written = 0
            for k, gi in enumerate(idx):
                if args.no_degenerate:
                    if any(shift[k, i] == -1 for i in range(args.dim)):
                        continue
                rec = {
                    "gid": int(done + gi),
                    "r2": float(r2[gi]),
                    "lam1": float(l1[gi]),
                    "matrix": args.matrix,
                    "shift": shift[k].tolist(),
                    "dir": dirs[di[k]].tolist(),
                    "z_num": int(zs[zi[k], 0]),
                    "z_den": int(zs[zi[k], 1]),
                }
                if pfq:
                    from pfq_gpu import map_shift_to_xy
                    xst, yst = map_shift_to_xy(shift[k], args.dim, args.box)
                    rec["x_start"] = xst
                    rec["y_start"] = yst
                outf.write(json.dumps(rec) + "\n")
                n_written += 1
            survivors += n_written
            outf.flush()
        done += n
        el = time.time() - t0
        thr = done / el if el else 0
        print(f"  screened {done:,}/{args.total:,}  survivors={survivors:,}  "
              f"max_r2={max_r2:+.4f}  {thr:,.0f} traj/s  ({thr*3600:,.0f}/h)",
              flush=True)

    outf.close()
    el = time.time() - t0
    print(f"\n[gpu_sweep] DONE {done:,} screened in {el:.1f}s "
          f"({done/el:,.0f}/s, {done/el*3600:,.0f}/h)  "
          f"survivors={survivors:,} -> {args.out}")
    print(f"  max_r2 seen = {max_r2:+.6f} at gid={max_gid}")
    print(f"  funnel ratio: {survivors}/{done} = {survivors/max(done,1):.2e}")
    print(f"  Stage 2: python3 verify_survivors.py --in {args.out} --dim {args.dim}")


if __name__ == "__main__":
    main()
