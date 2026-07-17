"""Stage-1 GPU screening driver for companion CMFs.

Historic production sweeps used r_2 only. The revised driver can additionally
emit and screen the full spectral ladder with ``--full-ladder``. A positive
ratio at k>=3 is only a spectral candidate; it is never labelled J* without
visibility analysis and exact verification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg
from params import build_pools, gen_params


def _candidate_from_ladder(ratios, threshold, upper=1.05):
    stable = np.isfinite(ratios) & (np.abs(ratios) < upper) & (ratios > threshold)
    values = np.where(stable, ratios, -np.inf)
    best_index = np.argmax(values, axis=1)
    best_value = values[np.arange(len(values)), best_index]
    keep = np.isfinite(best_value) & (best_value > -np.inf)
    return keep, best_value, best_index + 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--total", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--z-max", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--gid0", type=int, default=0, help="absolute first gid for this task")
    ap.add_argument("--backend", choices=["cupy", "torch", "numpy"], default="numpy")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--z-fixed", default=None)
    ap.add_argument("--out", default="survivors.jsonl")
    ap.add_argument("--require-pos-lam1", action="store_true")
    ap.add_argument("--no-degenerate", action="store_true")
    ap.add_argument("--full-ladder", action="store_true",
                    help="screen r_2,...,r_d; outputs k_candidate, not J*")
    args = ap.parse_args()

    zs_override = None
    if args.z_fixed:
        zs_override = np.asarray(
            [tuple(map(int, item.split(","))) for item in args.z_fixed.split(";")],
            dtype=np.int64)
    dirs, zs = build_pools(args.dim, z_max=args.z_max)
    if zs_override is not None:
        zs = zs_override
    nsh = cg.nshift_for(args.dim)

    if args.backend == "cupy":
        from proxy_kernel import ProxyKernel
        kernel = ProxyKernel(args.dim, nsteps=args.nrank, box=args.box,
                             z_max=args.z_max, dtype=args.dtype,
                             zs_override=zs_override)
        run = lambda gid0,n: kernel.run(args.seed,args.gid0+gid0,n,full_ladder=args.full_ladder)
    elif args.backend == "torch":
        from proxy_kernel import run_torch
        run = lambda gid0,n: run_torch(
            args.seed,args.gid0+gid0,n,args.dim,nsteps=args.nrank,box=args.box,z_max=args.z_max,
            device=args.device,dtype=args.dtype,zs_override=zs_override,
            full_ladder=args.full_ladder)
    else:
        from proxy_kernel import run_numpy
        run = lambda gid0,n: run_numpy(
            args.seed,args.gid0+gid0,n,args.dim,nsteps=args.nrank,box=args.box,z_max=args.z_max,
            dtype=args.dtype,zs_override=zs_override,full_ladder=args.full_ladder)

    t0 = time.time(); done = survivors = 0
    with open(args.out,"w") as out:
        while done < args.total:
            n = min(args.batch,args.total-done)
            spectral, lam1 = run(done,n)
            if hasattr(spectral,"get"):
                spectral = spectral.get(); lam1 = lam1.get()
            spectral = np.asarray(spectral); lam1 = np.asarray(lam1)

            if args.full_ladder:
                hit, score, k_candidate = _candidate_from_ladder(spectral,args.thresh)
            else:
                score = spectral
                hit = np.isfinite(score) & (score > args.thresh)
                k_candidate = np.full(n,2,dtype=int)
            if args.require_pos_lam1:
                hit &= lam1 > 0

            idx = np.nonzero(hit)[0]
            if idx.size:
                gids = (args.gid0+done+idx).astype(np.uint64)
                shift,di,zi = gen_params(args.seed,gids,nsh,args.box,len(dirs),len(zs))
                for local,pos in enumerate(idx):
                    if args.no_degenerate and cg.is_degenerate(
                        shift[local],dirs[di[local]],args.dim,1,args.nrank):
                        continue
                    rec = {"gid": int(args.gid0+done+pos),
                           "score": float(score[pos]),
                           "k_candidate": int(k_candidate[pos]),
                           "lam1": float(lam1[pos]),
                           "shift": shift[local].tolist(),
                           "dir": dirs[di[local]].tolist(),
                           "z_num": int(zs[zi[local],0]),
                           "z_den": int(zs[zi[local],1]),
                           "product_order": "right"}
                    if args.full_ladder: rec["ratios"] = spectral[pos].tolist()
                    else: rec["r2"] = float(score[pos])
                    out.write(json.dumps(rec)+"\n"); survivors += 1
                out.flush()
            done += n
            elapsed = time.time()-t0
            print(f"screened {done:,}/{args.total:,}; survivors={survivors:,}; "
                  f"{done/max(elapsed,1e-12):,.0f} traj/s",flush=True)

    elapsed = time.time()-t0
    print(f"DONE {done:,} in {elapsed:.2f}s; {done/elapsed:,.0f} traj/s; "
          f"survivors={survivors:,} -> {args.out}")


if __name__ == "__main__":
    main()
