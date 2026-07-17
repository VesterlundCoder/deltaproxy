"""
spectrum_diag.py
================
Multi-ratio Lyapunov-spectrum diagnostic for the pFq families (4F3, 5F4, 6F5,
8F7). The production proxy uses ONLY r_2 = -lambda_2/lambda_1. This tool records
the WHOLE subdominant ladder

    r_k = -lambda_k / lambda_1 ,   k = 2, 3, ..., N

over trajectories drawn from the EXACT SAME parameter distribution as the LUMI
GPU sweep (same splitmix64 gen_params, same pfq_dir_pool, same z-pool, same
shift->x/y mapping). Purpose: understand the J* Lyapunov geometry per dimension
-- specifically whether the barren r_2 in 5F4/8F7 hides a positive deeper mode
(r_3, r_4, ...) and what drives the 4F3 degenerate flood.

It does NOT stitch a theory; it reports the raw distribution of each ratio so the
dynamics can be inspected directly.

Usage:
  python3 spectrum_diag.py --family 4f3 --nsamp 40000 --depth 200
  python3 spectrum_diag.py --family all --nsamp 40000 --depth 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_proxy"))

import cmf_generic as cg                                    # noqa: E402
from pfq_gpu import PFqBuilder, pfq_dir_pool, map_shift_to_xy  # noqa: E402
from params import gen_params, z_pool_arr                   # noqa: E402

FAMILIES = {"4f3": 4, "5f4": 5, "6f5": 6, "8f7": 8}
DEGEN_HI = 1.05          # |r_k| >= this  ==> degenerate (pole/collapse) artifact
POS_THRESH = 0.02        # the production survivor threshold on r_2


def sample_ladder_pfq(dim, nsamp, depth, box, z_max, seed):
    """Draw `nsamp` trajectories exactly as the pFq sweep does (map_shift_to_xy,
    pfq_dir_pool) and return the full ratio ladder for each.
    Returns (ratios (n,N-1), lam (n,N), bad (n,))."""
    nsh = cg.nshift_for(dim)                # = 2*dim-1
    dirs = pfq_dir_pool(dim)
    zs = z_pool_arr(z_max)
    ndir, nz = len(dirs), len(zs)
    gids = np.arange(0, nsamp, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, ndir, nz)
    xy = [map_shift_to_xy(shift[t], dim, box) for t in range(nsamp)]
    xstart = np.array([r[0] for r in xy], dtype=np.float64)
    ystart = np.array([r[1] for r in xy], dtype=np.float64)
    startmat = np.concatenate([xstart, ystart], axis=1)     # (n, nsh)
    zval = (zs[zi, 0] / zs[zi, 1]).astype(np.float64)

    builder = PFqBuilder(dim, dim - 1, np)
    N = builder.N
    ratios = np.full((nsamp, N - 1), np.nan)
    lam = np.full((nsamp, N), np.nan)
    bad = np.ones(nsamp, dtype=bool)
    for d in range(ndir):
        grp = np.nonzero(di == d)[0]
        if grp.size == 0:
            continue
        st_np = startmat[grp]
        base0 = np.zeros(grp.size)
        startv = [st_np[:, a].copy() for a in range(nsh)]
        zv = zval[grp].copy()
        gr, gl, gb = builder.lyapunov_ladder_pfq(startv, dirs[d], zv, depth, base0)
        ratios[grp] = gr
        lam[grp] = gl
        bad[grp] = gb
    return ratios, lam, bad


def sample_ladder_companion(dim, nsamp, depth, box, z_max, seed):
    """Draw `nsamp` trajectories exactly as the COMPANION (esym) sweep does
    (cg.dir_pool, shift used directly as recurrence coefficients) and return the
    full ratio ladder. This is the construction used by the production 6F5/8F7
    GPU sweeps. Returns (ratios (n,dim-1), lam (n,dim), bad (n,))."""
    from proxy_batch import lyapunov_batch, ratios_batch
    from params import build_pools
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=z_max)
    ndir, nz = len(dirs), len(zs)
    gids = np.arange(0, nsamp, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, ndir, nz)
    dirv = dirs[di].astype(float)
    zn = zs[zi, 0].astype(float)
    zd = zs[zi, 1].astype(float)
    lam = lyapunov_batch(shift.astype(float), dirv, zn, zd, dim, depth, np)
    ratios = ratios_batch(lam, np)                          # (n, dim-1)
    l1 = lam[:, 0]
    bad = ~np.isfinite(l1) | (l1 == 0)
    return ratios, lam, bad


def sample_ladder(dim, nsamp, depth, box, z_max, seed, matrix="pfq"):
    if matrix == "pfq":
        return sample_ladder_pfq(dim, nsamp, depth, box, z_max, seed)
    return sample_ladder_companion(dim, nsamp, depth, box, z_max, seed)


def summarize(fam, dim, ratios, lam, bad, depth):
    N = dim
    n = ratios.shape[0]
    finite_row = np.isfinite(ratios).all(axis=1)
    n_pole = int(bad.sum())
    out = {"family": fam, "dim": dim, "N": N, "nsamp": n, "depth": depth,
           "n_pole_or_nonfinite": n_pole, "ratios": {}}

    # per-ratio distribution
    for k in range(N - 1):                                  # ratios[:,k] == r_{k+2}
        rk = ratios[:, k]
        f = rk[np.isfinite(rk)]
        nondegen = f[np.abs(f) < DEGEN_HI]
        rec = {
            "index_k": k + 2,
            "n_finite": int(f.size),
            "n_positive": int((f > 0).sum()),
            "n_gt_thresh": int((f > POS_THRESH).sum()),
            "n_degenerate_ge_1.05": int((np.abs(f) >= DEGEN_HI).sum()),
            "max": (float(np.max(f)) if f.size else None),
            "min": (float(np.min(f)) if f.size else None),
            "median": (float(np.median(f)) if f.size else None),
            "max_nondegenerate": (float(np.max(nondegen)) if nondegen.size else None),
            "p99_nondegenerate": (float(np.percentile(nondegen, 99)) if nondegen.size else None),
        }
        out["ratios"][f"r{k + 2}"] = rec

    # robust detector: first meaningful (finite, |.|<DEGEN_HI) ratio per traj,
    # scanning r_2, r_3, ...  (mirrors cmf_generic.spectral_dhat_robust)
    robust = np.full(n, np.nan)
    driver = np.zeros(n, dtype=int)
    for i in range(n):
        for k in range(N - 1):
            v = ratios[i, k]
            if np.isfinite(v) and abs(v) < DEGEN_HI:
                robust[i] = v
                driver[i] = k + 2
                break
    rf = robust[np.isfinite(robust)]
    out["robust_detector"] = {
        "n_meaningful": int(rf.size),
        "n_positive": int((rf > 0).sum()),
        "n_gt_thresh": int((rf > POS_THRESH).sum()),
        "max": (float(np.max(rf)) if rf.size else None),
        "median": (float(np.median(rf)) if rf.size else None),
        "driver_hist": {int(d): int((driver[np.isfinite(robust)] == d).sum())
                        for d in sorted(set(driver[np.isfinite(robust)].tolist()))},
    }
    # does ANY deeper ratio (k>=3) rescue a non-positive/degenerate r_2?
    r2 = ratios[:, 0]
    r2_dead = ~(np.isfinite(r2) & (r2 > POS_THRESH) & (np.abs(r2) < DEGEN_HI))
    deeper = ratios[:, 1:]
    deeper_pos = np.zeros(n, dtype=bool)
    if deeper.shape[1]:
        deeper_pos = np.any(np.isfinite(deeper) & (deeper > POS_THRESH)
                            & (np.abs(deeper) < DEGEN_HI), axis=1)
    out["deeper_rescue"] = {
        "n_r2_dead": int(r2_dead.sum()),
        "n_rescued_by_deeper": int((r2_dead & deeper_pos).sum()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all",
                    help="4f3 | 5f4 | 6f5 | 8f7 | all")
    ap.add_argument("--nsamp", type=int, default=40000)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--z-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--matrix", default="pfq", choices=["pfq", "companion"],
                    help="pFq exact hypergeometric CMF, or the generic esym "
                         "companion used by the production 6F5/8F7 sweeps")
    ap.add_argument("--outdir", default="spectrum_diag_out")
    args = ap.parse_args()

    fams = list(FAMILIES) if args.family == "all" else [args.family]
    os.makedirs(args.outdir, exist_ok=True)
    mtag = args.matrix
    summaries = {}
    for fam in fams:
        dim = FAMILIES[fam]
        t0 = time.time()
        label = f"pFq({dim},{dim - 1})" if mtag == "pfq" else f"companion(dim={dim})"
        print(f"[diag] {fam} [{mtag}] {label} N={dim}  nsamp={args.nsamp} "
              f"depth={args.depth} box={args.box} ...", flush=True)
        ratios, lam, bad = sample_ladder(dim, args.nsamp, args.depth,
                                         args.box, args.z_max, args.seed, mtag)
        np.savez_compressed(os.path.join(args.outdir, f"{fam}_{mtag}_ladder.npz"),
                            ratios=ratios, lam=lam, bad=bad)
        s = summarize(fam, dim, ratios, lam, bad, args.depth)
        s["matrix"] = mtag
        s["wall_s"] = round(time.time() - t0, 1)
        summaries[fam] = s
        print(f"  done {s['wall_s']}s  poles={s['n_pole_or_nonfinite']}  "
              f"robust_pos={s['robust_detector']['n_positive']}  "
              f"deeper_rescue={s['deeper_rescue']['n_rescued_by_deeper']}", flush=True)

    outname = f"spectrum_diag_summary_{mtag}.json"
    with open(os.path.join(args.outdir, outname), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[diag] summary -> {args.outdir}/{outname}")


if __name__ == "__main__":
    main()
