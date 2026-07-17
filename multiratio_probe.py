"""
multiratio_probe.py
===================
Validate the multi-ratio "hidden driver" idea against EXACT arithmetic.

Standard detector: r_2 = -lambda_2/lambda_1. Hypothesis: when r_2 is
non-meaningful (degenerate, or fails to match the observable's true delta), a
deeper ratio r_k = -lambda_k/lambda_1 (k>2) may be the real driver because the
last-column observable lives in a subspace that does not couple to mode 2
(typical for degenerate / block-structured companion cocycles).

For each sampled trajectory we compute:
  * the full ratio ladder r_2..r_dim (float Lyapunov),
  * the TRUE delta via exact pure-integer double-depth (independent_delta),
and report which ratio index best matches truth, plus 'rescue' cases where
|r_2 - truth| is large but some |r_k - truth| (k>2) is small.

Usage:
  python3 multiratio_probe.py --dim 6 --n 800 --nrank 160 --nverify 100 --workers 6
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import Pool

import numpy as np

import cmf_generic as cg


# A few hand-picked DEGENERATE 6F5 seeds (rows collapse -> effective lower dim),
# where the second Lyapunov mode is expected NOT to drive the observable.
DEGEN_6F5 = [
    # known degenerate 3F2-in-6F5 hit (delta -> +0.5): f=[0,0,0,1,1,1]
    dict(shift=[-1, -1, -1, 0, 0, 0, -2, -2, -2, -2, 3], dir=[0]*10 + [1],
         z_num=1, z_den=3, tag="known_degen_3F2"),
]


def _task(args):
    shift, dirv, zn, zd, nrank, nverify, dim = args
    try:
        ratios, lam = cg.spectral_ratios(shift, dirv, zn, zd, nrank, dim)
    except Exception:
        return None
    if not np.all(np.isfinite(lam)):
        return None
    try:
        truth = cg.independent_delta(shift, dirv, zn, zd, nverify, dim)
    except Exception:
        truth = None
    if truth is None:
        return None
    return {
        "ratios": ratios.tolist(),
        "lam": lam.tolist(),
        "truth": float(truth),
    }


def sample_tasks(dim, n, rng, nrank, nverify, include_degen):
    nsh = cg.nshift_for(dim)
    SEED = cg.default_seed(dim)
    DIRS = cg.dir_pool(dim)
    ZS = cg.z_pool(z_max=1.0)
    tasks = []
    if include_degen and dim == 6:
        for d in DEGEN_6F5:
            tasks.append((d["shift"], d["dir"], d["z_num"], d["z_den"],
                          nrank, nverify, dim))
    modes = ["local", "broad", "uniform", "degen_like"]
    for _ in range(n):
        m = rng.choice(modes)
        if m == "local":
            shift = [SEED[i] + rng.randint(-2, 2) for i in range(nsh)]
        elif m == "broad":
            shift = [SEED[i] + rng.randint(-5, 5) for i in range(nsh)]
        elif m == "degen_like":
            # force a collapse: set some numerator roots f_i = 0 (shift_i = -1)
            shift = [rng.randint(-4, 4) for _ in range(nsh)]
            for i in rng.sample(range(dim), rng.randint(1, dim // 2)):
                shift[i] = -1            # f_i = shift_i + 1 = 0 -> row collapses
        else:
            shift = [rng.randint(-6, 6) for _ in range(nsh)]
        dirv = list(rng.choice(DIRS))
        zn, zd = rng.choice(ZS)
        tasks.append((shift, dirv, zn, zd, nrank, nverify, dim))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--nrank", type=int, default=160)
    ap.add_argument("--nverify", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--match-tol", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = sample_tasks(args.dim, args.n, rng, args.nrank, args.nverify, True)
    print(f"[multiratio] dim={args.dim}  {len(tasks)} trajectories  "
          f"N={args.nrank} exact2N={2*args.nverify}  tol={args.match_tol}")

    with Pool(args.workers) as pool:
        res = [r for r in pool.map(_task, tasks) if r is not None]
    print(f"  computable: {len(res)}/{len(tasks)}")

    # for each trajectory, which ratio index best matches truth?
    best_k = []          # 2-based driver index that best matches truth
    err_r2 = []          # |r_2 - truth|
    err_best = []        # min_k |r_k - truth|
    rescue = []          # r_2 fails (>tol or degenerate) but some k>2 matches
    r2_works = 0
    for r in res:
        ratios = np.array(r["ratios"])
        truth = r["truth"]
        diffs = np.abs(ratios - truth)
        kbest = int(np.nanargmin(diffs))           # 0-based -> r_{kbest+2}
        best_k.append(kbest + 2)
        e2 = diffs[0]
        eb = diffs[kbest]
        err_r2.append(e2)
        err_best.append(eb)
        r2_ok = (e2 <= args.match_tol) and (abs(ratios[0]) < 1.05)
        r2_works += int(r2_ok)
        if (not r2_ok) and eb <= args.match_tol and (kbest + 2) > 2:
            rescue.append((r, kbest + 2, e2, eb))

    n = len(res)
    err_r2 = np.array(err_r2); err_best = np.array(err_best)
    from collections import Counter
    dist = Counter(best_k)
    print("\n  best-matching driver index (2 = standard detector):")
    for k in sorted(dist):
        print(f"    r_{k}: {dist[k]:>4}  ({100*dist[k]/n:5.1f}%)")
    print(f"\n  r_2 matches truth (tol, non-degenerate): {r2_works}/{n} "
          f"({100*r2_works/n:.1f}%)")
    print(f"  MAE  r_2 vs truth : {np.nanmean(err_r2):.4f}")
    print(f"  MAE  best-k vs truth: {np.nanmean(err_best):.4f}")
    print(f"  RESCUES (r_2 fails, deeper r_k matches): {len(rescue)}")
    for r, k, e2, eb in rescue[:12]:
        print(f"    driver r_{k}: truth={r['truth']:+.4f}  "
              f"r_2={r['ratios'][0]:+.4f}(err {e2:.3f})  "
              f"r_{k}={r['ratios'][k-2]:+.4f}(err {eb:.3f})")
    if rescue:
        frac = len(rescue) / n
        print(f"\n  => deeper-driver rescue rate: {100*frac:.2f}% of computable "
              f"trajectories had a hidden driver the standard r_2 missed.")
    else:
        print("\n  => no rescues: r_2 (or its degeneracy flag) already captures "
              "the true driver in this sample.")


if __name__ == "__main__":
    main()
