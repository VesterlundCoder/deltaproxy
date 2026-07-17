#!/usr/bin/env python3
"""
validation_holdout.py
======================
Independent validation of the Lyapunov proxy on holdout sets that are
structurally disjoint from the training/discovery data.

Five holdout types:
  1. Shard holdout: different RNG seed (disjoint gid ranges)
  2. Direction holdout: directions not used in the main sweep
  3. Z-holdout: z-values not used in the main sweep
  4. Full-dimensional-only: exclude degenerate (f_i=0) trajectories
  5. Adversarial degeneracy: near-boundary spectral gap

For each holdout, compute proxy vs exact on N trajectories and report
sign-agreement, MAE, precision, recall.

Usage:
  python3 validation_holdout.py --dim 6 --n 2000 --depth 120 --box 6 \
      --workers 8 --out validation_results.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_proxy"))
import cmf_generic as cg
from spectral_delta import lyapunov_spectrum
from params import gen_params, build_pools


def proxy_one(args):
    rec, dim, N = args
    lam = lyapunov_spectrum(
        lambda n: cg.build_M_float(n+1, rec["shift"], rec["dir"],
                                   rec["z_num"], rec["z_den"], dim), dim, N)
    if lam[0] == 0 or not np.isfinite(lam[0]): return float("nan"), lam
    return float(-lam[1] / lam[0]), lam


def exact_one(args):
    rec, dim, N = args
    return cg.independent_delta(rec["shift"], rec["dir"],
                                rec["z_num"], rec["z_den"], N, dim)


def evaluate(cands, dim, N, workers, label):
    """Run proxy + exact on candidates, return metrics dict."""
    pargs = [(c, dim, N) for c in cands]
    with Pool(workers) as p:
        proxy_res = p.map(proxy_one, pargs)
    proxy_d = np.array([r[0] for r in proxy_res])
    proxy_lam = [r[1] for r in proxy_res]

    eargs = [(c, dim, N) for c in cands]
    with Pool(workers) as p:
        exact_res = p.map(exact_one, eargs)
    exact_d = np.array([d if d is not None else float("nan") for d in exact_res])

    fin = np.isfinite(proxy_d) & np.isfinite(exact_d)
    pp = proxy_d[fin] > 0.0
    ep = exact_d[fin] > 0.0
    tp, fp = int((pp & ep).sum()), int((pp & ~ep).sum())
    fn, tn = int((~pp & ep).sum()), int((~pp & ~ep).sum())
    sign_agree = float(np.mean((proxy_d[fin] > 0) == (exact_d[fin] > 0))) if fin.sum() else 0.0
    mae = float(np.mean(np.abs(proxy_d[fin] - exact_d[fin]))) if fin.sum() else 0.0
    pearson = float(np.corrcoef(proxy_d[fin], exact_d[fin])[0,1]) if fin.sum() > 2 else 0.0

    # Spectral gap stats
    gaps = np.array([lam[0]-lam[1] for lam in proxy_lam if np.all(np.isfinite(lam))])

    m = {
        "label": label, "n": len(cands), "n_finite": int(fin.sum()),
        "sign_agreement": sign_agree, "mae": mae, "pearson_r": pearson,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1),
        "exact_positive_rate": float(ep.mean()) if ep.size else 0,
        "median_spectral_gap": float(np.median(gaps)) if gaps.size else 0,
        "min_spectral_gap": float(np.min(gaps)) if gaps.size else 0,
    }
    print(f"  [{label}] n={m['n_finite']}  sign={sign_agree:.4f}  "
          f"MAE={mae:.4f}  r={pearson:.4f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return m


def gen_with_seed(seed, n, dim, box):
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    gids = np.arange(0, n, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, ndir, nz)
    return [{"gid": int(gids[k]), "shift": shift[k].tolist(),
             "dir": dirs[di[k]].tolist(),
             "z_num": int(zs[zi[k], 0]), "z_den": int(zs[zi[k], 1])}
            for k in range(n)]


def gen_direction_holdout(seed, n, dim, box, exclude_indices):
    """Generate candidates using only directions NOT in exclude_indices."""
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    allowed = [i for i in range(ndir) if i not in exclude_indices]
    if not allowed:
        raise ValueError("All directions excluded")
    rng = np.random.RandomState(seed)
    cands = []
    for k in range(n):
        di = allowed[rng.randint(len(allowed))]
        zi = rng.randint(nz)
        sh = rng.randint(-box, box+1, size=nsh)
        cands.append({"gid": k, "shift": sh.tolist(),
                      "dir": dirs[di].tolist(),
                      "z_num": int(zs[zi, 0]), "z_den": int(zs[zi, 1])})
    return cands


def gen_z_holdout(seed, n, dim, box, exclude_z_pairs):
    """Generate candidates using z-values NOT in exclude_z_pairs."""
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    allowed = [i for i in range(nz)
               if (int(zs[i,0]), int(zs[i,1])) not in exclude_z_pairs]
    if not allowed:
        raise ValueError("All z-values excluded")
    rng = np.random.RandomState(seed)
    cands = []
    for k in range(n):
        di = rng.randint(ndir)
        zi = allowed[rng.randint(len(allowed))]
        sh = rng.randint(-box, box+1, size=nsh)
        cands.append({"gid": k, "shift": sh.tolist(),
                      "dir": dirs[di].tolist(),
                      "z_num": int(zs[zi, 0]), "z_den": int(zs[zi, 1])})
    return cands


def gen_full_dim_only(seed, n, dim, box):
    """Generate candidates excluding degenerate (f_i=0) trajectories.
    f_i = shift[i] + 1, so reject shift[i] == -1 for i in range(dim)."""
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    rng = np.random.RandomState(seed)
    cands = []
    while len(cands) < n:
        sh = rng.randint(-box, box+1, size=nsh)
        if any(sh[i] == -1 for i in range(dim)):
            continue
        di = rng.randint(ndir)
        zi = rng.randint(nz)
        cands.append({"gid": len(cands), "shift": sh.tolist(),
                      "dir": dirs[di].tolist(),
                      "z_num": int(zs[zi, 0]), "z_den": int(zs[zi, 1])})
    return cands


def gen_adversarial(seed, n, dim, box, gap_range=(0.25, 2.0)):
    """Generate candidates and keep only those with small spectral gap."""
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    rng = np.random.RandomState(seed)
    cands = []
    attempts = 0
    while len(cands) < n and attempts < n * 50:
        attempts += 1
        sh = rng.randint(-box, box+1, size=nsh)
        di = rng.randint(ndir)
        zi = rng.randint(nz)
        rec = {"gid": len(cands), "shift": sh.tolist(),
               "dir": dirs[di].tolist(),
               "z_num": int(zs[zi, 0]), "z_den": int(zs[zi, 1])}
        # Quick proxy to check spectral gap
        lam = lyapunov_spectrum(
            lambda nn: cg.build_M_float(nn+1, rec["shift"], rec["dir"],
                                        rec["z_num"], rec["z_den"], dim),
            dim, min(60, 120))
        if np.all(np.isfinite(lam)) and gap_range[0] < (lam[0]-lam[1]) < gap_range[1]:
            cands.append(rec)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--n", type=int, default=2000, help="per holdout set")
    ap.add_argument("--depth", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="validation_results.json")
    args = ap.parse_args()

    print(f"=== VALIDATION HOLDOUT STUDY dim={args.dim} N={args.depth} ===\n")

    all_results = {"parameters": vars(args), "holdouts": []}

    # 1. Shard holdout (different seed)
    print("1. Shard holdout (seed=99999, disjoint from training seed=20260626)")
    cands = gen_with_seed(99999, args.n, args.dim, args.box)
    m = evaluate(cands, args.dim, args.depth, args.workers, "shard_holdout")
    all_results["holdouts"].append(m)

    # 2. Direction holdout (exclude most common directions)
    print("\n2. Direction holdout (exclude dir indices 0-5)")
    cands = gen_direction_holdout(88888, args.n, args.dim, args.box,
                                  exclude_indices=set(range(6)))
    m = evaluate(cands, args.dim, args.depth, args.workers, "direction_holdout")
    all_results["holdouts"].append(m)

    # 3. Z-holdout (exclude z=1/2, -1/2, 1/3, -1/3)
    print("\n3. Z-holdout (exclude common z-values)")
    cands = gen_z_holdout(77777, args.n, args.dim, args.box,
                          exclude_z_pairs={(1,2),(-1,2),(1,3),(-1,3),(1,1),(-1,1)})
    m = evaluate(cands, args.dim, args.depth, args.workers, "z_holdout")
    all_results["holdouts"].append(m)

    # 4. Full-dimensional only (no degenerate)
    print("\n4. Full-dimensional only (reject shift[i]==-1)")
    cands = gen_full_dim_only(66666, args.n, args.dim, args.box)
    m = evaluate(cands, args.dim, args.depth, args.workers, "full_dim_only")
    all_results["holdouts"].append(m)

    # 5. Adversarial degeneracy (small spectral gap)
    print("\n5. Adversarial degeneracy (spectral gap in [0.25, 2.0])")
    cands = gen_adversarial(55555, min(args.n, 500), args.dim, args.box)
    m = evaluate(cands, args.dim, args.depth, args.workers, "adversarial")
    all_results["holdouts"].append(m)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'Holdout':<25} {'n':>6} {'sign':>8} {'MAE':>8} {'r':>8} {'prec':>6} {'rec':>6}")
    for h in all_results["holdouts"]:
        print(f"{h['label']:<25} {h['n_finite']:>6} {h['sign_agreement']:>8.4f} "
              f"{h['mae']:>8.4f} {h['pearson_r']:>8.4f} "
              f"{h['precision']:>6.3f} {h['recall']:>6.3f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
