#!/usr/bin/env python3
"""
benchmark_proxy_vs_exact.py
============================
Reproducible benchmark: Lyapunov spectral proxy vs exact integer arithmetic.

Four metrics:
  1. Raw throughput (traj/s)
  2. Statistical enrichment (precision/recall/sign-agreement)
  3. End-to-end discovery speed-up
  4. Verification cost on the selected tail

Usage:
  python3 benchmark_proxy_vs_exact.py --dim 6 --n-candidates 5000 \
      --depth 120 --box 6 --workers 8 --out benchmark_results.json
"""
from __future__ import annotations
import argparse, json, os, platform, socket, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_proxy"))
import cmf_generic as cg
from spectral_delta import lyapunov_spectrum
from params import gen_params, build_pools


def env_info():
    info = {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__}
    try:
        import mpmath; info["mpmath"] = mpmath.__version__
    except: pass
    try:
        import torch; info["torch"] = torch.__version__
        if torch.cuda.is_available(): info["gpu"] = torch.cuda.get_device_name(0)
    except: pass
    return info


def gen_candidates(seed, n, dim, box):
    nsh = cg.nshift_for(dim)
    dirs, zs = build_pools(dim, z_max=1.0)
    ndir, nz = len(dirs), len(zs)
    gids = np.arange(0, n, dtype=np.uint64)
    shift, di, zi = gen_params(seed, gids, nsh, box, ndir, nz)
    return [{"gid": int(gids[k]), "shift": shift[k].tolist(),
             "dir": dirs[di[k]].tolist(),
             "z_num": int(zs[zi[k], 0]), "z_den": int(zs[zi[k], 1])}
            for k in range(n)]


def proxy_one(args):
    rec, dim, N = args
    lam = lyapunov_spectrum(
        lambda n: cg.build_M_float(n+1, rec["shift"], rec["dir"],
                                   rec["z_num"], rec["z_den"], dim), dim, N)
    if lam[0] == 0 or not np.isfinite(lam[0]): return float("nan")
    return float(-lam[1] / lam[0])


def exact_one(args):
    rec, dim, N = args
    return cg.independent_delta(rec["shift"], rec["dir"],
                                rec["z_num"], rec["z_den"], N, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--n-candidates", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=120)
    ap.add_argument("--box", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--n-exact", type=int, default=2000)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    env = env_info()
    print(f"=== BENCHMARK dim={args.dim} N={args.depth} box=+-{args.box} ===")
    print(f"  host={env['hostname']}  candidates={args.n_candidates:,}")

    cands = gen_candidates(args.seed, args.n_candidates, args.dim, args.box)

    # Phase 1: proxy throughput
    print("\n--- Proxy throughput ---")
    pargs = [(c, args.dim, args.depth) for c in cands]
    with Pool(args.workers) as p:
        p.map(proxy_one, pargs[:32])  # warm-up
    t0 = time.time()
    with Pool(args.workers) as p:
        proxy_d = p.map(proxy_one, pargs)
    pt = time.time() - t0
    proxy_d = np.array(proxy_d)
    pthr = args.n_candidates / pt
    print(f"  {pt:.3f}s -> {pthr:,.0f} traj/s ({pthr*3600:,.0f}/h)")

    # Phase 2: exact throughput
    print("\n--- Exact integer delta throughput ---")
    n_ex = min(args.n_exact, args.n_candidates)
    eargs = [(c, args.dim, args.depth) for c in cands[:n_ex]]
    with Pool(args.workers) as p:
        p.map(exact_one, eargs[:8])  # warm-up
    t0 = time.time()
    with Pool(args.workers) as p:
        exact_d = p.map(exact_one, eargs)
    et = time.time() - t0
    exact_d = np.array([d if d is not None else float("nan") for d in exact_d])
    ethr = n_ex / et
    print(f"  {et:.3f}s for {n_ex:,} -> {ethr:,.1f} traj/s ({ethr*3600:,.0f}/h)")

    # Phase 3: metrics
    pd = proxy_d[:n_ex]
    fin = np.isfinite(pd) & np.isfinite(exact_d)
    pp = pd[fin] > args.thresh
    ep = exact_d[fin] > 0.0
    tp, fp = int((pp & ep).sum()), int((pp & ~ep).sum())
    fn, tn = int((~pp & ep).sum()), int((~pp & ~ep).sum())
    sign_agree = float(np.mean((pd[fin] > 0) == (exact_d[fin] > 0)))
    mae = float(np.mean(np.abs(pd[fin] - exact_d[fin])))

    # End-to-end
    survivors = int((proxy_d > args.thresh).sum())
    exact_verify_time = (survivors / n_ex) * et if survivors > 0 else 0
    total_proxy = pt + exact_verify_time
    total_exact = (args.n_candidates / n_ex) * et
    e2e = total_exact / total_proxy if total_proxy > 0 else float("inf")

    raw_speedup = pthr / ethr

    R = {
        "environment": env,
        "parameters": vars(args),
        "timing": {"proxy_s": pt, "proxy_per_s": pthr,
                    "exact_s": et, "exact_n": n_ex, "exact_per_s": ethr},
        "metrics": {
            "raw_throughput_speedup": raw_speedup,
            "sign_agreement": sign_agree, "mae": mae,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1),
            "proxy_survivors": survivors,
            "e2e_speedup": e2e,
            "cost_per_hit_s": total_proxy / max(tp, 1),
        },
    }
    print(f"\n=== RESULTS ===")
    print(f"  Raw throughput speedup: {raw_speedup:,.0f}x")
    print(f"  Sign agreement: {sign_agree:.4f}  MAE: {mae:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  E2E speedup: {e2e:,.0f}x")
    print(f"  Cost per confirmed hit: {total_proxy/max(tp,1):.2f}s")

    with open(args.out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
