"""
stress_test_proxy.py
=====================
Map the FAILURE MODES of the spectral detector dhat=-lam2/lam1 for the 6F5 cocycle:
where does the cheap float64 QR proxy *disagree* with exact arithmetic, and which
spectral diagnostics predict that failure?

We sample widely over the 11-D space (and force a few pathological families),
compute for each trajectory:
  - dhat (float64 QR, N)             : the proxy
  - lam1, lam2, lam3                 : leading Lyapunov exponents
  - d_exact (pure-int double-depth)  : ground truth at Nv
and bucket the agreement against the diagnostics
  - gap12 = lam1 - lam2   (spectral gap; identity needs gap12>0)
  - gap23 = lam2 - lam3   (subdominant cluster; row-pair alignment needs gap23>0)
  - |lam1|                (denominator growth; dhat=-lam2/lam1 unstable as lam1->0)
  - dhat magnitude        (the known degenerate tail dhat>~1.1)
A trajectory is a FAILURE if exact is uncomputable, signs disagree, or
|dhat - d_exact| > tol.

Also: depth-bias study — dhat at N in {60,120,240,480} for a subset, reporting drift.

Usage:
  python3 stress_test_proxy.py --n 6000 --nrank 120 --nverify 80 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

import cmf_generic as cg
from enrich import SEED_SHIFT, SEED_DIR

DIM = 6
NSH = 11


def _spectrum(shift, dirv, zn, zd, N):
    return cg.lyapunov_spectrum(
        lambda n: cg.build_M_float(n + 1, shift, dirv, zn, zd, DIM), DIM, N)


def _one(args):
    shift, dirv, zn, zd, nrank, nverify, family = args
    try:
        lam = _spectrum(shift, dirv, zn, zd, nrank)
        if lam[0] == 0 or not np.isfinite(lam[0]):
            dhat = float("nan")
        else:
            dhat = float(-lam[1] / lam[0])
        d_exact = cg.independent_delta(shift, dirv, zn, zd, nverify, DIM)
    except Exception as e:
        return {"family": family, "err": str(e)[:60]}
    return {"family": family, "dhat": dhat,
            "lam1": float(lam[0]), "lam2": float(lam[1]), "lam3": float(lam[2]),
            "d_exact": d_exact}


def _depth(args):
    shift, dirv, zn, zd, depths = args
    out = {}
    for N in depths:
        try:
            lam = _spectrum(shift, dirv, zn, zd, N)
            out[N] = float(-lam[1] / lam[0]) if lam[0] else float("nan")
        except Exception:
            out[N] = float("nan")
    return out


def sample_family(rng, family):
    """Return (shift, dir, zn, zd) drawn from a named regime."""
    DIRS = cg.dir_pool(DIM)
    ZS = cg.z_pool()
    if family == "local":
        shift = [SEED_SHIFT[i] + rng.randint(-2, 2) for i in range(NSH)]
        dirv = list(SEED_DIR) if rng.random() < .5 else list(rng.choice(DIRS))
        zn, zd = (7, 20) if rng.random() < .3 else rng.choice(ZS)
    elif family == "uniform":
        shift = [rng.randint(-6, 6) for _ in range(NSH)]
        dirv = list(rng.choice(DIRS)); zn, zd = rng.choice(ZS)
    elif family == "wide":                    # very large shifts (stress floats)
        shift = [rng.randint(-30, 30) for _ in range(NSH)]
        dirv = list(rng.choice(DIRS)); zn, zd = rng.choice(ZS)
    elif family == "extreme_z":               # |z| far from the trained band
        shift = [rng.randint(-6, 6) for _ in range(NSH)]
        dirv = list(rng.choice(DIRS))
        zn, zd = rng.choice([(rng.randint(2, 50), 1), (1, rng.randint(20, 200)),
                             (rng.randint(-50, -2), 1)])
    elif family == "multi_advance":           # many roots advancing -> fast collapse
        shift = [SEED_SHIFT[i] + rng.randint(-3, 3) for i in range(NSH)]
        dirv = [rng.choice([0, 0, 1, 2]) for _ in range(NSH)]
        zn, zd = rng.choice(ZS)
    else:  # near_seed_z_small : tiny |z| -> near-rank-deficient last column
        shift = [SEED_SHIFT[i] + rng.randint(-2, 2) for i in range(NSH)]
        dirv = list(rng.choice(DIRS)); zn, zd = (rng.choice([1, -1]), rng.randint(8, 12))
    return shift, dirv, zn, zd


def bucketize(rows, key, edges, tol):
    """Failure rate / sign-disagreement per bucket of a diagnostic `key`."""
    print(f"\n  by {key}:")
    print(f"    {'bucket':>16} {'n':>6} {'sign_dis':>9} {'fail':>7} "
          f"{'MAE':>8} {'med|err|':>9}")
    vals = np.array([r[key] for r in rows])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        grp = [r for r, mm in zip(rows, m) if mm]
        if not grp:
            continue
        comp = [r for r in grp if r["d_exact"] is not None
                and np.isfinite(r["dhat"])]
        if comp:
            errs = np.array([abs(r["dhat"] - r["d_exact"]) for r in comp])
            sdis = np.mean([(r["dhat"] > 0) != (r["d_exact"] > 0) for r in comp])
            mae = errs.mean(); med = np.median(errs)
        else:
            sdis = mae = med = float("nan")
        fail = np.mean([(r["d_exact"] is None) or (not np.isfinite(r["dhat"]))
                        or (r["d_exact"] is not None and np.isfinite(r["dhat"])
                            and abs(r["dhat"] - r["d_exact"]) > tol) for r in grp])
        lbl = f"[{lo:g},{hi:g})"
        print(f"    {lbl:>16} {len(grp):>6} {sdis:>9.3f} {fail:>7.3f} "
              f"{mae:>8.3f} {med:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--nverify", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="stress_test_proxy.json")
    args = ap.parse_args()

    families = ["local", "uniform", "wide", "extreme_z", "multi_advance",
                "near_seed_z_small"]
    rng = random.Random(args.seed)
    tasks = []
    for k in range(args.n):
        fam = families[k % len(families)]
        sh, dv, zn, zd = sample_family(rng, fam)
        tasks.append((sh, dv, zn, zd, args.nrank, args.nverify, fam))

    print(f"[STRESS] {args.n} trajectories, proxy N={args.nrank}, "
          f"exact 2N={2*args.nverify}, tol={args.tol}, {args.workers} workers")
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(_one, tasks, chunksize=16)
    errs = [r for r in res if "err" in r]
    rows = [r for r in res if "err" not in r]
    print(f"  computed {len(rows)} in {time.time()-t0:.0f}s; "
          f"{len(errs)} hard exceptions")

    # derived diagnostics
    for r in rows:
        r["gap12"] = r["lam1"] - r["lam2"]
        r["gap23"] = r["lam2"] - r["lam3"]
        r["abs_lam1"] = abs(r["lam1"])
        r["dhat_mag"] = abs(r["dhat"]) if np.isfinite(r["dhat"]) else 1e9

    comp = [r for r in rows if r["d_exact"] is not None and np.isfinite(r["dhat"])]
    nan_dhat = sum(1 for r in rows if not np.isfinite(r["dhat"]))
    no_exact = sum(1 for r in rows if r["d_exact"] is None)
    sdis = np.mean([(r["dhat"] > 0) != (r["d_exact"] > 0) for r in comp])
    mae = np.mean([abs(r["dhat"] - r["d_exact"]) for r in comp])
    print(f"\n  overall: computable {len(comp)}/{len(rows)}  "
          f"NaN dhat={nan_dhat}  exact-fail={no_exact}")
    print(f"  sign-disagreement={sdis:.3f}  MAE={mae:.3f}")

    # per-family summary
    print("\n  by family:")
    print(f"    {'family':>18} {'n':>5} {'sign_dis':>9} {'fail':>7} {'MAE':>8}")
    fam_stats = {}
    for fam in families:
        g = [r for r in rows if r["family"] == fam]
        c = [r for r in g if r["d_exact"] is not None and np.isfinite(r["dhat"])]
        sd = np.mean([(r["dhat"] > 0) != (r["d_exact"] > 0) for r in c]) if c else float("nan")
        fl = np.mean([(r["d_exact"] is None) or (not np.isfinite(r["dhat"]))
                      or (abs(r["dhat"] - r["d_exact"]) > args.tol if r["d_exact"] is not None and np.isfinite(r["dhat"]) else False)
                      for r in g]) if g else float("nan")
        ma = np.mean([abs(r["dhat"] - r["d_exact"]) for r in c]) if c else float("nan")
        fam_stats[fam] = {"n": len(g), "sign_dis": float(sd), "fail": float(fl),
                          "mae": float(ma)}
        print(f"    {fam:>18} {len(g):>5} {sd:>9.3f} {fl:>7.3f} {ma:>8.3f}")

    # diagnostic buckets
    bucketize(rows, "gap12", [0, .25, .5, 1, 2, 4, 100], args.tol)
    bucketize(rows, "gap23", [0, .1, .25, .5, 1, 100], args.tol)
    bucketize(rows, "abs_lam1", [0, .5, 1, 2, 4, 8, 100], args.tol)
    bucketize(rows, "dhat_mag", [0, .5, 1, 1.1, 1.5, 5, 1e12], args.tol)

    # depth-bias study on a random subset (recompute dhat at several depths)
    depths = [60, 120, 240, 480]
    drift = []
    sub_tasks = rng.sample(tasks, min(40, len(tasks)))
    with Pool(args.workers) as pool:
        dres = pool.map(_depth, [(t[0], t[1], t[2], t[3], depths) for t in sub_tasks])
    for d in dres:
        v = [d[N] for N in depths if np.isfinite(d[N])]
        if len(v) >= 2:
            drift.append(max(v) - min(v))
    drift = np.array(drift) if drift else np.array([np.nan])
    print(f"\n  depth-bias (dhat drift over N={depths}): "
          f"median={np.nanmedian(drift):.3f}  p90={np.nanpercentile(drift,90):.3f}  "
          f"max={np.nanmax(drift):.3f}")

    out = {"n": len(rows), "computable": len(comp), "nan_dhat": nan_dhat,
           "exact_fail": no_exact, "hard_exceptions": len(errs),
           "sign_disagreement": float(sdis), "mae": float(mae),
           "tol": args.tol, "by_family": fam_stats,
           "depth_drift_median": float(np.nanmedian(drift)),
           "depth_drift_max": float(np.nanmax(drift))}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
