"""
neighborhood_e2.py
==================
Deep classical matrix-multiplication study around the e-2 CF hit.

Generates 100k perturbations of the base shift vector, screens with
float64 Lyapunov proxy (fast), then runs exact integer double-depth
delta + full PSLQ battery on every survivor with delta >= -0.1.

Base hit:
  shift = [-1,-1,0,-1,-1,1,-2,-2,-1,-2,1]
  dir   = [0,0,0,0,0,0,0,0,0,0,1]
  z = 1/1, dim = 6, L = e-2
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
from multiprocessing import Pool

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg
import pslq_companion as pc
from proxy_batch import lyapunov_batch, ratios_batch

DIM = 6
BOX = 6
BASE_SHIFT = np.array([-1, -1, 0, -1, -1, 1, -2, -2, -1, -2, 1], dtype=np.int64)
BASE_DIR = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
ZN, ZD = 1, 1
DELTA_MIN = -0.1
N_PROXY = 120
N_EXACT = 300
N_PSLQ = 500
DPS = 400

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "cluster_out", "neighborhood_e2")


def gen_perturbations(n_target=100000):
    nsh = len(BASE_SHIFT)
    perturbs = set()
    perturbs.add(tuple(BASE_SHIFT.tolist()))
    for i in range(nsh):
        for d in [-2, -1, 1, 2]:
            s = BASE_SHIFT.copy()
            s[i] += d
            if all(-BOX <= x <= BOX for x in s):
                perturbs.add(tuple(s.tolist()))
    for i in range(nsh):
        for j in range(i + 1, nsh):
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    s = BASE_SHIFT.copy()
                    s[i] += di
                    s[j] += dj
                    if all(-BOX <= x <= BOX for x in s):
                        perturbs.add(tuple(s.tolist()))
    for i in range(nsh):
        for j in range(i + 1, nsh):
            for k in range(j + 1, nsh):
                for di in [-1, 1]:
                    s = BASE_SHIFT.copy()
                    s[i] += di
                    s[j] += di
                    s[k] += di
                    if all(-BOX <= x <= BOX for x in s):
                        perturbs.add(tuple(s.tolist()))
    rng = random.Random(20260707)
    while len(perturbs) < n_target:
        s = BASE_SHIFT.copy()
        n_pert = rng.randint(1, 5)
        for _ in range(n_pert):
            idx = rng.randint(0, nsh - 1)
            s[idx] += rng.choice([-2, -1, 1, 2])
        if all(-BOX <= x <= BOX for x in s):
            perturbs.add(tuple(s.tolist()))
    return [np.array(p, dtype=np.int64) for p in list(perturbs)[:n_target]]


def screen_batch(args):
    shifts, dirv, zn, zd, dim, nsteps = args
    B = len(shifts)
    shift_arr = np.array(shifts, dtype=np.float64)
    dir_arr = np.tile(dirv.astype(np.float64), (B, 1))
    zn_arr = np.full(B, float(zn))
    zd_arr = np.full(B, float(zd))
    lam = lyapunov_batch(shift_arr, dir_arr, zn_arr, zd_arr, dim, nsteps, np)
    r = ratios_batch(lam, np)
    return r[:, 0]


def exact_delta_one(args):
    shift, dirv = args
    try:
        mp.mp.dps = 200
        d, L, pair = pc.delta_and_limit(
            list(shift), list(dirv), ZN, ZD, DIM, N_EXACT)
        if d is None:
            return None
        return {
            "shift": list(shift),
            "dir": list(dirv),
            "z_num": ZN, "z_den": ZD,
            "delta_exact": float(d),
            "L_preview": mp.nstr(L, 30),
            "pair": pair,
        }
    except Exception:
        return None


def pslq_one(rec):
    try:
        mp.mp.dps = DPS
        d, L, pair = pc.delta_and_limit(
            rec["shift"], rec["dir"], ZN, ZD, DIM, N_PSLQ)
        rec["delta_pslq"] = float(d)
        rec["L"] = mp.nstr(L, 50)
        hits = pc.pslq_identify(L, 100000, 50000)
        if hits:
            real = [h for h in hits
                    if h.get("coeffs", [0])[0] != 0
                    and any(c != 0 for c in h.get("coeffs", [0])[2:])]
            rec["pslq_all"] = hits
            rec["pslq_real"] = real
            return len(real) > 0
        rec["pslq_all"] = []
        rec["pslq_real"] = []
        return False
    except Exception as e:
        rec["error"] = str(e)[:100]
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    shifts = gen_perturbations(100000)
    nsh = len(BASE_SHIFT)
    print(f"[nbhd] {len(shifts)} perturbed shifts in {time.time()-t0:.1f}s",
          flush=True)

    dirs = [BASE_DIR]
    for i in range(nsh):
        d = np.zeros(nsh, dtype=np.int64)
        d[i] = 1
        dirs.append(d)
    print(f"[nbhd] {len(dirs)} dirs -> {len(shifts)*len(dirs)} total trajectories",
          flush=True)

    # Phase 1: spectral proxy
    print(f"[nbhd] Phase 1: proxy screen (N={N_PROXY}, band=[-0.15,0.15])",
          flush=True)
    t1 = time.time()
    proxy_survivors = []
    BATCH = 5000
    for di, dirv in enumerate(dirs):
        for b0 in range(0, len(shifts), BATCH):
            b1 = min(b0 + BATCH, len(shifts))
            r2 = screen_batch((shifts[b0:b1], dirv, ZN, ZD, DIM, N_PROXY))
            finite = np.isfinite(r2)
            hit = finite & (r2 > -0.15) & (r2 < 0.15)
            idx = np.nonzero(hit)[0]
            for j in idx:
                proxy_survivors.append((shifts[b0 + j], dirv))
        if (di + 1) % 5 == 0:
            print(f"  dir {di+1}/{len(dirs)}, proxy survivors: {len(proxy_survivors)}",
                  flush=True)
    print(f"[nbhd] Phase 1: {len(proxy_survivors)} proxy survivors "
          f"from {len(shifts)*len(dirs)} in {time.time()-t1:.0f}s", flush=True)

    # Phase 2: exact delta
    print(f"[nbhd] Phase 2: exact delta (N={N_EXACT}, delta>={DELTA_MIN})",
          flush=True)
    t2 = time.time()
    tasks = [(s, d) for s, d in proxy_survivors]
    with Pool(6) as pool:
        results = pool.map(exact_delta_one, tasks, chunksize=20)
    survivors = [r for r in results if r is not None and r["delta_exact"] >= DELTA_MIN]
    print(f"[nbhd] Phase 2: {len(survivors)} exact survivors "
          f"from {len(proxy_survivors)} in {time.time()-t2:.0f}s", flush=True)

    sf = open(os.path.join(OUT, "survivors.jsonl"), "w")
    for r in survivors:
        sf.write(json.dumps(r) + "\n")
    sf.close()

    # Phase 3: PSLQ
    print(f"[nbhd] Phase 3: PSLQ on {len(survivors)} survivors (N={N_PSLQ}, dps={DPS})",
          flush=True)
    t3 = time.time()
    hits_file = open(os.path.join(OUT, "pslq_hits.jsonl"), "w")
    all_file = open(os.path.join(OUT, "all_pslq_results.jsonl"), "w")
    n_hits = 0
    for i, rec in enumerate(survivors):
        is_hit = pslq_one(rec)
        all_file.write(json.dumps(rec) + "\n")
        all_file.flush()
        if is_hit:
            n_hits += 1
            hits_file.write(json.dumps(rec) + "\n")
            hits_file.flush()
            real = rec.get("pslq_real", [])
            print(f"  [{i+1}/{len(survivors)}] HIT delta={rec['delta_pslq']:.4f} "
                  f"L={rec['L'][:25]}... {len(real)} real relations", flush=True)
            for r in real[:3]:
                print(f"    -> {r['type']}: {r['names']} coeffs={r['coeffs']}",
                      flush=True)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(survivors)}] processed, {n_hits} hits "
                  f"({time.time()-t3:.0f}s)", flush=True)
    hits_file.close()
    all_file.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"[nbhd] DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  trajectories:    {len(shifts)*len(dirs)}")
    print(f"  proxy survivors: {len(proxy_survivors)}")
    print(f"  exact survivors: {len(survivors)} (delta >= {DELTA_MIN})")
    print(f"  PSLQ hits:       {n_hits}")
    print(f"  -> {OUT}/")

    survivors.sort(key=lambda r: r.get("delta_pslq", r.get("delta_exact", -999)), reverse=True)
    print(f"\n  Top 10 by delta:")
    for r in survivors[:10]:
        d = r.get("delta_pslq", r.get("delta_exact", "?"))
        L = r.get("L", r.get("L_preview", "?"))[:30]
        real = r.get("pslq_real", [])
        hit_str = f"  {real[0]['type']}:{real[0]['coeffs']}" if real else ""
        print(f"    delta={d:+.4f}  L={L}...{hit_str}")


if __name__ == "__main__":
    main()
