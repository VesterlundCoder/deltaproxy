"""
guided_e_search.py
==================
Guided search for e-related limits near the e-2 CF hit.

Instead of blind delta (self-convergence rate), uses CALIBRATED delta
against known e-related targets:

    δ_cal(T) = -log|L - T| / log|q| - 1

where L = p_{2N}/q_{2N} is the convergent at depth 2N.

If the CF converges to T:  |L-T| → 0  =>  δ_cal → +∞  (large positive)
If the CF converges elsewhere: |L-T| → const  =>  δ_cal → -1

Targets: e-2, e-1/2, e-3/2, e-1/4, e-3/4, e-3, e-5/2, e^(-1), e^(-1/2), ...

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
import math
import random
from multiprocessing import Pool
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg

DIM = 6
BOX = 6
BASE_SHIFT = [-1, -1, 0, -1, -1, 1, -2, -2, -1, -2, 1]
BASE_DIR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
ZN, ZD = 1, 1
N_DEPTH = 2000  # exact depth (2N = 4000 matrix multiplications)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "cluster_out", "guided_e_search")


def build_targets():
    """Build dictionary of e-related target values."""
    mp.mp.dps = 100
    e = mp.e
    targets = {}

    # e - n for integer n
    for n in range(-3, 6):
        name = f"e-{n}" if n > 0 else f"e+{-n}" if n < 0 else "e"
        targets[name] = e - n

    # e - p/q for user-specified rationals
    for pq in [(1,2), (3,2), (1,4), (3,4), (5,2), (7,3), (8,3), (9,4), (11,4)]:
        p, q = pq
        targets[f"e-{p}/{q}"] = e - mp.mpf(p) / mp.mpf(q)

    # Negative: n - e
    for n in range(0, 5):
        targets[f"{n}-e"] = mp.mpf(n) - e

    # e^(±1/n)
    for n in [1, 2, 3, 4]:
        targets[f"e^(1/{n})"] = mp.e ** (mp.mpf(1) / n)
        targets[f"e^(-1/{n})"] = mp.e ** (mp.mpf(-1) / n)

    # e^(±p/q)
    for p, q in [(1,2), (3,2), (1,4), (3,4)]:
        targets[f"e^({p}/{q})"] = mp.e ** (mp.mpf(p) / mp.mpf(q))
        targets[f"e^(-{p}/{q})"] = mp.e ** (mp.mpf(-p) / mp.mpf(q))

    # e^(1/n) - 1 (near 0.7 range)
    for n in [2, 3, 4]:
        targets[f"e^(1/{n})-1"] = mp.e ** (mp.mpf(1) / n) - 1

    # log-related
    targets["1/log(2)"] = 1 / mp.log(2)
    targets["log(2)"] = mp.log(2)

    return targets


def gen_trajectories(n_target=1000):
    """Generate trajectories near the e-2 hit."""
    nsh = len(BASE_SHIFT)
    trajs = set()

    # Base
    trajs.add((tuple(BASE_SHIFT), tuple(BASE_DIR), ZN, ZD))

    # Perturb shift by ±1 on 1-3 positions, keep base dir
    for i in range(nsh):
        for d in [-1, 1]:
            s = list(BASE_SHIFT)
            s[i] += d
            if all(-BOX <= x <= BOX for x in s):
                trajs.add((tuple(s), tuple(BASE_DIR), ZN, ZD))

    # Perturb shift by ±2 on 1 position
    for i in range(nsh):
        for d in [-2, 2]:
            s = list(BASE_SHIFT)
            s[i] += d
            if all(-BOX <= x <= BOX for x in s):
                trajs.add((tuple(s), tuple(BASE_DIR), ZN, ZD))

    # Paired ±1 perturbations
    for i in range(nsh):
        for j in range(i+1, nsh):
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    s = list(BASE_SHIFT)
                    s[i] += di
                    s[j] += dj
                    if all(-BOX <= x <= BOX for x in s):
                        trajs.add((tuple(s), tuple(BASE_DIR), ZN, ZD))

    # Try different dir vectors (move advancing root)
    for pos in range(nsh):
        d = [0] * nsh
        d[pos] = 1
        trajs.add((tuple(BASE_SHIFT), tuple(d), ZN, ZD))
        # Also with z = -1
        trajs.add((tuple(BASE_SHIFT), tuple(d), -1, 1))

    # Perturbed shift + different advancing positions
    for i in range(nsh):
        s = list(BASE_SHIFT)
        s[i] += 1
        if all(-BOX <= x <= BOX for x in s):
            for pos in range(nsh):
                d = [0] * nsh
                d[pos] = 1
                trajs.add((tuple(s), tuple(d), ZN, ZD))

    # z = -1 with base params
    trajs.add((tuple(BASE_SHIFT), tuple(BASE_DIR), -1, 1))

    # Paired dir (two advancing roots)
    for (i, j) in [(9, 10), (5, 10), (0, 10), (4, 10)]:
        d = [0] * nsh
        d[i] = 1
        d[j] = 1
        trajs.add((tuple(BASE_SHIFT), tuple(d), ZN, ZD))
        trajs.add((tuple(BASE_SHIFT), tuple(d), -1, 1))

    # Random perturbations to fill
    rng = random.Random(42)
    while len(trajs) < n_target:
        s = list(BASE_SHIFT)
        n_pert = rng.randint(1, 3)
        for _ in range(n_pert):
            idx = rng.randint(0, nsh - 1)
            s[idx] += rng.choice([-1, 1])
        if all(-BOX <= x <= BOX for x in s):
            # Random dir
            d = [0] * nsh
            d[rng.randint(0, nsh-1)] = 1
            z = rng.choice([(1, 1), (-1, 1)])
            trajs.add((tuple(s), tuple(d), z[0], z[1]))

    return list(trajs)[:n_target]


def calibrated_eval(args):
    """Compute calibrated delta against all targets for one trajectory."""
    shift, dirv, zn, zd, targets_serialized = args

    # Rebuild targets in this process
    mp.mp.dps = 100
    targets = {}
    for name, val_str in targets_serialized.items():
        targets[name] = mp.mpf(val_str)

    try:
        # Exact integer double-depth walk
        dim = DIM
        P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
        snapN = None
        for n in range(1, 2 * N_DEPTH + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, list(shift), list(dirv), zn, zd, dim), dim)
            if n == N_DEPTH:
                snapN = [row[:] for row in P]

        last = dim - 1
        results = []

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                q2 = P[j][last]
                p2 = P[i][last]
                if q2 == 0 or p2 == 0:
                    continue

                # Limit L = p2/q2 at depth 2N
                L = mp.mpf(p2) / mp.mpf(q2)
                log_q2 = cg.log_bigint(q2)
                if log_q2 <= 0:
                    continue

                # Also compute blind delta
                qn = snapN[j][last]
                pn = snapN[i][last]
                blind_d = None
                if qn != 0 and pn != 0:
                    cross = pn * q2 - p2 * qn
                    if cross != 0:
                        log_err = cg.log_bigint(cross) - cg.log_bigint(qn) - cg.log_bigint(q2)
                        log_qn = cg.log_bigint(qn)
                        if log_qn > 0:
                            blind_d = -(1.0 + log_err / log_qn)

                # Calibrated delta against each target
                best_cal = None
                best_target = None
                for tname, T in targets.items():
                    diff = abs(L - T)
                    if diff == 0:
                        cal_d = float('inf')
                    else:
                        log_diff = mp.log(diff)
                        cal_d = float(-log_diff / log_q2 - 1.0)
                    if best_cal is None or cal_d > best_cal:
                        best_cal = cal_d
                        best_target = tname

                results.append({
                    "pair": (i, j),
                    "L": mp.nstr(L, 40),
                    "blind_delta": blind_d,
                    "best_cal_delta": best_cal,
                    "best_target": best_target,
                    "log_q": log_q2,
                })

        # Find best result by calibrated delta
        if not results:
            return None
        best = max(results, key=lambda r: r["best_cal_delta"])
        best["shift"] = list(shift)
        best["dir"] = list(dirv)
        best["z_num"] = zn
        best["z_den"] = zd

        # Also show top-3 calibrated targets for the best pair
        # Recompute for the best pair
        bi, bj = best["pair"]
        p2, q2 = P[bi][last], P[bj][last]
        L_best = mp.mpf(p2) / mp.mpf(q2)
        all_cal = {}
        for tname, T in targets.items():
            diff = abs(L_best - T)
            if diff == 0:
                all_cal[tname] = float('inf')
            else:
                all_cal[tname] = float(-mp.log(diff) / cg.log_bigint(q2) - 1.0)
        # Sort by calibrated delta
        sorted_cal = sorted(all_cal.items(), key=lambda x: -x[1])
        best["all_targets_top5"] = sorted_cal[:5]

        return best

    except Exception as e:
        return {"error": str(e)[:200], "shift": list(shift), "dir": list(dirv)}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # Build targets
    targets = build_targets()
    print(f"[guided] {len(targets)} e-related targets:", flush=True)
    for name, val in sorted(targets.items()):
        print(f"  {name:20s} = {mp.nstr(val, 25)}", flush=True)

    # Serialize targets for multiprocessing
    targets_serialized = {name: mp.nstr(val, 80) for name, val in targets.items()}

    # Generate trajectories
    trajs = gen_trajectories(1000)
    print(f"\n[guided] {len(trajs)} trajectories near e-2 hit (depth={N_DEPTH})",
          flush=True)

    # Run calibrated evaluation
    tasks = [(s, d, zn, zd, targets_serialized) for s, d, zn, zd in trajs]
    print(f"[guided] Running on 6 cores...", flush=True)

    with Pool(6) as pool:
        results = pool.map(calibrated_eval, tasks, chunksize=5)

    elapsed = time.time() - t0

    # Filter out errors
    valid = [r for r in results if r and "error" not in r]
    errors = [r for r in results if r and "error" in r]
    print(f"\n[guided] {len(valid)} valid results, {len(errors)} errors in {elapsed:.0f}s",
          flush=True)

    # Sort by best calibrated delta
    valid.sort(key=lambda r: r.get("best_cal_delta", -999), reverse=True)

    # Print top 20
    print(f"\n{'='*80}")
    print(f"TOP 20 by calibrated delta (depth={N_DEPTH}):")
    print(f"{'='*80}")
    print(f"{'#':>3s}  {'cal_δ':>8s}  {'blind_δ':>8s}  {'target':>12s}  {'L':>30s}  {'shift'}")
    print(f"{'-'*80}")
    for i, r in enumerate(valid[:20]):
        cal = r["best_cal_delta"]
        blind = r.get("blind_delta")
        blind_s = f"{blind:+.4f}" if blind is not None else "  N/A "
        target = r["best_target"]
        L = r["L"][:30]
        s = r["shift"]
        print(f"{i+1:3d}  {cal:+8.4f}  {blind_s}  {target:>12s}  {L:>30s}  {s}")

    # Show detailed target breakdown for top 5
    print(f"\n{'='*80}")
    print(f"DETAILED target breakdown for top 5:")
    print(f"{'='*80}")
    for i, r in enumerate(valid[:5]):
        print(f"\n--- #{i+1} ---")
        print(f"  shift = {r['shift']}")
        print(f"  dir   = {r['dir']}")
        print(f"  z     = {r['z_num']}/{r['z_den']}")
        print(f"  L     = {r['L']}")
        print(f"  blind_delta = {r.get('blind_delta', 'N/A')}")
        print(f"  pair  = {r['pair']}")
        print(f"  Top-5 calibrated targets:")
        for tname, cal in r.get("all_targets_top5", []):
            marker = " <<<" if cal > 0.1 else ""
            print(f"    {tname:20s}  cal_δ = {cal:+.6f}{marker}")

    # Save all results
    out_file = os.path.join(OUT, "guided_results.jsonl")
    with open(out_file, "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")
    print(f"\n[guided] All results -> {out_file}")

    # Summary stats
    cal_deltas = [r["best_cal_delta"] for r in valid]
    blind_deltas = [r["blind_delta"] for r in valid if r["blind_delta"] is not None]

    print(f"\n[guided] SUMMARY:")
    print(f"  trajectories:  {len(valid)}")
    print(f"  depth:         {N_DEPTH} (2N={2*N_DEPTH} matrix mults)")
    print(f"  cal_δ > 0:     {sum(1 for d in cal_deltas if d > 0)}  (converging to a target)")
    print(f"  cal_δ > 0.1:   {sum(1 for d in cal_deltas if d > 0.1)}  (strong convergence)")
    print(f"  cal_δ > 0.5:   {sum(1 for d in cal_deltas if d > 0.5)}  (very strong)")
    print(f"  cal_δ max:     {max(cal_deltas):.4f}")
    print(f"  blind_δ max:   {max(blind_deltas):.4f}" if blind_deltas else "")
    print(f"  time:          {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
