"""
discover.py
===========
Active-learning discovery over the widened 6F5 space  theta = (shift, dir, z).

Beyond enrich.py (fixed dir/z): here we also search the advancing-direction
pattern `dir` and the parameter `z = z_num/z_den`, and we run an ADAPTIVE loop
(Test 2 / adaptive-discovery in the validity framework) against a random control.

Cheap signals per candidate (float QR Lyapunov):
    delta_hat(N)              spectral surrogate  = -lam2/lam1
    drift = |dhat(N) - dhat(N/2)|     finite-N "uncertainty" (still climbing?)
    acquisition (UCB)         a = delta_hat + kappa * drift
The +0.37 discovery was a *climbing* point (dhat 0.35 -> 0.40), so high drift is
a real signal for "this limit may be much larger" — exactly what we want to spend
the expensive Tier-2 budget on.

Loop (per round):
  1. Build a candidate pool (round 0: random; later: neighborhoods of best hits
     + fresh random), score all cheaply, rank by acquisition.
  2. Verify the top-B with the expensive Tier-2 arithmetic delta.
  3. Promote verified hits (delta_arith > seed) to seeds for the next round.
  4. A parallel RANDOM control verifies B random candidates each round.
Track best / mean verified delta per round: adaptive should pull away from random.

Usage:
  python3 discover.py --rounds 4 --pool 1000 --budget 12 --nrank 180 \
                      --nverify 120 --kappa 1.5
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction

import numpy as np

from enrich import (build_M_float, arith_delta, spectral_dhat,
                    SEED_SHIFT, SEED_DIR)
from spectral_delta import lyapunov_spectrum

DIM = 6

# ── widened search space ─────────────────────────────────────────────────────
# dir: sparse advancing patterns (single / paired roots), the regime where 6F5
# trajectories actually converge.  position i<6 -> numerator root, i>=6 -> denom.
def _dir_pool():
    pats = []
    for i in range(11):                      # single advancing root, +1
        d = [0] * 11; d[i] = 1; pats.append(d)
    for i in range(6, 11):                    # single advancing root, +2 (faster)
        d = [0] * 11; d[i] = 2; pats.append(d)
    # a few paired patterns seen in real hits
    for (i, j) in [(4, 5), (6, 8), (8, 9), (4, 8)]:
        d = [0] * 11; d[i] = 1; d[j] = 1; pats.append(d)
    pats.append(list(SEED_DIR))
    return pats

DIR_POOL = _dir_pool()


def _z_pool():
    zs = set()
    for q in (1, 2, 3, 4, 5, 6, 10, 20):
        for p in range(-2 * q, 2 * q + 1):
            if p == 0:
                continue
            fr = Fraction(p, q)
            # |z| >= 1 diverges (no convergent ratio) -> restrict to (-1, 1)
            if 0.1 <= abs(float(fr)) < 1.0:
                zs.add((fr.numerator, fr.denominator))
    return sorted(zs)

Z_POOL = _z_pool()


def sample_candidate(rng, box, seed=None):
    if seed is None:
        shift = [SEED_SHIFT[i] + rng.randint(-box, box) for i in range(11)]
        dirv = list(rng.choice(DIR_POOL))
        zn, zd = rng.choice(Z_POOL)
    else:
        shift = [seed["shift"][i] + rng.randint(-1, 1) for i in range(11)]
        dirv = seed["dir"] if rng.random() < 0.7 else list(rng.choice(DIR_POOL))
        if rng.random() < 0.7:
            zn, zd = seed["z_num"], seed["z_den"]
        else:
            zn, zd = rng.choice(Z_POOL)
    return {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd}


# ── cheap scoring with finite-N drift (uncertainty) ──────────────────────────
def score(cand, N):
    d_full, lam = spectral_dhat(cand["shift"], cand["dir"],
                                cand["z_num"], cand["z_den"], N)
    if not np.isfinite(d_full):
        return None
    d_half, _ = spectral_dhat(cand["shift"], cand["dir"],
                              cand["z_num"], cand["z_den"], N // 2)
    drift = abs(d_full - d_half) if np.isfinite(d_half) else 0.0
    return {"dhat": d_full, "drift": drift, "lam1": float(lam[0])}


def key(c):
    return (tuple(c["shift"]), tuple(c["dir"]), c["z_num"], c["z_den"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=12, help="Tier-2 verifies/round")
    ap.add_argument("--box", type=int, default=2)
    ap.add_argument("--nrank", type=int, default=180)
    ap.add_argument("--nverify", type=int, default=120)
    ap.add_argument("--kappa", type=float, default=1.5, help="UCB drift weight")
    ap.add_argument("--dhat_floor", type=float, default=-0.05,
                    help="only reward drift when dhat exceeds this")
    ap.add_argument("--promote", type=float, default=0.0,
                    help="verified delta above which a hit becomes a seed")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="discover_results.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"Search space: dir patterns={len(DIR_POOL)}, z values={len(Z_POOL)}, "
          f"shift box=±{args.box}")
    print(f"Acquisition = dhat + {args.kappa}*drift  (UCB on finite-N climb)\n")

    seeds = []          # verified hits promoted across rounds
    seen = set()
    adaptive_log, random_log = [], []
    best_overall = None

    def verify(cands, label, rnd):
        out = []
        for i, c in enumerate(cands):
            da = arith_delta(c["shift"], c["dir"], c["z_num"], c["z_den"],
                             args.nverify, c["lam1"])
            if da is None:
                continue
            c["delta_arith"] = da
            out.append(c)
            print(f"  [{label} r{rnd} {i+1}/{len(cands)}] "
                  f"acq={c['acq']:+.4f} dhat={c['dhat']:+.4f} "
                  f"drift={c['drift']:.4f} -> darith={da:+.4f}  "
                  f"z={c['z_num']}/{c['z_den']}")
        return out

    for rnd in range(args.rounds):
        # build pool: fresh random + neighborhoods of current seeds
        pool, tries = [], 0
        while len(pool) < args.pool and tries < args.pool * 5:
            tries += 1
            seed = rng.choice(seeds) if (seeds and rng.random() < 0.5) else None
            c = sample_candidate(rng, args.box, seed)
            if key(c) in seen:
                continue
            seen.add(key(c))
            s = score(c, args.nrank)
            if s is None:
                continue
            c.update(s)
            # gated UCB: drift only rewarded once dhat is near the boundary,
            # otherwise huge-drift divergent junk dominates the ranking.
            bonus = args.kappa * c["drift"] if c["dhat"] > args.dhat_floor else 0.0
            c["acq"] = c["dhat"] + bonus
            pool.append(c)

        pool.sort(key=lambda c: c["acq"], reverse=True)
        top = pool[:args.budget]
        rand = rng.sample(pool, min(args.budget, len(pool)))

        print(f"=== ROUND {rnd}  (pool={len(pool)}, seeds={len(seeds)}) ===")
        adv = verify(top, "ADAPT", rnd)
        ctl = verify(rand, "RAND ", rnd)

        adv_d = [c["delta_arith"] for c in adv]
        ctl_d = [c["delta_arith"] for c in ctl]
        adaptive_log.append({"round": rnd,
                             "best": max(adv_d) if adv_d else None,
                             "mean": float(np.mean(adv_d)) if adv_d else None,
                             "hit_rate_pos": float(np.mean([d > 0 for d in adv_d])) if adv_d else None})
        random_log.append({"round": rnd,
                           "best": max(ctl_d) if ctl_d else None,
                           "mean": float(np.mean(ctl_d)) if ctl_d else None,
                           "hit_rate_pos": float(np.mean([d > 0 for d in ctl_d])) if ctl_d else None})

        # promote positive adaptive hits to seeds (focus the next round there)
        for c in adv:
            if c["delta_arith"] > args.promote:
                seeds.append(c)
        seeds = sorted(seeds, key=lambda c: c["delta_arith"], reverse=True)[:8]
        if adv:
            rbest = max(adv, key=lambda c: c["delta_arith"])
            if best_overall is None or rbest["delta_arith"] > best_overall["delta_arith"]:
                best_overall = rbest
        print(f"  -> adaptive best={adaptive_log[-1]['best']}, "
              f"random best={random_log[-1]['best']}, seeds carried={len(seeds)}\n")

    # ── summary ──────────────────────────────────────────────────────────────
    print("=" * 66)
    print("ADAPTIVE DISCOVERY SUMMARY   (best delta_arith per round)")
    print("=" * 66)
    print(f"{'round':>6} {'adapt_best':>11} {'rand_best':>10} "
          f"{'adapt_mean':>11} {'rand_mean':>10} {'adapt_hit+':>11}")
    def f(x):
        return f"{x:+.4f}" if isinstance(x, float) else " None "
    for a, r in zip(adaptive_log, random_log):
        hr = a["hit_rate_pos"]
        hr_s = f"{hr:.2f}" if hr is not None else "NA"
        print(f"{a['round']:>6} {f(a['best']):>11} {f(r['best']):>10} "
              f"{f(a['mean']):>11} {f(r['mean']):>10} {hr_s:>11}")
    if best_overall:
        print(f"\nBEST OVERALL: delta_arith={best_overall['delta_arith']:+.4f}")
        print(f"  shift={best_overall['shift']}")
        print(f"  dir  ={best_overall['dir']}   z={best_overall['z_num']}/{best_overall['z_den']}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "adaptive": adaptive_log,
                   "random": random_log, "best": best_overall,
                   "final_seeds": seeds}, f, indent=2)
    print(f"\nJSON -> {args.out}")


if __name__ == "__main__":
    main()
