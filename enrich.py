"""
enrich.py
=========
Discovery-engine test: does the cheap spectral ranker compress the 11-D search
space?  (Operationalises the enrichment factor — the strongest success
criterion.)

11-D parameter space (verify_znegsweep_hits.py convention):
    theta = shift[11]   (6 numerator shifts + 5 denominator shifts)
    f_i = shift[i]   + n*dir[i]   + 1      (i=0..5)
    g_j = shift[6+j] + n*dir[6+j] + 2      (j=0..4)
    h_f = -z_num,  h_g = z_den
    c[0] = h_f*ef[6];  c[k] = h_g*eg[6-k] + h_f*ef[6-k]  (k=1..5)
    M(n): 6x6 companion, subdiag=1, last col = c

Protocol:
  1. Draw a large POOL of random integer shift vectors in a box.
  2. Score every pool point with the CHEAP spectral ranker  delta_hat = -lam2/lam1
     (float QR Lyapunov, depth N_rank).                       [~ms / point]
  3. MODEL set    = top-K by delta_hat.
     RANDOM set   = uniform sample from the pool.
  4. Verify BOTH with the EXPENSIVE Tier-2 arithmetic delta (high-precision
     integer convergent, best of 30 row pairs).
  5. hit = (delta_arith > threshold).
     enrichment = hit_rate(model) / hit_rate(random).

A "hit" is a high-delta (irrationality-relevant) 6F5 trajectory; these are rare
in random shift space but cluster near the g4 boundary the ranker can smell.

Usage:
  python3 enrich.py --pool 4000 --topk 40 --nrand 80 \
                    --nrank 200 --nverify 160 --thresh 0.08
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import mpmath as mp

from spectral_delta import lyapunov_spectrum

DIM = 6

# f0g4 hit-B seed (z=7/20): centre of the search box
SEED_SHIFT = [-2, -2, 0, 0, -2, 0, -2, -2, -2, 0, -2]
SEED_DIR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
Z_NUM, Z_DEN = 7, 20


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builders
# ─────────────────────────────────────────────────────────────────────────────
def _esym_float(vals):
    n = len(vals)
    e = np.zeros(n + 1)
    e[0] = 1.0
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def build_M_float(n, shift, dirv, z_num, z_den):
    h_f, h_g = float(-z_num), float(z_den)
    f = [shift[i] + n * dirv[i] + 1 for i in range(6)]
    g = [shift[6 + j] + n * dirv[6 + j] + 2 for j in range(5)]
    ef = _esym_float(f)
    eg = np.concatenate([_esym_float(g), [0.0]])
    c = np.zeros(DIM)
    c[0] = h_f * ef[6]
    for k in range(1, DIM):
        c[k] = h_g * eg[6 - k] + h_f * ef[6 - k]
    M = np.zeros((DIM, DIM))
    for rr in range(1, DIM):
        M[rr, rr - 1] = 1.0
    M[:, DIM - 1] = c
    return M


def _esym_mp(vals):
    n = len(vals)
    e = [mp.mpf(0)] * (n + 1)
    e[0] = mp.mpf(1)
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def build_M_mp(n, shift, dirv, z_num, z_den):
    h_f, h_g = mp.mpf(-z_num), mp.mpf(z_den)
    f = [mp.mpf(shift[i] + n * dirv[i] + 1) for i in range(6)]
    g = [mp.mpf(shift[6 + j] + n * dirv[6 + j] + 2) for j in range(5)]
    ef = _esym_mp(f)
    eg = _esym_mp(g) + [mp.mpf(0)]
    c = [mp.mpf(0)] * DIM
    c[0] = h_f * ef[6]
    for k in range(1, DIM):
        c[k] = h_g * eg[6 - k] + h_f * ef[6 - k]
    M = mp.zeros(DIM)
    for rr in range(1, DIM):
        M[rr, rr - 1] = mp.mpf(1)
    for rr in range(DIM):
        M[rr, DIM - 1] = c[rr]
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 (cheap)  /  Tier 2 (expensive)
# ─────────────────────────────────────────────────────────────────────────────
def spectral_dhat(shift, dirv, z_num, z_den, N):
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den), DIM, N)
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return float("nan"), lam
    return float(-lam[1] / lam[0]), lam


def arith_delta(shift, dirv, z_num, z_den, N, lam1_hint=None):
    if lam1_hint is None or not np.isfinite(lam1_hint):
        lam1_hint, _ = 8.0, None
    digits = abs(lam1_hint) * N / 2.302585
    mp.mp.dps = int(2.6 * digits) + 120

    P = mp.eye(DIM)
    snapN = None
    for n in range(1, 2 * N + 1):
        P = P * build_M_mp(n, shift, dirv, z_num, z_den)
        if n == N:
            snapN = mp.matrix(P)

    best = None
    for i in range(DIM):
        for j in range(DIM):
            if i == j:
                continue
            qn, q2 = snapN[j, 5], P[j, 5]
            if qn == 0 or q2 == 0:
                continue
            err = abs(snapN[i, 5] / qn - P[i, 5] / q2)
            if err == 0:
                continue
            lq = mp.log(abs(qn))
            if lq == 0:
                continue
            d = float(-(1 + mp.log(err) / lq))
            if best is None or d > best:
                best = d
    return best


# ─────────────────────────────────────────────────────────────────────────────
def random_shift(rng, box):
    return [SEED_SHIFT[i] + rng.randint(-box, box) for i in range(11)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--box", type=int, default=3, help="shift half-width per dim")
    ap.add_argument("--topk", type=int, default=40, help="model set size")
    ap.add_argument("--nrand", type=int, default=80, help="random verify set size")
    ap.add_argument("--nrank", type=int, default=200, help="ranker Lyapunov depth")
    ap.add_argument("--nverify", type=int, default=160, help="Tier-2 depth")
    ap.add_argument("--thresh", default="0.0,0.08,0.12",
                    help="comma-separated delta thresholds for 'hit'")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="enrich_results.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    thresholds = [float(t) for t in args.thresh.split(",")]

    # 1-2. pool + cheap ranker
    print(f"Scoring POOL of {args.pool} random 11-D shifts with cheap ranker "
          f"(depth {args.nrank}) ...")
    pool = []
    seen = set()
    while len(pool) < args.pool:
        s = tuple(random_shift(rng, args.box))
        if s in seen:
            continue
        seen.add(s)
        dhat, lam = spectral_dhat(list(s), SEED_DIR, Z_NUM, Z_DEN, args.nrank)
        if np.isfinite(dhat):
            pool.append({"shift": list(s), "dhat": dhat, "lam1": float(lam[0])})
    pool.sort(key=lambda p: p["dhat"], reverse=True)
    dhats = np.array([p["dhat"] for p in pool])
    print(f"  delta_hat range [{dhats.min():.4f}, {dhats.max():.4f}], "
          f"mean {dhats.mean():.4f}")

    # 3. model set (top-K) and random set
    model_set = pool[:args.topk]
    random_set = rng.sample(pool, min(args.nrand, len(pool)))

    # 4. Tier-2 verification
    def verify(group, label):
        print(f"\nVerifying {len(group)} {label} points with Tier-2 "
              f"(depth {args.nverify}) ...")
        for i, p in enumerate(group):
            da = arith_delta(p["shift"], SEED_DIR, Z_NUM, Z_DEN,
                             args.nverify, p["lam1"])
            p["delta_arith"] = da
            print(f"  [{label} {i+1}/{len(group)}] dhat={p['dhat']:+.4f} "
                  f"darith={da:+.4f}" if da is not None else
                  f"  [{label} {i+1}/{len(group)}] dhat={p['dhat']:+.4f} "
                  f"darith=None")
        return [p for p in group if p.get("delta_arith") is not None]

    model_v = verify(model_set, "MODEL")
    random_v = verify(random_set, "RANDOM")

    # 5. enrichment
    print("\n" + "=" * 64)
    print("ENRICHMENT REPORT")
    print("=" * 64)
    print(f"  ranker depth N={args.nrank}, verify depth N={args.nverify}, "
          f"box=±{args.box}, pool={args.pool}")
    report = {"args": vars(args), "thresholds": {}}
    for t in thresholds:
        m_hit = np.mean([p["delta_arith"] > t for p in model_v]) if model_v else 0.0
        r_hit = np.mean([p["delta_arith"] > t for p in random_v]) if random_v else 0.0
        # Laplace-smoothed enrichment to avoid div-by-zero
        ef = ((np.sum([p["delta_arith"] > t for p in model_v]) + 0.5) /
              (len(model_v) + 0.5)) / \
             ((np.sum([p["delta_arith"] > t for p in random_v]) + 0.5) /
              (len(random_v) + 0.5)) if model_v and random_v else float("nan")
        print(f"  delta>{t:+.2f}:  model hit-rate={m_hit:.3f} "
              f"({int(m_hit*len(model_v))}/{len(model_v)})   "
              f"random hit-rate={r_hit:.3f} "
              f"({int(r_hit*len(random_v))}/{len(random_v)})   "
              f"ENRICHMENT={ef:.1f}x")
        report["thresholds"][str(t)] = {
            "model_hit_rate": float(m_hit), "random_hit_rate": float(r_hit),
            "enrichment": float(ef)}

    best_model = max(model_v, key=lambda p: p["delta_arith"]) if model_v else None
    if best_model:
        print(f"\n  best MODEL hit: delta_arith={best_model['delta_arith']:+.4f}"
              f"  shift={best_model['shift']}")
    report["pool_dhat"] = {"min": float(dhats.min()), "max": float(dhats.max()),
                           "mean": float(dhats.mean())}
    report["model"] = model_v
    report["random"] = random_v
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON -> {args.out}")


if __name__ == "__main__":
    main()
