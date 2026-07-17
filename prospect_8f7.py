"""
prospect_8f7.py
===============
Find a PRODUCTIVE 8F7 region: screen a large sample with the cheap spectral proxy,
exact-verify the top-ranked candidates, and report the best confirmed positive-delta
8F7 trajectories (to seed a sweep). Also reports the proxy's enrichment on 8F7.

Usage:
  python3 prospect_8f7.py --screen 60000 --topk 80 --box 5 --nrank 120 \
      --nverify 120 --workers 8 --out prospect_8f7.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

import cmf_generic as cg

DIM = 8


def _screen(args):
    shift, dirv, zn, zd, nrank = args
    try:
        dhat, _ = cg.spectral_dhat(shift, dirv, zn, zd, nrank, DIM)
    except Exception:
        return None
    if not np.isfinite(dhat):
        return None
    return {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd,
            "dhat": float(dhat)}


def _verify(args):
    rec, nverify = args
    try:
        d = cg.independent_delta(rec["shift"], rec["dir"], rec["z_num"],
                                 rec["z_den"], nverify, DIM)
    except Exception:
        d = None
    rec = dict(rec)
    rec["delta_exact"] = d
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=int, default=60000)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--box", type=int, default=5)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--nverify", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="prospect_8f7.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    SEED = cg.default_seed(DIM)
    DIRS = cg.dir_pool(DIM)
    ZS = cg.z_pool()
    nsh = cg.nshift_for(DIM)

    # mixed sampling: half broad-around-seed, half fully uniform/blind
    tasks = []
    for k in range(args.screen):
        if k % 2 == 0:
            shift = [SEED[i] + rng.randint(-args.box, args.box) for i in range(nsh)]
        else:
            shift = [rng.randint(-6, 6) for _ in range(nsh)]
        dirv = list(rng.choice(DIRS))
        zn, zd = rng.choice(ZS)
        tasks.append((shift, dirv, zn, zd, args.nrank))

    print(f"[8F7 PROSPECT] screening {args.screen} with proxy ...", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        scored = [r for r in pool.map(_screen, tasks, chunksize=64) if r]
    print(f"  scored {len(scored)} in {time.time()-t0:.0f}s "
          f"({len(scored)/(time.time()-t0):.0f}/s)")
    dh = np.array([r["dhat"] for r in scored])
    print(f"  dhat range [{dh.min():.3f}, {dh.max():.3f}]  "
          f"frac dhat>0: {(dh>0).mean():.4f}  dhat>0.05: {(dh>0.05).mean():.4f}")

    scored.sort(key=lambda r: -r["dhat"])
    top = scored[:args.topk]
    # random control of equal size for enrichment
    control = random.Random(args.seed + 1).sample(scored, min(args.topk, len(scored)))

    print(f"  exact-verifying top-{len(top)} and {len(control)} random control "
          f"(2N={2*args.nverify}) ...", flush=True)
    with Pool(args.workers) as pool:
        top_v = pool.map(_verify, [(r, args.nverify) for r in top])
        ctrl_v = pool.map(_verify, [(r, args.nverify) for r in control])

    def pos_rate(group):
        vals = [r["delta_exact"] for r in group if r["delta_exact"] is not None]
        return (np.mean([v > 0 for v in vals]) if vals else 0.0), len(vals)

    mp_rate, mn = pos_rate(top_v)
    cp_rate, cn = pos_rate(ctrl_v)
    confirmed = sorted([r for r in top_v if r["delta_exact"] is not None
                        and r["delta_exact"] > 0],
                       key=lambda r: -r["delta_exact"])

    print("\n===== 8F7 PROSPECT REPORT =====")
    print(f"  model(top-dhat) positive-rate: {mp_rate:.3f} ({int(mp_rate*mn)}/{mn})")
    print(f"  random control positive-rate:  {cp_rate:.3f} ({int(cp_rate*cn)}/{cn})")
    print(f"  confirmed positive 8F7 trajectories: {len(confirmed)}")
    for r in confirmed[:10]:
        print(f"    delta={r['delta_exact']:.4f}  z={r['z_num']}/{r['z_den']}  "
              f"dhat={r['dhat']:.4f}  shift={r['shift']}  dir={r['dir']}")

    best = confirmed[0] if confirmed else None
    out = {"screened": len(scored), "dhat_max": float(dh.max()),
           "frac_dhat_pos": float((dh > 0).mean()),
           "model_pos_rate": float(mp_rate), "control_pos_rate": float(cp_rate),
           "n_confirmed": len(confirmed),
           "best_seed": ({"shift": best["shift"], "dir": best["dir"],
                          "z_num": best["z_num"], "z_den": best["z_den"],
                          "delta": best["delta_exact"]} if best else None),
           "confirmed": confirmed[:40]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")
    if best:
        print(f"BEST 8F7 SEED: shift={best['shift']} dir={best['dir']} "
              f"z={best['z_num']}/{best['z_den']} delta={best['delta_exact']:.4f}")


if __name__ == "__main__":
    main()
