"""
harvest_generic.py
==================
Dimension-generic discovery harvester (built on cmf_generic). Same design as
harvest.py but works for any hypergeometric order `--dim` (6F5 -> dim 6,
8F7 -> dim 8). Streams positives + background, periodic checkpoints, rolling
exact spot-check via the pure-integer engine.

For a blind null-result sweep it also tracks the GLOBAL maximum dhat seen and a
bounded list of the highest-dhat trajectories REGARDLESS of sign, so that the
absence of positives is documented quantitatively (not just "found nothing").

Usage (8F7, 2 workers launched externally):
  python3 harvest_generic.py --dim 8 --hours 24 --nrank 120 \
      --modes broad,uniform --seed 201 \
      --out sweep_8f7_24h/harvest_w1.json \
      --hits-out sweep_8f7_24h/positives_w1.jsonl \
      --background-out sweep_8f7_24h/background_w1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

import cmf_generic as cg

BRUTE_BASE_RATE = 1.2e-7
BRUTE_POS_PER_GPUH = 300 / 20000.0


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--nverify", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=0.0)
    ap.add_argument("--save-thresh", type=float, default=0.05)
    ap.add_argument("--modes", default="broad,uniform")
    ap.add_argument("--box-broad", type=int, default=5)
    ap.add_argument("--box-uniform", type=int, default=6)
    ap.add_argument("--box-wide", type=int, default=12,
                    help="shift half-width for the blind 'wide' mode")
    ap.add_argument("--rand-dir-prob", type=float, default=0.6,
                    help="prob. of a RANDOM multi-advance direction (wide/uniform)")
    ap.add_argument("--max-advance", type=int, default=3,
                    help="max number of simultaneously advancing roots for random dirs")
    ap.add_argument("--z-max", type=float, default=1.5,
                    help="upper bound on |z| for the z pool (widen to explore)")
    ap.add_argument("--checkpoint", type=float, default=120.0)
    ap.add_argument("--spotcheck-per-ckpt", type=int, default=2)
    ap.add_argument("--max-save", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--out", default="harvest_generic.json")
    ap.add_argument("--hits-out", default="harvest_generic_positives.jsonl")
    ap.add_argument("--background-out", default="harvest_generic_background.jsonl")
    ap.add_argument("--background-rate", type=float, default=0.001)
    args = ap.parse_args()

    dim = args.dim
    nsh = cg.nshift_for(dim)
    SEED = cg.default_seed(dim)
    DIRS = cg.dir_pool(dim)
    ZS = cg.z_pool(z_max=args.z_max)
    rng = random.Random(args.seed)
    modes = args.modes.split(",")

    def random_dir():
        """Random multi-advance direction: 1..max_advance roots advance by +1/+2.
        Covers the direction sub-space (also part of the 15-D search), not just
        the fixed single-advance pool."""
        d = [0] * nsh
        k = rng.randint(1, max(1, args.max_advance))
        for idx in rng.sample(range(nsh), k):
            d[idx] = rng.choice((1, 1, 2))
        return d

    def pick_dir(mode):
        if mode in ("wide", "uniform") and rng.random() < args.rand_dir_prob:
            return random_dir()
        return list(rng.choice(DIRS))

    def sample(mode):
        if mode == "local":
            shift = [SEED[i] + rng.randint(-2, 2) for i in range(nsh)]
        elif mode == "broad":
            shift = [SEED[i] + rng.randint(-args.box_broad, args.box_broad)
                     for i in range(nsh)]
        elif mode == "wide":  # blind, very wide box + random directions
            shift = [rng.randint(-args.box_wide, args.box_wide)
                     for _ in range(nsh)]
        else:  # uniform / blind
            shift = [rng.randint(-args.box_uniform, args.box_uniform)
                     for _ in range(nsh)]
        return shift, pick_dir(mode), *rng.choice(ZS)

    t0 = time.time()
    t_end = t0 + args.hours * 3600.0
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GENERIC HARVEST "
          f"{dim}F{dim-1} (dim={dim}), {args.hours}h, "
          f"N={args.nrank}, modes={modes}", flush=True)

    per_mode = {m: {"screened": 0, "pos": 0, "new_saved": 0} for m in modes}
    reservoir, RES_MAX, res_seen = [], 500, 0
    spot_conf, spot_tot = 0, 0
    best = []          # positives by dhat
    top_dhat = []      # highest dhat regardless of sign (null-result tracking)
    BEST_MAX = 200
    max_dhat_seen = -1e9
    bg_saved = 0
    saved_keys, saved_count = set(), 0
    i = 0
    last_ckpt = t0

    hitsf = open(args.hits_out, "a")
    bgf = open(args.background_out, "a")

    def checkpoint(final=False):
        nonlocal spot_conf, spot_tot, last_ckpt
        nck = args.spotcheck_per_ckpt * (10 if final else 1)
        if reservoir and nck:
            for p in rng.sample(reservoir, min(nck, len(reservoir))):
                try:
                    d = cg.independent_delta(p["shift"], p["dir"], p["z_num"],
                                             p["z_den"], args.nverify, dim)
                except Exception:
                    d = None
                if d is not None:
                    spot_tot += 1
                    spot_conf += int(d > 0)
        elapsed = time.time() - t0
        screened = sum(m["screened"] for m in per_mode.values())
        pos = sum(m["pos"] for m in per_mode.values())
        new_saved = sum(m["new_saved"] for m in per_mode.values())
        rate = screened / elapsed if elapsed else 0
        pos_rate = pos / elapsed if elapsed else 0
        hr = pos / screened if screened else 0
        prec = spot_conf / spot_tot if spot_tot else float("nan")
        summary = {
            "dim": dim, "elapsed_h": round(elapsed / 3600, 3),
            "screened": screened, "positives": pos, "new_saved": new_saved,
            "throughput_per_s": round(rate, 1),
            "positives_per_cpu_hour": round(pos_rate * 3600, 2),
            "overall_hit_rate": hr,
            "enrichment_vs_brute": hr / BRUTE_BASE_RATE if hr else 0,
            "max_dhat_seen": max_dhat_seen,
            "spotcheck_precision": prec, "spotcheck_n": spot_tot,
            "background_saved": bg_saved, "per_mode": per_mode,
            "top_dhat_any_sign": sorted(top_dhat, key=lambda x: -x["dhat"])[:50],
            "best_positives": sorted(best, key=lambda x: -x["dhat"])[:50],
            "args": vars(args),
        }
        atomic_write_json(args.out, summary)
        print(f"[{time.strftime('%H:%M:%S')}] {elapsed/3600:5.2f}h "
              f"screened={screened:,} ({rate:,.0f}/s) pos={pos:,} "
              f"new_saved={new_saved:,} max_dhat={max_dhat_seen:+.4f} "
              f"prec={prec:.3f}(n={spot_tot})", flush=True)
        last_ckpt = time.time()

    try:
        while time.time() < t_end:
            i += 1
            mode = modes[i % len(modes)]
            shift, dirv, zn, zd = sample(mode)
            try:
                dhat, _ = cg.spectral_dhat(shift, dirv, zn, zd, args.nrank, dim)
            except Exception:
                continue
            if not np.isfinite(dhat):
                if time.time() - last_ckpt > args.checkpoint:
                    checkpoint()
                continue
            per_mode[mode]["screened"] += 1
            if dhat > max_dhat_seen:
                max_dhat_seen = float(dhat)
            # track highest-dhat trajectories regardless of sign
            rec = {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd,
                   "dhat": float(dhat), "mode": mode}
            if len(top_dhat) < BEST_MAX:
                top_dhat.append(rec)
            elif dhat > top_dhat[-1]["dhat"]:
                top_dhat.append(rec)
                top_dhat.sort(key=lambda x: -x["dhat"]); del top_dhat[BEST_MAX:]
            # background sample (pos+neg, labeled)
            if rng.random() < args.background_rate:
                bgf.write(json.dumps(dict(rec, label=int(dhat > args.thresh))) + "\n")
                bgf.flush(); bg_saved += 1
            if dhat <= args.thresh:
                if time.time() - last_ckpt > args.checkpoint:
                    checkpoint()
                continue
            per_mode[mode]["pos"] += 1
            res_seen += 1
            if len(reservoir) < RES_MAX:
                reservoir.append(rec)
            else:
                j = rng.randint(0, res_seen - 1)
                if j < RES_MAX:
                    reservoir[j] = rec
            if len(best) < BEST_MAX:
                best.append(rec)
            elif dhat > best[-1]["dhat"]:
                best.append(rec); best.sort(key=lambda x: -x["dhat"]); del best[BEST_MAX:]
            if dhat > args.save_thresh and saved_count < args.max_save:
                key = (tuple(shift), tuple(dirv), zn, zd)
                if key not in saved_keys:
                    saved_keys.add(key)
                    per_mode[mode]["new_saved"] += 1
                    hitsf.write(json.dumps(dict(rec,
                                found_at_h=round((time.time()-t0)/3600, 3))) + "\n")
                    hitsf.flush(); saved_count += 1
            if time.time() - last_ckpt > args.checkpoint:
                checkpoint()
    except KeyboardInterrupt:
        print("\nInterrupted — final checkpoint.", flush=True)
    finally:
        checkpoint(final=True)
        hitsf.close(); bgf.close()

    print(f"\nDONE dim={dim}. Summary -> {args.out}")


if __name__ == "__main__":
    main()
