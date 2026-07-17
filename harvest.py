"""
harvest.py
==========
Long-running DISCOVERY-RATE harvester: how fast does the cheap spectral detector
find NEW positive-delta 6F5 trajectories across the 11-D space, vs brute force?

Brute-force baseline (LUMI):
    ~300 positives / 2.5e9 trajectories  =>  base rate p0 = 1.2e-7
    20,000 GPU-hours                     =>  ~0.015 positives / GPU-hour

Detector: spectral  delta_hat = -lam2/lam1  (float QR, depth N_rank).
Validated sign-accuracy 1.0 vs arithmetic ground truth; its finite-N bias is
NEGATIVE (~-0.009), so  delta_hat > 0  is a CONSERVATIVE certificate of a true
positive-delta trajectory (we under-count, never inflate).

Sampling modes (positives found *all over* the space, not just near known hits):
    local   : shift = SEED ± 2
    broad   : shift = SEED ± 5
    uniform : shift ~ U[-6,6]^11        (blind, like brute force)
dir from a sparse-advancing pool, z from a rational pool.

Designed for an unattended multi-hour run:
  * time-limited (--hours)
  * streams every NEW strong positive to a .jsonl (flushed) — nothing lost
  * periodic atomic checkpoint of the summary .json
  * rolling Tier-2 precision spot-check (a few verifies per checkpoint)
  * bounded memory (reservoir sample + capped save set)
  * background sampling: a random fraction of ALL screened candidates (positive
    AND negative) is streamed to a separate .jsonl with its label, so a level-set
    surrogate (delta=0 boundary) can be trained later -- positives-only data only
    yields a density model, never a boundary.

Usage:
  caffeinate -i python3 harvest.py --hours 8 --nrank 120 \
       --out harvest_8h.json --hits-out harvest_8h_positives.jsonl \
       --background-out harvest_8h_background.jsonl --background-rate 0.001
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from enrich import arith_delta, spectral_dhat, SEED_SHIFT
from discover import DIR_POOL, Z_POOL

PARENT = Path(__file__).resolve().parent.parent
HIT_FILES = ["znegsweep_hits_unique.jsonl", "large_hits_lirec.jsonl",
             "large_hits_positive_delta.jsonl", "large_hits_deep_rt.jsonl",
             "large_hits_self_lf_delta.jsonl"]
BRUTE_BASE_RATE = 1.2e-7              # ~300 / 2.5e9
BRUTE_POS_PER_GPUH = 300 / 20000.0   # 0.015


def load_known():
    known = set()
    for fn in HIT_FILES:
        p = PARENT / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            known.add((tuple(r.get("shift", [])), tuple(r.get("dir", []))))
    return known


def sample(rng, mode):
    if mode == "local":
        shift = [SEED_SHIFT[i] + rng.randint(-2, 2) for i in range(11)]
    elif mode == "broad":
        shift = [SEED_SHIFT[i] + rng.randint(-5, 5) for i in range(11)]
    else:  # uniform / blind
        shift = [rng.randint(-6, 6) for _ in range(11)]
    dirv = list(rng.choice(DIR_POOL))
    zn, zd = rng.choice(Z_POOL)
    return shift, dirv, zn, zd


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--nrank", type=int, default=120)
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="delta_hat cut for counting a positive")
    ap.add_argument("--save-thresh", type=float, default=0.05,
                    help="delta_hat cut for saving a trajectory to disk")
    ap.add_argument("--modes", default="local,broad,uniform")
    ap.add_argument("--checkpoint", type=float, default=120.0,
                    help="seconds between checkpoints")
    ap.add_argument("--spotcheck-per-ckpt", type=int, default=2)
    ap.add_argument("--max-save", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="harvest_8h.json")
    ap.add_argument("--hits-out", default="harvest_8h_positives.jsonl")
    ap.add_argument("--background-out", default="harvest_8h_background.jsonl",
                    help="random sample of ALL screened candidates (pos+neg) "
                         "with label, for level-set surrogate training")
    ap.add_argument("--background-rate", type=float, default=0.001,
                    help="fraction of screened candidates to save as background")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    modes = args.modes.split(",")
    known = load_known()
    t0 = time.time()
    t_end = t0 + args.hours * 3600.0
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    print(f"[{start_str}] HARVEST start: {args.hours}h, detector N={args.nrank}, "
          f"save_thresh={args.save_thresh}")
    print(f"Loaded {len(known)} known (shift,dir) trajectories.\n", flush=True)

    per_mode = {m: {"screened": 0, "pos": 0, "new_saved": 0} for m in modes}
    saved_keys = set()
    saved_count = 0
    reservoir, RES_MAX, res_seen = [], 500, 0
    spot_conf, spot_tot = 0, 0
    best = []          # top trajectories by dhat (bounded)
    BEST_MAX = 200
    bg_saved = 0       # background (pos+neg) records written
    i = 0
    last_ckpt = t0

    hitsf = open(args.hits_out, "a")
    bgf = open(args.background_out, "a")

    def checkpoint(final=False):
        nonlocal last_ckpt, spot_conf, spot_tot
        # rolling Tier-2 spot-check
        nck = (args.spotcheck_per_ckpt * (10 if final else 1))
        if reservoir and nck:
            for p in rng.sample(reservoir, min(nck, len(reservoir))):
                try:
                    da = arith_delta(p["shift"], p["dir"], p["z_num"],
                                     p["z_den"], 120)
                except Exception:
                    da = None
                if da is not None:
                    spot_tot += 1
                    spot_conf += int(da > 0)
        elapsed = time.time() - t0
        screened = sum(m["screened"] for m in per_mode.values())
        pos = sum(m["pos"] for m in per_mode.values())
        new_saved = sum(m["new_saved"] for m in per_mode.values())
        rate = screened / elapsed if elapsed else 0
        pos_rate = pos / elapsed if elapsed else 0
        hr = pos / screened if screened else 0
        prec = spot_conf / spot_tot if spot_tot else float("nan")
        summary = {
            "elapsed_s": round(elapsed, 1),
            "elapsed_h": round(elapsed / 3600, 3),
            "screened": screened, "positives": pos, "new_saved": new_saved,
            "throughput_per_s": round(rate, 1),
            "positives_per_s": round(pos_rate, 2),
            "positives_per_cpu_hour": round(pos_rate * 3600, 0),
            "overall_hit_rate": hr,
            "enrichment_vs_brute": hr / BRUTE_BASE_RATE if hr else 0,
            "speedup_pos_per_hour_vs_brute":
                (pos_rate * 3600) / BRUTE_POS_PER_GPUH if pos_rate else 0,
            "spotcheck_precision": prec,
            "spotcheck_n": spot_tot,
            "background_saved": bg_saved,
            "per_mode": per_mode,
            "best_positives": sorted(best, key=lambda x: -x["dhat"])[:50],
            "args": vars(args),
        }
        atomic_write_json(args.out, summary)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {elapsed/3600:5.2f}h  screened={screened:,} "
              f"({rate:,.0f}/s)  pos={pos:,} ({100*hr:.1f}%)  "
              f"new_saved={new_saved:,}  pos/h={pos_rate*3600:,.0f}  "
              f"enrich={hr/BRUTE_BASE_RATE if hr else 0:,.0f}x  "
              f"prec={prec:.3f}(n={spot_tot})", flush=True)
        last_ckpt = time.time()

    try:
        while time.time() < t_end:
            i += 1
            mode = modes[i % len(modes)]
            shift, dirv, zn, zd = sample(rng, mode)
            try:
                dhat, _ = spectral_dhat(shift, dirv, zn, zd, args.nrank)
            except Exception:
                continue
            per_mode[mode]["screened"] += 1
            # background sample: a random fraction of ALL screened candidates
            # (positive AND negative) so the delta=0 level set can be learned.
            if np.isfinite(dhat) and rng.random() < args.background_rate:
                bg_rec = {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd,
                          "dhat": float(dhat), "mode": mode,
                          "label": int(dhat > args.thresh)}
                bgf.write(json.dumps(bg_rec) + "\n")
                bgf.flush()
                bg_saved += 1
            if not (np.isfinite(dhat) and dhat > args.thresh):
                if time.time() - last_ckpt > args.checkpoint:
                    checkpoint()
                continue
            per_mode[mode]["pos"] += 1
            rec = {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd,
                   "dhat": float(dhat), "mode": mode}
            # reservoir sampling (precision spot-check)
            res_seen += 1
            if len(reservoir) < RES_MAX:
                reservoir.append(rec)
            else:
                j = rng.randint(0, res_seen - 1)
                if j < RES_MAX:
                    reservoir[j] = rec
            # bounded best list
            if len(best) < BEST_MAX:
                best.append(rec)
            elif dhat > best[-1]["dhat"]:
                best.append(rec)
                best.sort(key=lambda x: -x["dhat"])
                del best[BEST_MAX:]
            # stream strong NEW positives to disk
            if dhat > args.save_thresh and saved_count < args.max_save:
                is_new = (tuple(shift), tuple(dirv)) not in known
                key = (tuple(shift), tuple(dirv), zn, zd)
                if is_new and key not in saved_keys:
                    saved_keys.add(key)
                    per_mode[mode]["new_saved"] += 1
                    rec_out = dict(rec, found_at_h=round((time.time()-t0)/3600, 3))
                    hitsf.write(json.dumps(rec_out) + "\n")
                    hitsf.flush()
                    saved_count += 1
            if time.time() - last_ckpt > args.checkpoint:
                checkpoint()
    except KeyboardInterrupt:
        print("\nInterrupted — writing final checkpoint.", flush=True)
    finally:
        checkpoint(final=True)
        hitsf.close()
        bgf.close()

    print(f"\nDONE. Summary -> {args.out}   New positives -> {args.hits_out}"
          f"   Background -> {args.background_out}")


if __name__ == "__main__":
    main()
