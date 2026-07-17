#!/usr/bin/env python3
"""
sample_pslq_lumi.py
===================
Random sampling + PSLQ identification pipeline for the millions of positive-delta
survivors from the LUMI 4F3 (431M) and 5F4 GPU sweeps.

Strategy:
  1. Reservoir-sample N survivors from a (potentially huge) survivor JSONL.
  2. For each sampled survivor, compute exact integer delta (pure-integer double-depth
     walk) to confirm δ > 0.
  3. For confirmed positives, compute the high-precision limit L from the deepest
     convergent and run the full PSLQ battery (pslq_companion.pslq_identify).
  4. Stratify by delta magnitude: sample more heavily from high-δ tail (where PSLQ
     hits are most likely), while still covering the bulk.
  5. Report all non-rational PSLQ hits with full trajectory parameters.

Usage:
  # Sample 5000 from 4F3 LUMI survivors, exact-verify + PSLQ
  python3 sample_pslq_lumi.py --in lumi_survivors_4f3/survivors_all.jsonl \
      --dim 4 --out pslq_4f3_sampled --n-sample 5000 --workers 8 \
      --N 90 --dps 200 --maxcoeff 100000

  # Sample 2000 from 5F4, focus on top-delta band
  python3 sample_pslq_lumi.py --in lumi_survivors_5f4/survivors_all.jsonl \
      --dim 5 --out pslq_5f4_sampled --n-sample 2000 --workers 8 \
      --delta-band 0.05 1.0

  # Use reservoir sampling on a file too large to fit in memory
  python3 sample_pslq_lumi.py --in huge_survivors.jsonl --dim 4 \
      --out pslq_out --n-sample 10000 --reservoir --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg
from pslq_companion import delta_and_limit, pslq_identify


def reservoir_sample(filepath, k, delta_band=None):
    """Reservoir-sample k lines from a (potentially huge) JSONL file.
    If delta_band=(lo, hi) is given, only consider lines with r2 in [lo, hi]."""
    reservoir = []
    n_seen = 0
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            r2 = rec.get("r2", 0)
            if delta_band is not None:
                lo, hi = delta_band
                if r2 < lo or r2 > hi:
                    continue
            n_seen += 1
            if len(reservoir) < k:
                reservoir.append(rec)
            else:
                j = random.randint(0, n_seen - 1)
                if j < k:
                    reservoir[j] = rec
    print(f"  reservoir: scanned {n_seen:,} eligible records, kept {len(reservoir)}")
    return reservoir


def load_and_sample(filepath, k, delta_band=None, use_reservoir=False):
    """Load all records (if feasible) or reservoir-sample."""
    if use_reservoir:
        return reservoir_sample(filepath, k, delta_band)

    records = []
    n_seen = 0
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            r2 = rec.get("r2", 0)
            if delta_band is not None:
                lo, hi = delta_band
                if r2 < lo or r2 > hi:
                    continue
            n_seen += 1
            records.append(rec)

    print(f"  loaded {n_seen:,} eligible records")

    if n_seen <= k:
        print(f"  keeping all {n_seen} (fewer than sample size {k})")
        return records

    # Stratified sampling: 50% from top-delta, 30% from mid, 20% random
    records.sort(key=lambda r: r.get("r2", 0), reverse=True)
    n_top = int(k * 0.5)
    n_mid = int(k * 0.3)
    n_rand = k - n_top - n_mid

    top = records[:n_top]
    mid_start = n_top
    mid_end = min(mid_start + n_mid * 3, len(records))
    mid = random.sample(records[mid_start:mid_end], min(n_mid, mid_end - mid_start))

    remaining = records[mid_end:]
    rand = random.sample(remaining, min(n_rand, len(remaining))) if remaining else []

    sample = top + mid + rand
    random.shuffle(sample)
    print(f"  sampled {len(sample)}: {len(top)} top-delta + {len(mid)} mid + {len(rand)} random")
    return sample


def process_one(args):
    """Exact delta + PSLQ for one sampled survivor."""
    rec, dim, N, dps, maxcoeff, maxsteps = args
    mp.mp.dps = dps

    out = {
        "gid": rec.get("gid"),
        "r2": rec.get("r2"),
        "lam1": rec.get("lam1"),
        "shift": rec["shift"],
        "dir": rec["dir"],
        "z_num": rec["z_num"],
        "z_den": rec["z_den"],
    }

    try:
        d, L, pair = delta_and_limit(
            rec["shift"], rec["dir"],
            rec["z_num"], rec["z_den"], dim, N
        )
        out["delta_exact"] = d
        out["pair"] = list(pair) if pair else None

        if d is not None and d > 0.0 and L is not None:
            out["L"] = mp.nstr(L, 50)
            out["pslq"] = pslq_identify(L, maxcoeff, maxsteps)
        else:
            out["L"] = None
            out["pslq"] = []
    except Exception as e:
        out["delta_exact"] = None
        out["L"] = None
        out["pslq"] = []
        out["error"] = str(e)[:200]

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Sample + PSLQ identify from LUMI positive-delta survivors",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="infile", required=True, help="survivor JSONL file")
    ap.add_argument("--dim", type=int, required=True, help="CMF dimension (4 for 4F3, 5 for 5F4)")
    ap.add_argument("--out", required=True, help="output prefix (writes _results.jsonl, _hits.jsonl)")
    ap.add_argument("--n-sample", type=int, default=5000, help="number to sample")
    ap.add_argument("--workers", type=int, default=8, help="parallel workers")
    ap.add_argument("--N", type=int, default=90, help="exact walk depth (double-depth = 2N)")
    ap.add_argument("--dps", type=int, default=200, help="mpmath decimal precision")
    ap.add_argument("--maxcoeff", type=int, default=100000, help="PSLQ max coefficient")
    ap.add_argument("--maxsteps", type=int, default=20000, help="PSLQ max steps")
    ap.add_argument("--delta-band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"), help="only sample r2 in [LO, HI]")
    ap.add_argument("--reservoir", action="store_true",
                    help="use reservoir sampling (for files too large to load)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    results_path = args.out + "_results.jsonl"
    hits_path = args.out + "_hits.jsonl"
    summary_path = args.out + "_summary.json"

    # Phase 1: Sample
    print(f"Phase 1: Sampling {args.n_sample} from {args.infile}")
    t0 = time.time()
    sample = load_and_sample(
        args.infile, args.n_sample,
        delta_band=args.delta_band,
        use_reservoir=args.reservoir
    )
    t1 = time.time()
    print(f"  sampling done in {t1-t0:.1f}s, {len(sample)} records")

    if not sample:
        print("No records to process. Exiting.")
        return

    # Phase 2: Exact delta + PSLQ
    print(f"\nPhase 2: Exact delta + PSLQ on {len(sample)} samples "
          f"with {args.workers} workers (dim={args.dim}, N={args.N}, dps={args.dps})")

    task_args = [(r, args.dim, args.N, args.dps, args.maxcoeff, args.maxsteps)
                 for r in sample]

    rf = open(results_path, "w")
    hf = open(hits_path, "w")

    total_pos = 0
    total_pslq = 0
    total_nonrat = 0
    t2 = time.time()

    with Pool(args.workers) as pool:
        for i, result in enumerate(
                pool.imap_unordered(process_one, task_args, chunksize=4), 1):
            de = result.get("delta_exact")
            rf.write(json.dumps(result) + "\n")
            rf.flush()

            if de is not None and de > 0.0:
                total_pos += 1
                pslq_hits = result.get("pslq", [])
                if pslq_hits:
                    total_pslq += len(pslq_hits)
                    for h in pslq_hits:
                        if h.get("type") != "rational":
                            total_nonrat += 1
                            hf.write(json.dumps(result) + "\n")
                            hf.flush()
                            print(f"  *** PSLQ HIT [{h['type']}]: "
                                  f"delta={de:+.4f} "
                                  f"z={result['z_num']}/{result['z_den']} "
                                  f"L={result.get('L','?')[:30]} "
                                  f"coeffs={h['coeffs']}")

            if i % 200 == 0:
                el = time.time() - t2
                rate = i / el if el else 0
                print(f"  processed {i}/{len(sample)}  "
                      f"pos_delta={total_pos}  "
                      f"pslq_hits={total_pslq}  "
                      f"non_rational={total_nonrat}  "
                      f"({rate:.1f}/s)")

    rf.close()
    hf.close()
    t3 = time.time()

    # Summary
    summary = {
        "infile": args.infile,
        "dim": args.dim,
        "n_sampled": len(sample),
        "n_positive_delta": total_pos,
        "n_pslq_hits_total": total_pslq,
        "n_pslq_non_rational": total_nonrat,
        "positive_rate": total_pos / max(len(sample), 1),
        "pslq_rate": total_pslq / max(total_pos, 1),
        "N": args.N,
        "dps": args.dps,
        "maxcoeff": args.maxcoeff,
        "elapsed_phase2_s": t3 - t2,
        "results_file": results_path,
        "hits_file": hits_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Sampled:            {len(sample):,}")
    print(f"  Positive delta:     {total_pos:,} ({total_pos/max(len(sample),1)*100:.1f}%)")
    print(f"  PSLQ hits (total):  {total_pslq:,}")
    print(f"  PSLQ hits (non-rat):{total_nonrat:,}")
    print(f"  Phase 2 time:       {t3-t2:.1f}s ({len(sample)/(t3-t2):.1f}/s)")
    print(f"  Results -> {results_path}")
    print(f"  Hits ->    {hits_path}")
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
