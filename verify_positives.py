"""
verify_positives.py
===================
Double-check EVERY harvested positive with REAL (exact) matrix multiplication.

For each saved positive (screened by the float spectral detector delta_hat):
  1. recompute lam1 = leading Lyapunov exponent (float, cheap) to size the precision,
  2. run arith_delta: exact mpmath companion-matrix product P = M(1)..M(2N),
     delta = best over the 30 last-column row-pairs (Tier-2 ground truth).

A positive is CONFIRMED iff delta_arith > thresh (default 0.0). This filters the
~2% float artifacts (e.g. delta_hat ~ 28 from a near-zero lam1) and yields the
true precision of the cheap detector + exact delta for every genuine hit.

Usage:
  python3 verify_positives.py --in harvest_8h_positives.jsonl \
      --out verified_positives.jsonl --summary verify_summary.json \
      --nverify 120 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from enrich import arith_delta, spectral_dhat


def _verify_one(args):
    rec, nverify = args
    shift = rec["shift"]; dirv = rec["dir"]
    zn = rec["z_num"]; zd = rec["z_den"]
    out = {"shift": shift, "dir": dirv, "z_num": zn, "z_den": zd,
           "dhat": rec.get("dhat"), "mode": rec.get("mode")}
    try:
        dh, lam = spectral_dhat(shift, dirv, zn, zd, nverify)
        lam1 = float(lam[0])
    except Exception:
        lam1 = None
    out["lam1"] = lam1
    try:
        da = arith_delta(shift, dirv, zn, zd, nverify, lam1)
    except Exception as e:
        da = None
        out["error"] = str(e)[:120]
    out["delta_arith"] = da
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="harvest_8h_positives.jsonl")
    ap.add_argument("--out", default="verified_positives.jsonl")
    ap.add_argument("--summary", default="verify_summary.json")
    ap.add_argument("--nverify", type=int, default=120,
                    help="Tier-2 depth N (exact product to 2N)")
    ap.add_argument("--thresh", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    n = len(recs)
    print(f"Verifying {n} positives with EXACT matrix products "
          f"(depth 2N={2*args.nverify}), {args.workers} workers ...", flush=True)

    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(
                _verify_one, [(r, args.nverify) for r in recs], chunksize=4), 1):
            results.append(out)
            if i % 100 == 0 or i == n:
                conf = sum(1 for r in results
                           if r["delta_arith"] is not None
                           and r["delta_arith"] > args.thresh)
                print(f"  {i}/{n}  confirmed so far={conf}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # stream all verified records
    with open(args.out, "w") as f:
        for r in sorted(results, key=lambda x: -(x["delta_arith"] or -9)):
            f.write(json.dumps(r) + "\n")

    valid = [r for r in results if r["delta_arith"] is not None]
    conf = [r for r in valid if r["delta_arith"] > args.thresh]
    failed = [r for r in results if r["delta_arith"] is None]
    da = np.array([r["delta_arith"] for r in conf]) if conf else np.array([])
    # artifacts: float screen said strong-positive but exact says <= thresh
    artifacts = [r for r in valid if r["delta_arith"] <= args.thresh]

    def frac(x): return len(x) / n if n else 0.0

    summary = {
        "n_positives": n,
        "n_verified_ok": len(valid),
        "n_compute_failed": len(failed),
        "n_confirmed": len(conf),
        "n_artifacts_arith_nonpositive": len(artifacts),
        "precision_confirmed": len(conf) / len(valid) if valid else None,
        "thresh": args.thresh,
        "nverify_depth": args.nverify,
        "delta_arith_confirmed": {
            "min": float(da.min()) if da.size else None,
            "median": float(np.median(da)) if da.size else None,
            "mean": float(da.mean()) if da.size else None,
            "max": float(da.max()) if da.size else None,
            "ge_0_10": int((da >= 0.10).sum()) if da.size else 0,
            "ge_0_20": int((da >= 0.20).sum()) if da.size else 0,
            "ge_0_30": int((da >= 0.30).sum()) if da.size else 0,
            "ge_0_40": int((da >= 0.40).sum()) if da.size else 0,
        },
        "top20_confirmed": sorted(
            [{"shift": r["shift"], "dir": r["dir"],
              "z_num": r["z_num"], "z_den": r["z_den"],
              "dhat": r["dhat"], "delta_arith": r["delta_arith"]}
             for r in conf], key=lambda x: -x["delta_arith"])[:20],
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ VERIFICATION SUMMARY ================")
    print(f"positives checked      : {n}")
    print(f"computed (valid)       : {len(valid)}   failed: {len(failed)}")
    print(f"CONFIRMED (delta>{args.thresh}) : {len(conf)}  "
          f"=> precision {summary['precision_confirmed']:.4f}")
    print(f"float artifacts (arith<=0): {len(artifacts)}")
    if da.size:
        print(f"confirmed delta: min {da.min():.4f}  median "
              f"{np.median(da):.4f}  max {da.max():.4f}")
        print(f"  >=0.10: {(da>=0.10).sum()}  >=0.20: {(da>=0.20).sum()}  "
              f">=0.30: {(da>=0.30).sum()}  >=0.40: {(da>=0.40).sum()}")
    print(f"verified records -> {args.out}")
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
