"""
verify_survivors.py
===================
Stage-2: exact confirmation of Stage-1 GPU survivors.

The GPU proxy (float r_2) is a fast, conservative RANKER. Each survivor is here
re-checked with the non-circular pure-integer double-depth delta
(cmf_generic.independent_delta) -- the same V1 engine used to certify the 6F5
hits -- so the final positives are exact, not float estimates.

Usage:
  python3 verify_survivors.py --in survivors_6f5.jsonl --dim 6 \
      --nverify 100 --workers 6 --thresh 0.0 --out confirmed_6f5.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmf_generic as cg                       # noqa: E402


def _verify(task):
    rec, nverify, dim = task
    try:
        d = cg.independent_delta(rec["shift"], rec["dir"], rec["z_num"],
                                 rec["z_den"], nverify, dim)
    except Exception:
        d = None
    rec = dict(rec)
    rec["delta_exact"] = (None if d is None else float(d))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--nverify", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="confirm rows whose EXACT delta exceeds this")
    ap.add_argument("--out", default="confirmed.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    print(f"[verify] {len(rows)} survivors  dim={args.dim}  "
          f"exact depth 2N={2*args.nverify}")
    tasks = [(r, args.nverify, args.dim) for r in rows]
    with Pool(args.workers) as pool:
        out = pool.map(_verify, tasks)

    confirmed = [r for r in out if r["delta_exact"] is not None
                 and r["delta_exact"] > args.thresh]
    confirmed.sort(key=lambda r: -r["delta_exact"])
    with open(args.out, "w") as f:
        for r in confirmed:
            f.write(json.dumps(r) + "\n")

    # proxy-vs-exact agreement stats
    pairs = [(r["r2"], r["delta_exact"]) for r in out
             if r["delta_exact"] is not None]
    if pairs:
        r2 = np.array([p[0] for p in pairs]); ex = np.array([p[1] for p in pairs])
        sign_agree = np.mean((r2 > 0) == (ex > 0))
        mae = np.mean(np.abs(r2 - ex))
        print(f"  proxy r_2 vs exact: sign-agree={sign_agree:.3f}  MAE={mae:.4f}")
    print(f"  CONFIRMED (exact delta > {args.thresh}): {len(confirmed)}/{len(rows)}"
          f" -> {args.out}")
    for r in confirmed[:10]:
        print(f"    delta_exact={r['delta_exact']:+.4f}  r_2={r['r2']:+.4f}  "
              f"z={r['z_num']}/{r['z_den']}  shift={r['shift']}")


if __name__ == "__main__":
    main()
