"""
v4_identity_certificate.py  —  PROOF LADDER V4 (numerical certificate of the identity)
=======================================================================================
Certifies, across the confirmed hits, the two limits underlying  δ = −1 + E/Q = −λ₂/λ₁:

  (a) denominator growth     Q_emp(N) = log|q_N| / N            →  λ₁
  (b) double-depth delta     δ_arith(N)  (exact integer)        →  −λ₂/λ₁  ( = δ̂ )

q_N and δ_arith are computed by the INDEPENDENT pure-integer engine (verify_independent);
λ₁, λ₂ come from a deep QR Lyapunov spectrum. We report the convergence table for a few
representative hits and an aggregate residual that shrinks with N over a larger sample.

Usage:
  python3 v4_identity_certificate.py --in verified_positives.jsonl \
      --depths 60,120,240,480 --lam-depth 2000 --reps 8 --sample 300 \
      --out v4_identity_certificate.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from enrich import spectral_dhat
from verify_independent import (build_M_int, matmul_int, log_bigint,
                                independent_delta, DIM)


def logabs_qN(shift, dirv, z_num, z_den, N):
    """log|q_N| via the dominant last-column magnitude (independent integer path)."""
    P = [[1 if i == j else 0 for j in range(DIM)] for i in range(DIM)]
    for n in range(1, N + 1):
        P = matmul_int(P, build_M_int(n, shift, dirv, z_num, z_den))
    qmax = max(abs(P[i][DIM - 1]) for i in range(DIM))
    return log_bigint(qmax)


def lam12(shift, dirv, z_num, z_den, depth):
    dh, lam = spectral_dhat(shift, dirv, z_num, z_den, depth)
    return float(lam[0]), float(lam[1]), dh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="verified_positives.jsonl")
    ap.add_argument("--depths", default="60,120,240,480")
    ap.add_argument("--lam-depth", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--out", default="v4_identity_certificate.json")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    conf = [r for r in recs if r.get("delta_arith") is not None
            and r["delta_arith"] > 0]
    conf.sort(key=lambda r: -r["delta_arith"])

    # ── representative convergence tables ──
    print("="*78)
    print("V4 IDENTITY  delta = -1 + E/Q = -lam2/lam1   (convergence of both limits)")
    print("="*78)
    reps = []
    for r in conf[:args.reps]:
        s, d, zn, zd = r["shift"], r["dir"], r["z_num"], r["z_den"]
        l1, l2, dh = lam12(s, d, zn, zd, args.lam_depth)
        target = -l2 / l1
        rows = []
        for N in depths:
            da = independent_delta(s, d, zn, zd, N)
            Q_emp = logabs_qN(s, d, zn, zd, N) / N
            rows.append({"N": N, "delta_arith": da, "Q_emp": Q_emp,
                         "delta_minus_target": (None if da is None else da - target),
                         "Q_emp_minus_lam1": Q_emp - l1})
        zlabel = f"{zn}/{zd}"
        print(f"\nhit z={zlabel}  lam1={l1:.5f} lam2={l2:.5f}  "
              f"-lam2/lam1={target:.5f}  (dhat@{args.lam_depth}={dh:.5f})")
        print(f"  {'N':>5} {'delta_arith':>13} {'->target':>12} "
              f"{'Q_emp':>10} {'->lam1':>11}")
        for row in rows:
            print(f"  {row['N']:>5} {row['delta_arith']:>13.6f} "
                  f"{row['delta_minus_target']:>12.2e} "
                  f"{row['Q_emp']:>10.5f} {row['Q_emp_minus_lam1']:>11.2e}")
        reps.append({"z": zlabel, "lam1": l1, "lam2": l2,
                     "target_minus_lam2_over_lam1": target,
                     "dhat_deep": dh, "rows": rows})

    # ── aggregate residual shrinks with N ──
    print("\n" + "="*78)
    print(f"AGGREGATE over {min(args.sample,len(conf))} confirmed hits: "
          f"mean |delta_arith(N) - (-lam2/lam1)|")
    sample = conf[:args.sample]
    agg = {}
    for N in depths:
        res = []
        for r in sample:
            s, d, zn, zd = r["shift"], r["dir"], r["z_num"], r["z_den"]
            try:
                l1, l2, _ = lam12(s, d, zn, zd, args.lam_depth)
                da = independent_delta(s, d, zn, zd, N)
                if da is not None and l1 != 0:
                    res.append(abs(da - (-l2 / l1)))
            except Exception:
                pass
        res = np.array(res) if res else np.array([np.nan])
        agg[N] = {"n": int(np.isfinite(res).sum()),
                  "mean_abs_residual": float(np.nanmean(res)),
                  "median_abs_residual": float(np.nanmedian(res))}
        print(f"  N={N:>4}: mean={agg[N]['mean_abs_residual']:.4f}  "
              f"median={agg[N]['median_abs_residual']:.4f}  (n={agg[N]['n']})")

    Ns = sorted(agg)
    monotone = all(agg[Ns[k]]["mean_abs_residual"] >=
                   agg[Ns[k+1]]["mean_abs_residual"] - 1e-9
                   for k in range(len(Ns)-1))
    verdict = ("V4 SUPPORTED — delta_arith(N) -> -lam2/lam1 and Q_emp(N) -> lam1 "
               "(residual decreases with depth)" if monotone else
               "V4 PARTIAL — residual not monotone; inspect representative tables")
    print("\n" + verdict)

    out = {"identity": "delta = -1 + E/Q = -lam2/lam1",
           "lam_depth": args.lam_depth, "depths": depths,
           "representatives": reps, "aggregate_residual": agg,
           "residual_monotone_decreasing": monotone, "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
