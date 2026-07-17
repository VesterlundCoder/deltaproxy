"""
verify_independent.py  —  PROOF LADDER V1 (deterministic, non-circular replication)
====================================================================================
A *fully independent* exact computation of the double-depth delta, sharing NO code
with enrich.py (no build_M_float / build_M_mp / lyapunov_spectrum, no mpmath matrix).

Independence (to break circularity):
  * elementary symmetric polynomials via integer POLYNOMIAL CONVOLUTION  prod (t+r_i)
    (different algorithm from enrich._esym_*),
  * companion matrix built with pure Python `int` entries (the integer-cleared gauge
    h_f=-z_num, h_g=z_den makes EVERY entry an integer),
  * matrix product done by a hand-written integer matmul,
  * delta from EXACT integer cross-products  cross = p_N q_2N - p_2N q_N,
  * logarithms of the resulting big integers via a top-53-bit mantissa (pure math.log),
    so mpmath is never used at all.

If this reproduces enrich.arith_delta (which uses mpmath + a different construction)
on the confirmed hits, the V2 exact result is independently replicated => V1 passed.

Usage:
  python3 verify_independent.py --in verified_positives.jsonl --nverify 120 \
      --workers 8 --out v1_independent.jsonl --summary v1_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

DIM = 6


# ── independent elementary symmetric polynomials: coeffs of prod_i (t + r_i) ──
def esym_int(roots):
    """Return [e0, e1, ..., e_k] where prod (t + r_i) = sum e_j t^{k-j}.
    Computed by integer polynomial convolution (independent of enrich._esym_*)."""
    poly = [1]                      # represents constant polynomial 1
    for r in roots:
        # multiply poly(t) by (t + r): convolution with [1, r]
        new = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new[i] += c             # t * c
            new[i + 1] += c * r     # r * c
        poly = new
    # poly = [coeff t^k, coeff t^{k-1}, ..., coeff t^0] = [e0, e1, ..., e_k]
    return poly


def build_M_int(n, shift, dirv, z_num, z_den):
    """6x6 integer companion matrix for step n (independent construction)."""
    h_f, h_g = -z_num, z_den
    f = [shift[i] + n * dirv[i] + 1 for i in range(6)]      # 6 f-roots
    g = [shift[6 + j] + n * dirv[6 + j] + 2 for j in range(5)]  # 5 g-roots
    ef = esym_int(f)                 # length 7: e0..e6
    eg = esym_int(g) + [0]           # length 7: e0..e5 then pad to index 6
    c = [0] * DIM
    c[0] = h_f * ef[6]
    for k in range(1, DIM):
        c[k] = h_g * eg[6 - k] + h_f * ef[6 - k]
    M = [[0] * DIM for _ in range(DIM)]
    for rr in range(1, DIM):
        M[rr][rr - 1] = 1
    for rr in range(DIM):
        M[rr][DIM - 1] = c[rr]
    return M


def matmul_int(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(m):
            a = Ai[k]
            if a:
                Bk = B[k]
                for j in range(p):
                    Ci[j] += a * Bk[j]
    return C


def log_bigint(n):
    """Natural log of a positive Python int to ~double precision, no mpmath."""
    n = abs(int(n))
    if n == 0:
        return float("-inf")
    b = n.bit_length()
    shift = max(0, b - 53)
    return math.log(n >> shift) + shift * math.log(2.0)


def independent_delta(shift, dirv, z_num, z_den, N):
    """Exact double-depth delta via pure-integer matrix products and big-int logs."""
    P = [[1 if i == j else 0 for j in range(DIM)] for i in range(DIM)]  # identity
    snapN = None
    for n in range(1, 2 * N + 1):
        P = matmul_int(P, build_M_int(n, shift, dirv, z_num, z_den))
        if n == N:
            snapN = [row[:] for row in P]
    best = None
    for i in range(DIM):
        for j in range(DIM):
            if i == j:
                continue
            pn, qn = snapN[i][DIM - 1], snapN[j][DIM - 1]
            p2, q2 = P[i][DIM - 1], P[j][DIM - 1]
            if qn == 0 or q2 == 0:
                continue
            cross = pn * q2 - p2 * qn          # exact integer numerator of the error
            if cross == 0:
                continue
            # log|err| = log|cross| - log|qn| - log|q2|
            log_err = log_bigint(cross) - log_bigint(qn) - log_bigint(q2)
            log_q = log_bigint(qn)
            if log_q == 0:
                continue
            d = -(1.0 + log_err / log_q)
            if best is None or d > best:
                best = d
    return best


def _one(args):
    rec, N = args
    out = {"shift": rec["shift"], "dir": rec["dir"],
           "z_num": rec["z_num"], "z_den": rec["z_den"],
           "delta_arith": rec.get("delta_arith")}
    try:
        out["delta_independent"] = independent_delta(
            rec["shift"], rec["dir"], rec["z_num"], rec["z_den"], N)
    except Exception as e:
        out["delta_independent"] = None
        out["error"] = str(e)[:120]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="verified_positives.jsonl")
    ap.add_argument("--out", default="v1_independent.jsonl")
    ap.add_argument("--summary", default="v1_summary.json")
    ap.add_argument("--nverify", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |delta_independent - delta_arith| to count as a match")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    # only rows that the mpmath path could compute (have a delta_arith)
    recs = [r for r in recs if r.get("delta_arith") is not None]
    if args.limit:
        recs = recs[:args.limit]
    n = len(recs)
    print(f"V1 independent replication of {n} hits (pure-integer path, "
          f"depth 2N={2*args.nverify}), {args.workers} workers ...", flush=True)

    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(
                _one, [(r, args.nverify) for r in recs], chunksize=4), 1):
            results.append(out)
            if i % 200 == 0 or i == n:
                print(f"  {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    diffs = []
    sign_ok = 0
    both = 0
    ZERO_BAND = 1e-9   # |delta| below this is a delta=0 boundary tie, not a sign conflict
    for r in results:
        di, da = r["delta_independent"], r["delta_arith"]
        if di is None or da is None:
            continue
        both += 1
        diffs.append(abs(di - da))
        if (di > 0) == (da > 0) or (abs(di) < ZERO_BAND and abs(da) < ZERO_BAND):
            sign_ok += 1
    import numpy as np
    diffs = np.array(diffs) if diffs else np.array([0.0])
    matches = int((diffs <= args.tol).sum())

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "n_checked": n,
        "n_both_computed": both,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "median_abs_diff": float(np.median(diffs)),
        "n_match_within_tol": matches,
        "match_rate": matches / both if both else None,
        "sign_agreement": sign_ok / both if both else None,
        "tol": args.tol, "nverify": args.nverify,
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": ("V1 PASSED — independent integer path reproduces mpmath path"
                    if both and matches / both > 0.99 and sign_ok == both
                    else "V1 REVIEW — discrepancies, inspect"),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n============== V1 INDEPENDENT REPLICATION ==============")
    print(f"checked (both paths computed): {both}/{n}")
    print(f"max |Δindep − Δarith| = {diffs.max():.2e}   median = "
          f"{np.median(diffs):.2e}")
    print(f"match within tol {args.tol}: {matches}/{both}  "
          f"({summary['match_rate']:.4f})")
    print(f"sign agreement: {sign_ok}/{both}")
    print(summary["verdict"])
    print(f"-> {args.out} , {args.summary}")


if __name__ == "__main__":
    main()
