"""
validate.py
===========
Is the cheap spectral field a VALID surrogate for the expensive arithmetic
ground truth?  (Operationalises the success criteria: spectral consistency,
depth-stability / Jaccard, enrichment factor.)

Two expense tiers, validated against each other:

  Tier 2 (ground truth, expensive, rational theta only):
      arithmetic delta from the integer convergent
          delta_N = -1 - log|r_N - r_2N| / log|q_N|
      r = P[i,5]/P[j,5],  q = P[j,5],  best over all 30 row pairs,
      P = M(1)...M(N) built in high-precision mpmath.   (preprint definition)

  Tier 1 (surrogate, cheap):
      spectral delta-hat = -lambda_2/lambda_1 from float QR Lyapunov.

With the chart reference denominator fixed to the z-denominator (d_ref = z_den)
the float and exact integer matrices are IDENTICAL, so this is a clean test of
"does the Lyapunov-gap delta predict the convergent delta?".

Sub-commands:
  python3 validate.py sanity                      # single-point cross-check
  python3 validate.py consistency [...]           # Tier1 vs Tier2 dataset + metrics
  python3 validate.py depth [...]                 # Jaccard stability of Omega+
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
import mpmath as mp

from field3d import build_6f5
from spectral_delta import lyapunov_spectrum

DIM = 6
F_ROOTS = (-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — arithmetic ground-truth delta (high precision, exact integer-cleared)
# ─────────────────────────────────────────────────────────────────────────────
def _esym_mp(vals):
    n = len(vals)
    e = [mp.mpf(0)] * (n + 1)
    e[0] = mp.mpf(1)
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def build_M_mp(n, z_num, z_den, sigma, r):
    h_f = mp.mpf(-z_num)
    h_g = mp.mpf(z_den)
    f = [mp.mpf(x) for x in F_ROOTS]
    gp = mp.mpf(r) / 2
    g = [gp, gp, gp, mp.mpf(sigma) + n + 2, gp]
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


def arith_delta(z_num, z_den, sigma, r, N, dps=None):
    """Tier-2 ground-truth delta via the integer convergent (best row pair)."""
    if dps is None:
        lam = spectral_lyap(z_num, z_den, sigma, r, max(60, N // 4))
        digits = abs(lam[0]) * N / math.log(10.0)
        dps = int(2.6 * digits) + 120
    mp.mp.dps = dps

    P = mp.eye(DIM)
    snapN = None
    for n in range(1, 2 * N + 1):
        P = P * build_M_mp(n, z_num, z_den, sigma, r)
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
# Tier 1 — spectral surrogate (float QR), matched n-range (n = 1..N)
# ─────────────────────────────────────────────────────────────────────────────
def spectral_lyap(z_num, z_den, sigma, r, N):
    z = z_num / z_den
    return lyapunov_spectrum(
        lambda n: build_6f5(n + 1, z, sigma, r, d_ref=z_den), DIM, N)


def spectral_dhat(z_num, z_den, sigma, r, N):
    lam = spectral_lyap(z_num, z_den, sigma, r, N)
    return (-lam[1] / lam[0] if lam[0] != 0 else float("nan")), lam


# ─────────────────────────────────────────────────────────────────────────────
def cmd_sanity(_):
    print("Single-point cross-check (z=7/20, sigma=0, r=0):")
    for N in (100, 200, 400):
        dhat, lam = spectral_dhat(7, 20, 0, 0.0, N)
        da = arith_delta(7, 20, 0, 0.0, N)
        print(f"  N={N:>4}  delta_hat(Tier1)={dhat:+.4f}   "
              f"delta_arith(Tier2)={da:+.4f}   diff={abs(dhat-da):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_consistency(args):
    ks = [int(x) for x in args.ks.split(",")]          # z = k/20
    sigmas = [float(x) for x in args.sigmas.split(",")]
    rs = np.linspace(0.0, args.rmax, args.nr)
    N = args.depth
    z_den = 20

    rows = []
    print(f"Building labeled dataset: {len(ks)*len(sigmas)*len(rs)} rational "
          f"points at N={N} (Tier1 + Tier2)...")
    for k in ks:
        for s in sigmas:
            for r in rs:
                dhat, lam = spectral_dhat(k, z_den, s, float(r), N)
                da = arith_delta(k, z_den, s, float(r), N)
                rows.append({
                    "z_num": k, "z_den": z_den, "z": k / z_den,
                    "sigma": s, "r": float(r), "N": N,
                    "delta_hat": dhat, "delta_arith": da,
                    "E_over_Q": float(-lam[1] / lam[0] + 1),
                    "lambda1": float(lam[0]), "lambda2": float(lam[1]),
                })
                print(f"  z={k}/20 s={s:+.1f} r={r:.2f}  "
                      f"dhat={dhat:+.4f}  darith={da:+.4f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {args.csv}")

    _report_metrics(rows, args.png)


def _report_metrics(rows, png):
    dh = np.array([r["delta_hat"] for r in rows])
    da = np.array([r["delta_arith"] for r in rows])
    ok = np.isfinite(dh) & np.isfinite(da)
    dh, da = dh[ok], da[ok]
    m = len(dh)

    mae = float(np.mean(np.abs(dh - da)))
    bias = float(np.mean(dh - da))
    corr = float(np.corrcoef(dh, da)[0, 1]) if m > 1 else float("nan")
    sign_acc = float(np.mean(np.sign(dh) == np.sign(da)))

    print("\n=== Spectral-consistency metrics (Tier1 vs Tier2) ===")
    print(f"  points              : {m}")
    print(f"  delta MAE           : {mae:.4f}")
    print(f"  mean bias (hat-arith): {bias:+.4f}")
    print(f"  Pearson corr        : {corr:.4f}")
    print(f"  SIGN accuracy       : {sign_acc:.3f}")

    for thr in (0.0, 0.10, 0.15):
        pred = dh > thr
        truth = da > thr
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        fn = int(np.sum(~pred & truth))
        base = float(np.mean(truth))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        enrich = (prec / base) if (base > 0 and np.isfinite(prec)) else float("nan")
        print(f"  -- threshold delta>{thr:.2f}: base_rate={base:.3f} "
              f"precision={prec:.3f} recall={rec:.3f} enrichment={enrich:.2f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(da, dh, s=18, alpha=0.6)
        lim = [min(da.min(), dh.min()), max(da.max(), dh.max())]
        ax[0].plot(lim, lim, "r--", lw=1, label="y=x")
        ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
        ax[0].set_xlabel("delta_arith (Tier2, ground truth)")
        ax[0].set_ylabel("delta_hat (Tier1, spectral surrogate)")
        ax[0].set_title(f"consistency  MAE={mae:.4f}  sign_acc={sign_acc:.2f}")
        ax[0].legend()
        ax[1].hist(dh - da, bins=20)
        ax[1].set_xlabel("delta_hat - delta_arith"); ax[1].set_title("residuals")
        fig.tight_layout(); fig.savefig(png, dpi=130)
        print(f"PNG -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_depth(args):
    """Jaccard stability of Omega+ = {delta_hat>0} across increasing N (cheap)."""
    xs = np.linspace(args.xmin, args.xmax, args.nx)
    ys = np.linspace(args.ymin, args.ymax, args.ny)
    rs = np.linspace(0.0, args.rmax, args.nr)
    Ns = [int(x) for x in args.depths.split(",")]
    z_den = 20

    masks = {}
    for N in Ns:
        pos = []
        for x in xs:
            z = 10.0 ** x
            knum = z * z_den
            for s in ys:
                for r in rs:
                    lam = lyapunov_spectrum(
                        lambda n: build_6f5(n + 1, z, s, float(r), d_ref=z_den),
                        DIM, N)
                    d = -lam[1] / lam[0] if lam[0] != 0 else float("nan")
                    pos.append(d > 0)
        masks[N] = np.array(pos)
        print(f"  N={N:>5}: Omega+ fraction = {masks[N].mean():.3f}")

    print("\n=== Depth-stability (Jaccard of Omega+) ===")
    for a, b in zip(Ns[:-1], Ns[1:]):
        A, B = masks[a], masks[b]
        inter = np.sum(A & B)
        union = np.sum(A | B)
        jac = inter / union if union else 1.0
        agree = float(np.mean(A == B))
        print(f"  J(Omega+_{a}, Omega+_{b}) = {jac:.3f}   "
              f"cell-agreement = {agree:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sanity").set_defaults(func=cmd_sanity)

    c = sub.add_parser("consistency")
    c.add_argument("--ks", default="3,7,11,15")
    c.add_argument("--sigmas", default="-2,0,2")
    c.add_argument("--nr", type=int, default=6)
    c.add_argument("--rmax", type=float, default=2.5)
    c.add_argument("--depth", type=int, default=200)
    c.add_argument("--csv", default="consistency.csv")
    c.add_argument("--png", default="consistency.png")
    c.set_defaults(func=cmd_consistency)

    d = sub.add_parser("depth")
    d.add_argument("--nx", type=int, default=11)
    d.add_argument("--ny", type=int, default=9)
    d.add_argument("--nr", type=int, default=9)
    d.add_argument("--xmin", type=float, default=-1.4)
    d.add_argument("--xmax", type=float, default=-0.2)
    d.add_argument("--ymin", type=float, default=-3.0)
    d.add_argument("--ymax", type=float, default=3.0)
    d.add_argument("--rmax", type=float, default=2.5)
    d.add_argument("--depths", default="200,500,1000,2000")
    d.set_defaults(func=cmd_depth)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
